import os, httpx, asyncio, tenacity
from typing import Any, Dict, Optional, List
from src.logging_conf import logger
from urllib.parse import quote_plus

class UnipileAuthError(RuntimeError): ...
class UnipileClient:
    def __init__(self,
        api_key: Optional[str]=None,
        dsn: Optional[str]=None,
        account_id: Optional[str]=None,
        timeout: float = 15.0):
        # If env vars present use them
        self.api_key = api_key or os.getenv("UNIPILE_API_KEY")
        self.dsn      = dsn or os.getenv("UNIPILE_DSN")
        self.account_id = account_id or os.getenv("UNIPILE_ACCOUNT_ID")
        if not all([self.api_key, self.dsn, self.account_id]):
            raise ValueError("Missing UNIPILE_API_KEY, UNIPILE_DSN or UNIPILE_ACCOUNT_ID")
        self.base = f"https://{self.dsn}/api/v1"
        self.session = httpx.AsyncClient(timeout=timeout, headers={"X-API-KEY": self.api_key})

    async def _request(self, method:str, path:str, **kw)->Any:
        url = f"{self.base}{path}"
        
        # For GET requests, add account_id to query params if not already there
        if method == "GET":
            if "params" not in kw: kw["params"] = {}
            if "account_id" not in kw["params"]:
                kw["params"]["account_id"] = self.account_id
        
        # For POST requests, we'll handle account_id in the specific methods
        # as it depends on the endpoint

        @tenacity.retry(stop=tenacity.stop_after_attempt(3),
                        wait=tenacity.wait_exponential(multiplier=1, min=2, max=8),
                        reraise=True)
        async def do():
            try:
                r = await self.session.request(method, url, **kw)
                if r.status_code==401: raise UnipileAuthError("Bad key / account id")
                r.raise_for_status()
                return r.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error {e.response.status_code} for {method} {url}: {str(e)}")
                if e.response.status_code == 429:
                    logger.warning("Rate limit exceeded. Retrying with backoff...")
                elif e.response.status_code == 404:
                    logger.error(f"Resource not found: {path}")
                elif e.response.status_code >= 500:
                    logger.error(f"Server error from Unipile API")
                raise
            except httpx.RequestError as e:
                logger.error(f"Request error for {method} {url}: {str(e)}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error for {method} {url}: {str(e)}")
                raise
                
        return await do()

    # helper endpoints
    async def get_profile(self, profile_url:str)->Dict:
        """Retrieve a LinkedIn profile by URL or identifier"""
        # Extract the profile identifier from the URL
        ident = profile_url.rstrip('/').split('/')[-1]
        return await self._request(
            "GET", 
            f"/users/{ident}",
            params={"account_id": self.account_id}
        )

    async def recent_posts(self, provider_id: str, limit: int = 5) -> List[Dict]:
        """
        Fetch recent posts from a LinkedIn profile.
        
        Args:
            provider_id: The provider-specific ID for the LinkedIn profile
            limit: Maximum number of posts to retrieve
            
        Returns:
            List of post objects
        """
        try:
            params = {
                "account_id": self.account_id,
                "limit": limit
            }
            
            # Use the proper endpoint format with the provider_id
            response = await self._request(
                "GET", 
                f"/linkedin/users/{provider_id}/posts", 
                params=params
            )
            
            # Ensure we always return a list
            items = response.get("items", [])
            if not isinstance(items, list):
                logger.warning(f"Expected list from posts API but got {type(items)}")
                return []
                
            return items
            
        except Exception as e:
            logger.warning(f"Error fetching posts for {provider_id}: {e}")
            # Return empty list instead of raising to make this non-fatal
            return []

    async def send_invitation(self, provider_id:str, message:str, send_at:Optional[str]=None)->Dict:
        """Send a connection invitation to a LinkedIn profile"""
        body = {
            "provider_id": provider_id,
            "account_id": self.account_id,
            "message": message
        }
        if send_at:
            body["send_at"] = send_at
        return await self._request("POST", "/users/invite", json=body)

    async def comment_post(self, post_id:str, message:str, send_at:Optional[str]=None)->Dict:
        """Comment on a LinkedIn post"""
        body = {
            "account_id": self.account_id, 
            "text": message
        }
        if send_at:
            body["send_at"] = send_at
        return await self._request(
            "POST", 
            f"/posts/{post_id}/comment",
            json=body
        )

    async def send_message(self, conversation_id:str, message:str, send_at:Optional[str]=None)->Dict:
        """Send a message in an existing conversation"""
        body={
            "conversation_id": conversation_id,
            "text": message,
            "account_id": self.account_id
        }
        if send_at: body["send_at"]=send_at
        return await self._request("POST","/messages", json=body)

    async def send_inmail(self, provider_id:str, subject:str, body:str)->Dict:
        """Send an InMail to a LinkedIn profile"""
        return await self._request("POST","/inmails", json={
            "provider_id": provider_id, 
            "subject": subject, 
            "body": body,
            "account_id": self.account_id
        })
    
    async def list_sent_invitations(self, limit:int=500)->List[Dict]:
        """Get list of sent invitations and their statuses"""
        try:
            # Use the working endpoint directly
            return await self._request(
                "GET", "/users/invitations",
                params={"direction": "sent", "limit": limit, "account_id": self.account_id}
            )
        except Exception as e:
            logger.error(f"Error fetching sent invitations: {str(e)}")
            return []

    async def list_relations(self, limit:int=1000)->List[Dict]:
        """Get list of all relations and their connection states"""
        try:
            # Use the working endpoint directly
            return await self._request(
                "GET", "/users/relations",
                params={"limit": limit, "account_id": self.account_id}
            )
        except Exception as e:
            logger.error(f"Error fetching relations: {str(e)}")
            return []

    async def list_conversations(self, limit:int=500)->List[Dict]:
        """Get list of conversations with unread counts and last message timestamps"""
        try:
            # Use the working endpoint directly
            return await self._request(
                "GET", "/users/conversations",
                params={"limit": limit, "account_id": self.account_id}
            )
        except Exception as e:
            logger.error(f"Error fetching conversations: {str(e)}")
            return []
            
    async def close(self):
        """Close the HTTP client session."""
        if self.session:
            await self.session.aclose()
            
    async def __aenter__(self):
        """Support using with async with."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close client when exiting context."""
        await self.close() 