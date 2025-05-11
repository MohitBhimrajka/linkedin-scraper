import os, httpx, asyncio, tenacity, json
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
        self.session = httpx.AsyncClient(
            timeout=timeout, 
            headers={"X-API-KEY": self.api_key}, 
            follow_redirects=True  # Enable following redirects for 301/302 responses
        )

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
                
                # Try to parse JSON response, but handle case where it's not JSON
                try:
                    json_response = r.json()
                except Exception as json_err:
                    # If JSON decoding fails, log the raw text and handle it appropriately
                    raw_text = r.text
                    logger.error(f"Failed to decode JSON response for {method} {url}. Raw text: {raw_text[:200]}...")
                    
                    # For list endpoints, return an empty list if we expected JSON but got something else
                    if method == "GET" and any(suffix in path for suffix in ["/invitations", "/relations", "/conversations"]):
                        logger.warning(f"Returning empty list for non-JSON response from {path}")
                        return []
                    
                    # For other endpoints, log and return the raw text - the caller will need to handle it
                    logger.warning(f"Returning raw response text for non-JSON response: {str(json_err)}")
                    return {"error": "Non-JSON response", "raw_text": raw_text}
                
                return json_response
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
            
            # Handle different possible response structures
            if isinstance(response, dict):
                # API may return {"items": [...]} structure
                items = response.get("items", [])
                if not isinstance(items, list):
                    logger.warning(f"Expected list for 'items' in posts API but got {type(items)}")
                    return []
                return items
            elif isinstance(response, list):
                # API may return direct list of posts
                return response
            else:
                # Unexpected response type
                logger.warning(f"Unexpected response type from posts API: {type(response)}")
                return []
                
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
        
        # No longer include account_id in query params since it's already in the body
        response = await self._request(
            "POST", 
            "/users/invite", 
            json=body
        )
        
        # Check for the expected success response structure
        # The Unipile API returns {'object': 'UserInvitationSent', 'invitation_id': '...'} on success
        if isinstance(response, dict) and response.get("object") == "UserInvitationSent" and response.get("invitation_id"):
            logger.info(f"Successfully sent invitation to {provider_id}. Invitation ID: {response.get('invitation_id')}")
        else:
            # Also check older API success indicators
            if response.get("status") == 201 or response.get("created") is True:
                logger.info(f"Successfully sent invitation to {provider_id} (legacy response)")
            else:
                logger.warning(f"Invitation to {provider_id} may not have been sent properly: {response}")
            
        return response

    async def comment_post(self, post_id:str, message:str, send_at:Optional[str]=None)->Dict:
        """Comment on a LinkedIn post"""
        body = {
            "account_id": self.account_id, 
            "text": message
        }
        if send_at:
            body["send_at"] = send_at
            
        logger.info(f"Attempting to comment on post_id: {post_id} with message: '{message[:30]}...'")
        
        try:
            response = await self._request(
                "POST", 
                f"/posts/{post_id}/comment",
                json=body
            )
            
            # Log response details
            if isinstance(response, dict):
                logger.info(f"Successfully posted comment to {post_id}. Response: {response}")
            else:
                logger.warning(f"Unexpected response type when commenting on post {post_id}: {type(response)}")
                
            return response
        except Exception as e:
            logger.error(f"Error posting comment to post {post_id}: {str(e)}")
            raise

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
            response_data = await self._request(
                "GET", "/users/invitations",
                params={"direction": "sent", "limit": limit, "account_id": self.account_id}
            )
            
            items_list = []
            if isinstance(response_data, dict):
                if "items" in response_data and isinstance(response_data["items"], list):
                    items_list = response_data["items"]
                    logger.debug(f"Using 'items' list from dictionary response for /users/invitations")
                # Check for the specific unexpected UserProfile response
                elif response_data.get("object") == "UserProfile" and ("public_identifier" in response_data or "provider_id" in response_data):
                    logger.error(f"Unexpected UserProfile object returned from /users/invitations. This may indicate an API issue or misconfiguration if you expected a list of invitations. Response: {str(response_data)[:300]}")
                    # Return empty list as this is not a list of invitations
                    return []
                else:
                    logger.error(f"Expected dict with 'items' list from /users/invitations, but 'items' key is missing or not a list. Response: {str(response_data)[:200]}")
                    return []
            elif isinstance(response_data, list):
                items_list = response_data
                logger.debug(f"Using direct list response for /users/invitations")
            else:
                logger.error(f"Expected dict with 'items' list or a direct list from /users/invitations, got {type(response_data)}. Response: {str(response_data)[:200]}")
                return []
            
            # Normalize response for compatibility with both API versions
            for item in items_list:
                if not isinstance(item, dict):
                    logger.warning(f"Skipping non-dict item in invitations list: {str(item)[:100]}")
                    continue
                if "providerId" in item and "provider_id" not in item:
                    item["provider_id"] = item["providerId"]
                if "connectionState" in item and "connection_state" not in item:
                    item["connection_state"] = item["connectionState"]
            
            return items_list
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTPStatusError after retries fetching sent invitations: {e}. Status: {e.response.status_code if e.response else 'N/A'}")
            if e.response and e.response.status_code >= 500:
                logger.error("Unipile server error (5xx) while fetching sent invitations. Returning empty list.")
            return []
        except Exception as e:
            logger.error(f"Generic error fetching sent invitations: {str(e)}")
            return []

    async def list_relations(self, limit:int=1000)->List[Dict]:
        """Get list of all relations and their connection states"""
        try:
            response_data = await self._request(
                "GET", "/users/relations",
                params={"limit": limit, "account_id": self.account_id}
            )
            
            items_list = [] # Ensure items_list is initialized
            if isinstance(response_data, dict) and "items" in response_data and isinstance(response_data["items"], list):
                items_list = response_data["items"]
                logger.debug(f"Using 'items' list from dictionary response for /users/relations")
            elif isinstance(response_data, list):
                items_list = response_data
                logger.debug(f"Using direct list response for /users/relations")
            else:
                logger.error(f"Expected dict with 'items' list or a direct list from /users/relations, got {type(response_data)}. Response: {str(response_data)[:200]}")
                return []
            
            # Normalize response for compatibility with both API versions
            for item in items_list:
                if not isinstance(item, dict):
                    logger.warning(f"Skipping non-dict item in relations list: {str(item)[:100]}")
                    continue
                if "providerId" in item and "provider_id" not in item:
                    item["provider_id"] = item["providerId"]
                if "connectionState" in item and "connection_state" not in item:
                    item["connection_state"] = item["connectionState"]
            
            return items_list
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTPStatusError after retries fetching relations: {e}. Status: {e.response.status_code if e.response else 'N/A'}")
            if e.response and e.response.status_code >= 500:
                logger.error("Unipile server error (5xx) while fetching relations. Returning empty list.")
            return []
        except Exception as e:
            logger.error(f"Generic error fetching relations: {str(e)}")
            return []

    async def list_conversations(self, limit:int=500)->List[Dict]:
        """Get list of conversations with unread counts and last message timestamps"""
        try:
            response_data = await self._request(
                "GET", "/users/conversations",
                params={"limit": limit, "account_id": self.account_id}
            )
            
            items_list = [] # Ensure items_list is initialized
            if isinstance(response_data, dict) and "items" in response_data and isinstance(response_data["items"], list):
                items_list = response_data["items"]
                logger.debug(f"Using 'items' list from dictionary response for /users/conversations")
            elif isinstance(response_data, list):
                items_list = response_data
                logger.debug(f"Using direct list response for /users/conversations")
            else:
                logger.error(f"Expected dict with 'items' list or a direct list from /users/conversations, got {type(response_data)}. Response: {str(response_data)[:200]}")
                return []
            
            for item in items_list:
                if not isinstance(item, dict):
                    logger.warning(f"Skipping non-dict item in conversations list: {str(item)[:100]}")
                    continue
                if "providerId" in item and "provider_id" not in item:
                    item["provider_id"] = item["providerId"]
            
            return items_list
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTPStatusError after retries fetching conversations: {e}. Status: {e.response.status_code if e.response else 'N/A'}")
            if e.response and e.response.status_code >= 500:
                logger.error("Unipile server error (5xx) while fetching conversations. Returning empty list.")
            return []
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