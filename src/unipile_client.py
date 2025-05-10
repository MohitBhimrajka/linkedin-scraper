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
            r = await self.session.request(method, url, **kw)
            if r.status_code==401: raise UnipileAuthError("Bad key / account id")
            r.raise_for_status()
            return r.json()
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

    async def recent_posts(self, provider_id:str, limit:int=3)->List[Dict]:
        """Get recent posts from a LinkedIn profile"""
        return await self._request(
            "GET", 
            f"/users/{provider_id}/posts",
            params={"account_id": self.account_id, "limit": limit}
        )

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
        return await self._request(
            "GET", "/linkinvitations",
            params={"direction": "sent", "limit": limit, "account_id": self.account_id}
        )

    async def list_relations(self, limit:int=1000)->List[Dict]:
        """Get list of all relations and their connection states"""
        return await self._request(
            "GET", "/relations",
            params={"limit": limit, "account_id": self.account_id}
        )

    async def list_conversations(self, limit:int=500)->List[Dict]:
        """Get list of conversations with unread counts and last message timestamps"""
        return await self._request(
            "GET", "/conversations",
            params={"limit": limit, "account_id": self.account_id}
        )
            
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