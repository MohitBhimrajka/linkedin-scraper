import os, httpx, asyncio, tenacity
from typing import Any, Dict, Optional, List
from src.logging_conf import logger

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
        
        # Only add account_id to query params for GET requests
        # For POST/PUT, it should be in the JSON body if needed
        if method.upper() == "GET":
            if "params" not in kw: kw["params"] = {}
            kw["params"].setdefault("account_id", self.account_id)
        elif "json" in kw and isinstance(kw["json"], dict) and "account_id" not in kw["json"]:
            # For POST requests with JSON body, add account_id to the body if it's not already there
            # and if the endpoint isn't already including it in the path
            if not path.startswith("/profiles/") and not path.startswith("/posts/"):
                kw["json"]["account_id"] = self.account_id

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
        path="/profiles/enrich"
        return await self._request("GET", path, params={"url": profile_url})

    async def recent_posts(self, profile_id:str, limit:int=3)->List[Dict]:
        return await self._request("GET", f"/profiles/{profile_id}/posts", params={"limit":limit})

    async def send_invitation(self, profile_id:str, message:str)->Dict:
        body={"recipient_id": profile_id, "account_id": self.account_id, "message": message}
        return await self._request("POST", "/invitations", json=body)

    async def comment_post(self, post_id:str, message:str)->Dict:
        return await self._request("POST", f"/posts/{post_id}/comments", json={"text":message})

    async def send_message(self, conversation_id:str, message:str, send_at:Optional[str]=None)->Dict:
        body={"conversation_id": conversation_id,"text":message}
        if send_at: body["send_at"]=send_at
        return await self._request("POST","/messages", json=body)

    async def send_inmail(self, profile_id:str, subject:str, body:str)->Dict:
        return await self._request("POST","/inmails", json={
            "recipient_id": profile_id, "subject": subject, "body": body})
            
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