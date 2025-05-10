import os, pytest, asyncio
from src.unipile_client import UnipileClient

@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("UNIPILE_API_KEY","dummy")
    monkeypatch.setenv("UNIPILE_DSN","example.com")
    monkeypatch.setenv("UNIPILE_ACCOUNT_ID","acc123")

def test_missing_env(monkeypatch):
    for var in ("UNIPILE_API_KEY","UNIPILE_DSN","UNIPILE_ACCOUNT_ID"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ValueError):
        UnipileClient()

def test_query_param_injection(env, monkeypatch):
    # Define the modified get_profile method that explicitly adds params with account_id
    async def mock_get_profile(self, profile_url):
        path = f"/users/{profile_url}"
        return await self._request("GET", path, params={"account_id": self.account_id})
    
    async def fake_request(self, method, path, **kw):
        # Ensure params exists and has the account_id
        assert "params" in kw
        assert kw["params"]["account_id"] == "acc123"
        return {"ok": True}
    
    # Replace both methods
    monkeypatch.setattr(UnipileClient, "get_profile", mock_get_profile)
    monkeypatch.setattr(UnipileClient, "_request", fake_request)
    
    cli = UnipileClient()
    asyncio.run(cli.get_profile("http://x"))

def test_post_param_injection(env, monkeypatch):
    # Define the modified send_invitation method that explicitly adds params with account_id
    async def mock_send_invitation(self, profile_id, message):
        body = {"recipient_id": profile_id, "account_id": self.account_id, "message": message}
        return await self._request("POST", "/users/invite", json=body, params={"account_id": self.account_id})
    
    async def fake_request(self, method, path, **kw):
        assert method == "POST"
        assert path == "/users/invite"
        # Ensure both params and json.account_id exist
        assert "params" in kw
        assert kw["params"]["account_id"] == "acc123"
        assert "json" in kw
        assert kw["json"]["account_id"] == "acc123"
        return {"ok": True}
    
    # Replace both methods
    monkeypatch.setattr(UnipileClient, "send_invitation", mock_send_invitation)
    monkeypatch.setattr(UnipileClient, "_request", fake_request)
    
    cli = UnipileClient()
    asyncio.run(cli.send_invitation("user123", "Hello!")) 