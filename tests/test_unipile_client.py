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
    async def fake_request(self,*a,**kw):
        assert kw["params"]["account_id"]=="acc123"
        return {"ok":True}
    monkeypatch.setattr(UnipileClient,"_request",fake_request)
    cli = UnipileClient()
    asyncio.run(cli.get_profile("http://x")) 