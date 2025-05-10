import os
import pytest
import asyncio
from src.campaign import run_campaign, CampaignStats
from src.transform import Profile

@pytest.fixture
def sample_profile():
    return Profile(
        linkedin_url="https://linkedin.com/in/test-user",
        title="Test Title",
        first_name="Test",
        last_name="User",
        description="Test description",
        profile_image_url="https://example.com/img.jpg"
    )

@pytest.mark.asyncio
async def test_generate_only_mode(monkeypatch, sample_profile):
    """Test that Generate only mode doesn't call Unipile APIs"""
    
    # Mock the UnipileClient methods to count calls
    api_calls = {"get_profile": 0, "send_invitation": 0}
    
    # Mock craft_messages to avoid Gemini dependency
    def mock_craft_messages(profile_data, recent_post=""):
        return {
            "connection": "Test connection message",
            "comment": "Test comment",
            "followups": ["Follow 1", "Follow 2", "Follow 3"],
            "inmail_subject": "Test subject",
            "inmail_body": "Test body"
        }
    
    # Mock UnipileClient methods
    async def mock_get_profile(self, url):
        api_calls["get_profile"] += 1
        return {"id": "123", "firstName": "Test", "lastName": "User"}
    
    async def mock_send_invitation(self, profile_id, message):
        api_calls["send_invitation"] += 1
        return {"success": True}
    
    async def mock_close(self):
        pass
    
    # Apply mocks
    monkeypatch.setattr("src.personalize.craft_messages", mock_craft_messages)
    monkeypatch.setattr("src.unipile_client.UnipileClient.get_profile", mock_get_profile)
    monkeypatch.setattr("src.unipile_client.UnipileClient.send_invitation", mock_send_invitation)
    monkeypatch.setattr("src.unipile_client.UnipileClient.close", mock_close)
    
    # Run campaign in Generate only mode
    stats = await run_campaign([sample_profile], mode="Generate only")
    
    # In Generate mode, we should get profile data but not send invitations
    assert api_calls["get_profile"] == 1
    assert api_calls["send_invitation"] == 0
    assert stats.generated == 1
    assert stats.sent == 0
    assert stats.skipped == 1

@pytest.mark.asyncio
async def test_invite_only_mode(monkeypatch, sample_profile):
    """Test that Invite only mode calls API but doesn't comment"""
    
    # Mock the UnipileClient methods to count calls
    api_calls = {"get_profile": 0, "send_invitation": 0, "comment_post": 0}
    
    # Mock craft_messages to avoid Gemini dependency
    def mock_craft_messages(profile_data, recent_post=""):
        return {
            "connection": "Test connection message",
            "comment": "Test comment",
            "followups": ["Follow 1", "Follow 2", "Follow 3"],
            "inmail_subject": "Test subject",
            "inmail_body": "Test body"
        }
    
    # Mock UnipileClient methods
    async def mock_get_profile(self, url):
        api_calls["get_profile"] += 1
        return {"id": "123", "firstName": "Test", "lastName": "User"}
    
    async def mock_send_invitation(self, profile_id, message):
        api_calls["send_invitation"] += 1
        return {"success": True}
        
    async def mock_recent_posts(self, profile_id, limit=1):
        return [{"id": "post123", "text": "Test post"}]
    
    async def mock_comment_post(self, post_id, message):
        api_calls["comment_post"] += 1
        return {"success": True}
    
    async def mock_close(self):
        pass
    
    # Apply mocks
    monkeypatch.setattr("src.personalize.craft_messages", mock_craft_messages)
    monkeypatch.setattr("src.unipile_client.UnipileClient.get_profile", mock_get_profile)
    monkeypatch.setattr("src.unipile_client.UnipileClient.send_invitation", mock_send_invitation)
    monkeypatch.setattr("src.unipile_client.UnipileClient.recent_posts", mock_recent_posts)
    monkeypatch.setattr("src.unipile_client.UnipileClient.comment_post", mock_comment_post)
    monkeypatch.setattr("src.unipile_client.UnipileClient.close", mock_close)
    
    # Run campaign in Invite only mode
    stats = await run_campaign([sample_profile], mode="Invite only")
    
    # In Invite mode, we should get profile data and send invitations but not comment
    assert api_calls["get_profile"] == 1
    assert api_calls["send_invitation"] == 1
    assert api_calls["comment_post"] == 0
    assert stats.generated == 1
    assert stats.sent == 1
    assert stats.skipped == 0 