from src.personalize import craft_messages
import pytest
from unittest.mock import patch, MagicMock
import json
from src.transform import Profile  # Import the Profile class

@pytest.fixture
def mock_gemini_response():
    # Create a mock response that returns a valid JSON
    mock_response = MagicMock()
    mock_response.text = """
    {
        "connection": "Noticed your PM work at Acme. I'm building product analytics tools and would love to connect!",
        "comment": "Great insights on product development!",
        "followups": [
            "Hope you're having a great week! Wanted to share this article on product management.",
            "Just saw this resource on analytics that might interest you given your role at Acme.",
            "Would love to hear your thoughts on our new product metrics. Quick chat sometime?"
        ],
        "inmail_subject": "Question about product management at Acme",
        "inmail_body": "Hi Sara, I noticed your experience as a Product Manager at Acme. I'm working on similar challenges and would love to learn from your approach."
    }
    """
    return mock_response

# Test now works correctly with the prompt template
def test_craft_messages(mock_gemini_response):
    # Create a Profile object instead of a dictionary
    sample_profile = Profile(
        linkedin_url="https://linkedin.com/in/sara-test",
        first_name="Sara",
        title="Product Manager @Acme"
    )
    
    # Mock the Gemini client call
    with patch('src.personalize.client.models.generate_content', return_value=mock_gemini_response):
        out = craft_messages(sample_profile)
    
    # Verify the result
    assert set(out.keys()) == {"connection", "comment", "followups", "inmail_subject", "inmail_body"}
    assert all(out["followups"])
    assert len(out["followups"]) == 3
    assert "Sara" in out["inmail_body"]

def test_craft_messages_fallback():
    """Test the fallback behavior when all API attempts fail"""
    # Create a Profile object instead of a dictionary
    sample_profile = Profile(
        linkedin_url="https://linkedin.com/in/sara-test",
        first_name="Sara",
        title="Product Manager @Acme"
    )
    
    # Create a patch for the entire craft_messages function but allow it to call through to the real implementation
    with patch('src.personalize.client.models.generate_content') as mock_generate:
        # Make all three attempts fail with different types of exceptions
        # Since the function retries 3 times, we need to configure our mock to fail all three times
        mock_generate.side_effect = [
            json.JSONDecodeError("Invalid JSON", "{", 1),
            Exception("API Error"),
            Exception("Network Error")
        ]
        
        # Call the function
        out = craft_messages(sample_profile)
    
    # Verify we got the fallback response
    assert set(out.keys()) == {"connection", "comment", "followups", "inmail_subject", "inmail_body"}
    assert all(out["followups"])
    assert len(out["followups"]) == 3

def test_message_structure():
    """Test that the output from craft_messages has the expected structure."""
    # Create a Profile object instead of a dictionary
    sample_profile = Profile(
        linkedin_url="https://linkedin.com/in/test-profile",
        first_name="Test",
        title="Test Role"
    )
    
    # Create a minimal mock that just returns a simple dictionary with the right structure
    with patch('src.personalize.client.models.generate_content', return_value=MagicMock(
        text=json.dumps({
            "connection": "Test connection message",
            "comment": "Test comment",
            "followups": ["Follow up 1", "Follow up 2", "Follow up 3"],
            "inmail_subject": "Test subject",
            "inmail_body": "Test body"
        })
    )):
        result = craft_messages(sample_profile)
    
    # Check the structure
    assert isinstance(result, dict)
    assert "connection" in result
    assert "comment" in result
    assert "followups" in result
    assert "inmail_subject" in result
    assert "inmail_body" in result
    assert isinstance(result["followups"], list)
    assert len(result["followups"]) == 3 