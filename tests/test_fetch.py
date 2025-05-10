import pytest
import asyncio
import os
from unittest.mock import patch, MagicMock, AsyncMock

from src.fetch import fetch_profiles, _make_request

@pytest.fixture
def setup_env():
    """Set up test environment."""
    # Save original env vars
    original_api_key = os.environ.get("GOOGLE_API_KEY")
    original_cx_id = os.environ.get("CX_ID")
    
    # Set test env vars
    os.environ["GOOGLE_API_KEY"] = "test_api_key"
    os.environ["CX_ID"] = "test_cx_id"
    
    yield
    
    # Restore original env vars
    if original_api_key:
        os.environ["GOOGLE_API_KEY"] = original_api_key
    else:
        os.environ.pop("GOOGLE_API_KEY", None)
        
    if original_cx_id:
        os.environ["CX_ID"] = original_cx_id
    else:
        os.environ.pop("CX_ID", None)

@pytest.mark.asyncio
async def test_fetch_profiles(setup_env):
    """Test fetch_profiles function."""
    # Setup mock response
    mock_results = [
        {"link": "https://www.linkedin.com/in/test1/", "title": "Test 1"},
        {"link": "https://www.linkedin.com/in/test2/", "title": "Test 2"}
    ]
    
    with patch("src.fetch._make_request") as mock_make_request:
        mock_make_request.return_value = mock_results
        
        # Run the test
        results = await fetch_profiles("test query", 1)
        
        # Verify results
        assert results == mock_results
        
        # Verify _make_request was called with correct parameters
        mock_make_request.assert_called_once()
        url, params = mock_make_request.call_args[0]
        assert url == "https://www.googleapis.com/customsearch/v1"
        assert params["key"] == "test_api_key"
        assert params["cx"] == "test_cx_id"
        assert params["q"] == "test query"
        assert params["start"] == 1
        assert params["num"] == 10

@pytest.mark.asyncio
async def test_fetch_profiles_missing_env_vars(setup_env):
    """Test fetch_profiles function with missing environment variables."""
    # Clear environment variables
    os.environ.pop("GOOGLE_API_KEY", None)
    
    # Run the test and expect ValueError
    with pytest.raises(ValueError):
        await fetch_profiles("test query", 1)
    
@pytest.mark.asyncio
async def test_make_request_success():
    """Test _make_request with a successful response."""
    # Create test data
    items_data = {
        "items": [
            {"link": "https://www.linkedin.com/in/test1/"},
            {"link": "https://www.linkedin.com/in/test2/"}
        ]
    }
    
    # Setup mock response with AsyncMock for json method
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=items_data)
    
    # Setup session mock
    mock_session_instance = MagicMock()
    mock_session_instance.get.return_value.__aenter__.return_value = mock_response
    
    with patch("aiohttp.ClientSession", return_value=MagicMock()) as mock_session:
        mock_session.return_value.__aenter__.return_value = mock_session_instance
        
        # Run the test
        url = "https://test-url.com"
        params = {"key": "value"}
        results = await _make_request(url, params)
        
        # Verify results
        assert len(results) == 2
        assert results[0]["link"] == "https://www.linkedin.com/in/test1/"
        assert results[1]["link"] == "https://www.linkedin.com/in/test2/"
        
        # Verify the session.get was called with correct parameters
        mock_session_instance.get.assert_called_once_with(url, params=params)

@pytest.mark.asyncio
async def test_make_request_no_results():
    """Test _make_request when no results are returned."""
    # Setup mock response with AsyncMock for json method
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={})  # No items key
    
    # Setup session mock
    mock_session_instance = MagicMock()
    mock_session_instance.get.return_value.__aenter__.return_value = mock_response
    
    with patch("aiohttp.ClientSession", return_value=MagicMock()) as mock_session:
        mock_session.return_value.__aenter__.return_value = mock_session_instance
        
        # Run the test
        results = await _make_request("https://test-url.com", {"key": "value"})
        
        # Verify results is an empty list
        assert results == [] 