import unittest
import asyncio
import os
from unittest.mock import patch, MagicMock

from src.fetch import fetch_profiles, _make_request


class TestFetch(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Mock environment variables
        os.environ["GOOGLE_API_KEY"] = "test_api_key"
        os.environ["CX_ID"] = "test_cx_id"
    
    @patch("src.fetch._make_request")
    def test_fetch_profiles(self, mock_make_request):
        """Test fetch_profiles function."""
        # Setup mock response
        mock_results = [
            {"link": "https://www.linkedin.com/in/test1/", "title": "Test 1"},
            {"link": "https://www.linkedin.com/in/test2/", "title": "Test 2"}
        ]
        mock_make_request.return_value = asyncio.Future()
        mock_make_request.return_value.set_result(mock_results)
        
        # Run the test
        results = asyncio.run(fetch_profiles("test query", 1))
        
        # Verify results
        self.assertEqual(results, mock_results)
        
        # Verify _make_request was called with correct parameters
        mock_make_request.assert_called_once()
        url, params = mock_make_request.call_args[0]
        self.assertEqual(url, "https://www.googleapis.com/customsearch/v1")
        self.assertEqual(params["key"], "test_api_key")
        self.assertEqual(params["cx"], "test_cx_id")
        self.assertEqual(params["q"], "test query")
        self.assertEqual(params["start"], 1)
        self.assertEqual(params["num"], 10)
    
    @patch("src.fetch._make_request")
    def test_fetch_profiles_missing_env_vars(self, mock_make_request):
        """Test fetch_profiles function with missing environment variables."""
        # Clear environment variables
        os.environ.pop("GOOGLE_API_KEY", None)
        
        # Run the test and expect ValueError
        with self.assertRaises(ValueError):
            asyncio.run(fetch_profiles("test query", 1))
        
        # Verify _make_request was not called
        mock_make_request.assert_not_called()
        
        # Restore environment variables for other tests
        os.environ["GOOGLE_API_KEY"] = "test_api_key"
    
    @patch("aiohttp.ClientSession")
    def test_make_request_success(self, mock_session):
        """Test _make_request with a successful response."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json.return_value = asyncio.Future()
        mock_response.json.return_value.set_result({
            "items": [
                {"link": "https://www.linkedin.com/in/test1/"},
                {"link": "https://www.linkedin.com/in/test2/"}
            ]
        })
        
        # Setup session mock
        mock_session_instance = MagicMock()
        mock_session.return_value.__aenter__.return_value = mock_session_instance
        mock_session_instance.get.return_value.__aenter__.return_value = mock_response
        
        # Run the test
        url = "https://test-url.com"
        params = {"key": "value"}
        results = asyncio.run(_make_request(url, params))
        
        # Verify results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["link"], "https://www.linkedin.com/in/test1/")
        self.assertEqual(results[1]["link"], "https://www.linkedin.com/in/test2/")
        
        # Verify the session.get was called with correct parameters
        mock_session_instance.get.assert_called_once_with(url, params=params)
    
    @patch("aiohttp.ClientSession")
    def test_make_request_no_results(self, mock_session):
        """Test _make_request when no results are returned."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json.return_value = asyncio.Future()
        mock_response.json.return_value.set_result({})  # No items key
        
        # Setup session mock
        mock_session_instance = MagicMock()
        mock_session.return_value.__aenter__.return_value = mock_session_instance
        mock_session_instance.get.return_value.__aenter__.return_value = mock_response
        
        # Run the test
        results = asyncio.run(_make_request("https://test-url.com", {"key": "value"}))
        
        # Verify results is an empty list
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main() 