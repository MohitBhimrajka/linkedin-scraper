import os
import aiohttp
from typing import Dict, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from src.logging_conf import logger


async def fetch_profiles(query: str, start: int, num: int = 10) -> List[Dict]:
    """
    Execute a Google Custom Search for LinkedIn profiles.
    
    Args:
        query: The search query
        start: Starting index for results (1-based)
        num: Number of results to fetch (default 10, Google CSE maximum)
        
    Returns:
        List of search result items from Google CSE API
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    cx_id = os.environ.get("CX_ID")
    
    if not api_key or not cx_id:
        raise ValueError("Missing required environment variables: GOOGLE_API_KEY, CX_ID")
    
    base_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cx_id,
        "q": query,
        "start": start,
        "num": num
    }
    
    logger.debug(f"Fetching profiles with query: {query}, start: {start}")
    
    try:
        result_items = await _make_request(base_url, params)
        logger.info(f"Found {len(result_items)} results for query starting at {start}")
        return result_items
    except Exception as e:
        logger.error(f"Error fetching profiles: {str(e)}")
        return []


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
async def _make_request(url: str, params: Dict) -> List[Dict]:
    """
    Make an HTTP request to the Google CSE API with retry logic.
    
    Args:
        url: The API endpoint URL
        params: Query parameters
        
    Returns:
        List of search result items
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            if response.status == 429:
                logger.warning("Rate limit exceeded (429), retrying with backoff...")
                response.raise_for_status()  # Trigger retry
                
            if response.status >= 500:
                logger.warning(f"Server error ({response.status}), retrying with backoff...")
                response.raise_for_status()  # Trigger retry
                
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"API error: {response.status} - {error_text}")
                response.raise_for_status()
            
            data = await response.json()
            
            # Extract the items or return empty list if no results
            items = data.get("items", [])
            return items 