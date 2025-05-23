import os
import aiohttp
import datetime
from typing import Dict, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from src.logging_conf import logger
import asyncio


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
    
    # Validate num to ensure we don't exceed CSE limits (max 10 per request, max 100 items total)
    if num > 10:
        logger.warning(f"Requested num={num} exceeds CSE maximum of 10, limiting to 10")
        num = 10
        
    # PageIterator now correctly handles max start values for Google's 100 result limit
    # So we don't need to adjust num here anymore
    
    # Generate explicit date range for past year instead of using 'date:r:1y' shortcut
    today = datetime.datetime.now()
    one_year_ago = today - datetime.timedelta(days=365)
    date_range = f"date:r:{one_year_ago.strftime('%Y%m%d')}:{today.strftime('%Y%m%d')}"
    
    base_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cx_id,
        "q": query,
        "start": start,
        "num": num,
        "sort": date_range, # Explicit date range for past year
        "safe": "active",
        "filter": "0",   # No duplicate content filtering to ensure we get all profiles
        "exactTerms": "linkedin profile",  # Ensure results mention "LinkedIn profile"
        "lr": "lang_en",  # Prioritize English results
        "fields": "items(title,link,snippet,pagemap)" # Optimize response size for efficiency
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
    # Initialize the rate_limited class attribute if it doesn't exist
    if not hasattr(_make_request, "rate_limited"):
        _make_request.rate_limited = False
    
    # Add a cooling-off period if we've had rate limit errors recently
    if _make_request.rate_limited:
        # Add an extra delay if we've been rate limited recently
        logger.info("Adding extra delay after recent rate limit error")
        await asyncio.sleep(5)
        _make_request.rate_limited = False
    
    async with aiohttp.ClientSession() as session:
        # Log the query being sent (mask API key)
        log_params = params.copy()
        if 'key' in log_params:
            log_params['key'] = 'MASKED'
        logger.debug(f"Making API request with params: {log_params}")
        
        async with session.get(url, params=params) as response:
            if response.status == 429:
                logger.warning("Rate limit exceeded (429), retrying with backoff...")
                # Mark that we've been rate limited for future calls
                _make_request.rate_limited = True
                response.raise_for_status()  # Trigger retry
                
            if response.status >= 500:
                logger.warning(f"Server error ({response.status}), retrying with backoff...")
                response.raise_for_status()  # Trigger retry
                
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"API error: {response.status} - {error_text}")
                response.raise_for_status()
            
            data = await response.json()
            
            # Log search information statistics
            search_info = data.get('searchInformation', {})
            total_results = search_info.get('totalResults', '0')
            search_time = search_info.get('searchTime', 0)
            logger.info(f"Search completed in {search_time}s with {total_results} total results")
            
            # Check for spelling corrections and log them
            if 'spelling' in data:
                corrected_query = data.get('spelling', {}).get('correctedQuery', '')
                if corrected_query:
                    logger.info(f"Google suggested spelling correction: '{corrected_query}'")
            
            # Extract the items or return empty list if no results
            items = data.get("items", [])
            
            # Log some basic info about the results
            if items:
                logger.debug(f"Retrieved {len(items)} items for this page")
            else:
                logger.warning("No items found in search results")
                
            return items 