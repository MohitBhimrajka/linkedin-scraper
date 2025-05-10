import os
import asyncio
import yaml
from typing import Dict, List

from src.pagination import PageIterator
from src.throttle import Batcher
from src.fetch import fetch_profiles
from src.transform import normalize_results
from src.sheets import append_rows
from src.logging_conf import logger


async def run(limit: int = 100):
    """
    Main runner function to execute the LinkedIn scraping process.
    
    Loads queries, fetches profiles, processes results, and saves to Google Sheets.
    
    Args:
        limit: Maximum number of results to fetch per query
    """
    # Load environment variables if not already loaded
    if "GOOGLE_API_KEY" not in os.environ or "CX_ID" not in os.environ:
        logger.error("Missing required environment variables. Please set GOOGLE_API_KEY and CX_ID.")
        return
    
    # Load queries from YAML configuration
    queries = load_queries()
    if not queries:
        logger.error("No queries found. Please check config/queries.yaml file.")
        return
    
    logger.info(f"Starting LinkedIn profile collection with {len(queries)} queries (limit: {limit})")
    
    # Create Batcher for rate limiting and parallelism
    batcher = Batcher(max_in_flight=10, delay=2)
    
    # Process each query
    all_results = []
    for query in queries:
        query_results = await process_query(query, limit, batcher)
        all_results.extend(query_results)
        logger.info(f"Completed query with {len(query_results)} unique profiles")
    
    # Normalize and deduplicate results
    unique_profiles = normalize_results(all_results)
    
    # Append to Google Sheets
    if unique_profiles:
        logger.info(f"Appending {len(unique_profiles)} profiles to Google Sheets")
        append_rows(unique_profiles)
    else:
        logger.warning("No profiles found to append")
    
    logger.info("LinkedIn profile collection completed successfully")


async def process_query(query: str, limit: int, batcher: Batcher) -> List[Dict]:
    """
    Process a single query with pagination.
    
    Args:
        query: Search query string
        limit: Maximum number of results to fetch
        batcher: Rate limiter instance
        
    Returns:
        List of search result items
    """
    # Create a pagination iterator
    page_iterator = PageIterator(limit=limit)
    
    # Create tasks for each page
    tasks = []
    for start in page_iterator:
        # Use context manager for rate limiting
        tasks.append(fetch_page(query, start, batcher))
    
    # Execute all tasks and gather results
    results = await asyncio.gather(*tasks)
    
    # Flatten list of lists
    flattened_results = [item for sublist in results for item in sublist]
    
    return flattened_results


async def fetch_page(query: str, start: int, batcher: Batcher) -> List[Dict]:
    """
    Fetch a single page of results with rate limiting.
    
    Args:
        query: Search query string
        start: Starting index for pagination
        batcher: Rate limiter instance
        
    Returns:
        List of search result items for this page
    """
    async with batcher:
        return await fetch_profiles(query, start)


def load_queries() -> List[str]:
    """
    Load search queries from YAML configuration.
    
    Returns:
        List of query strings
    """
    config_path = os.path.join("config", "queries.yaml")
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        if not config or "queries" not in config:
            logger.error("Invalid queries.yaml format. Expected 'queries' key.")
            return []
            
        # Extract query strings
        query_items = config["queries"]
        queries = [item["query"] for item in query_items if "query" in item]
        
        logger.info(f"Loaded {len(queries)} queries from configuration")
        return queries
        
    except Exception as e:
        logger.error(f"Error loading queries: {str(e)}")
        return [] 