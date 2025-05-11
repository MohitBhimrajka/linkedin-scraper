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
    icp_configs = load_queries()
    if not icp_configs:
        logger.error("No ICP configurations found. Please check config/queries.yaml file.")
        return
    
    logger.info(f"Starting LinkedIn profile collection with {len(icp_configs)} ICP configurations (limit: {limit})")
    
    # Create Batcher for rate limiting and parallelism
    batcher = Batcher(max_in_flight=10, delay=2)
    
    # Process each ICP configuration
    all_results = []
    for icp_config in icp_configs:
        description = icp_config["description"]
        negative_keywords = icp_config.get("negative_keywords", "")
        
        # Optimize the query with Gemini if available
        if "GEMINI_API_KEY" in os.environ:
            from streamlit_app import optimize_query_with_gemini
            query_variations = optimize_query_with_gemini(description, negative_keywords)
            
            # Process each query variation
            variation_results = []
            for query_variation in query_variations:
                query_results = await process_query(query_variation, limit, batcher)
                variation_results.extend(query_results)
                logger.info(f"Completed query variation with {len(query_results)} profiles")
            
            all_results.extend(variation_results)
            logger.info(f"Completed ICP '{description[:30]}...' with {len(variation_results)} total profiles from all variations")
        else:
            # Fallback to using the description directly as the query if Gemini is not available
            query_results = await process_query(description, limit, batcher)
            all_results.extend(query_results)
            logger.info(f"Completed ICP '{description[:30]}...' with {len(query_results)} profiles")
    
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


def load_queries() -> List[Dict]:
    """
    Load search queries from YAML configuration.
    
    Returns:
        List of dictionaries with query descriptions and negative keywords
    """
    config_path = os.path.join("config", "queries.yaml")
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        if not config or "queries" not in config:
            logger.error("Invalid queries.yaml format. Expected 'queries' key.")
            return []
            
        # Extract query items
        query_items = config["queries"]
        
        # Convert older format to newer format if needed
        result_queries = []
        for item in query_items:
            if "query" in item and "description" not in item:
                # Old format using 'query' key
                result_queries.append({
                    "description": item["query"],
                    "negative_keywords": ""
                })
            elif "description" in item:
                # New format
                if "negative_keywords" not in item:
                    item["negative_keywords"] = ""
                result_queries.append(item)
        
        logger.info(f"Loaded {len(result_queries)} ICP configurations from configuration")
        return result_queries
        
    except Exception as e:
        logger.error(f"Error loading queries: {str(e)}")
        return [] 