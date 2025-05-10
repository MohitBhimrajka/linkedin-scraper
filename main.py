#!/usr/bin/env python3
import os
import sys
import argparse
import asyncio
from dotenv import load_dotenv

from src.runner import run
from src.logging_conf import logger


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LinkedIn profile collector using Google Custom Search"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=100,
        help="Maximum number of results to fetch per query (default: 100)"
    )
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check for required environment variables
    required_vars = ["GOOGLE_API_KEY", "CX_ID", "GOOGLE_SHEET_ID"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set these variables in your .env file or environment.")
        sys.exit(1)
    
    # Log startup information
    logger.info(f"Starting LinkedIn profile collection (limit={args.limit})")
    
    try:
        # Run the async process
        asyncio.run(run(limit=args.limit))
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error running LinkedIn scraper: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 