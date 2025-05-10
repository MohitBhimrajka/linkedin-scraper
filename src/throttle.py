import asyncio
import time
from src.logging_conf import logger


class Batcher:
    """
    Async context manager for throttling API requests.
    
    Controls parallelism and implements rate limiting with delay between batches.
    Replaces the SplitInBatches → Wait pattern from n8n.
    """
    
    def __init__(self, max_in_flight: int = 10, delay: int = 2):
        """
        Initialize the rate limiter.
        
        Args:
            max_in_flight: Maximum number of concurrent requests
            delay: Seconds to wait between batches
        """
        self.max_in_flight = max_in_flight
        self.delay = delay
        self.semaphore = asyncio.Semaphore(max_in_flight)
        self.last_request_time = 0
        logger.debug(f"Batcher initialized with max_in_flight={max_in_flight}, delay={delay}s")
    
    async def __aenter__(self):
        """Enter the async context manager."""
        await self.semaphore.acquire()
        
        # Calculate and wait for the required delay since the last request
        now = time.time()
        time_since_last = now - self.last_request_time
        
        if time_since_last < self.delay and self.last_request_time > 0:
            wait_time = self.delay - time_since_last
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context manager."""
        self.semaphore.release() 