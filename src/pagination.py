from src.logging_conf import logger


class PageIterator:
    """
    Yields successive start values for Google Search pagination.
    
    Similar to the n8n "Page" node, this class yields start values (1, 11, 21, ...)
    for paginating through Google Custom Search results.
    """
    
    def __init__(self, limit: int = 100, batch: int = 10):
        """
        Initialize the page iterator.
        
        Args:
            limit: Maximum number of results to fetch (default 100)
            batch: Size of each batch (default 10, Google CSE maximum per request)
        """
        # Google CSE limit: can't get more than 100 results.
        # 'start' can be max 91 if batch is 10 (91 to 100).
        self.effective_limit = min(limit, 100)  # Actual max results we can fetch
        self.batch = min(batch, 10)  # Max 10 per request
        self.current = 0
        logger.debug(f"PageIterator initialized with effective_limit={self.effective_limit}, batch={self.batch}")
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """Return the next start value for Google Search pagination."""
        # Google search API uses 1-based indexing
        next_value = self.current + 1
        
        # Calculate max allowed start index based on Google's limitation
        # Max start value = 101 - batch (e.g., 91 for batch=10)
        max_start_value = 101 - self.batch
        
        # Stop iteration if we've reached the limit or would exceed max start value
        if next_value > self.effective_limit or next_value > max_start_value:
            logger.debug(f"PageIterator completed. Next start {next_value} exceeds limit {self.effective_limit} or max API start {max_start_value}.")
            raise StopIteration
        
        # Increment for next iteration
        self.current += self.batch
        
        logger.debug(f"PageIterator yielding start={next_value}")
        return next_value 