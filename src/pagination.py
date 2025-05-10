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
        self.limit = limit
        self.batch = batch
        self.current = 0
        logger.debug(f"PageIterator initialized with limit={limit}, batch={batch}")
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """Return the next start value for Google Search pagination."""
        # Google search API uses 1-based indexing
        next_value = self.current + 1
        
        # Stop iteration if we've reached the limit
        if next_value > self.limit:
            logger.debug(f"PageIterator completed after reaching limit of {self.limit}")
            raise StopIteration
        
        # Increment for next iteration
        self.current += self.batch
        
        logger.debug(f"PageIterator yielding start={next_value}")
        return next_value 