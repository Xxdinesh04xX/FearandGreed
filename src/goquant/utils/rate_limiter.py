"""
Rate limiting utilities for API calls.
"""

import asyncio
import time
from typing import Dict, Optional
from collections import deque
from dataclasses import dataclass


@dataclass
class RateLimit:
    """Rate limit configuration."""
    requests: int  # Number of requests allowed
    window: int    # Time window in seconds
    

class RateLimiter:
    """
    Asynchronous rate limiter for API calls.
    
    Implements a sliding window rate limiter that tracks request timestamps
    and enforces rate limits per endpoint or service.
    """
    
    def __init__(self):
        """Initialize the rate limiter."""
        self._limits: Dict[str, RateLimit] = {}
        self._requests: Dict[str, deque] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
    
    def add_limit(self, key: str, requests: int, window: int) -> None:
        """
        Add a rate limit for a specific key.
        
        Args:
            key: Identifier for the rate limit (e.g., 'twitter', 'reddit')
            requests: Number of requests allowed
            window: Time window in seconds
        """
        self._limits[key] = RateLimit(requests, window)
        self._requests[key] = deque()
        self._locks[key] = asyncio.Lock()
    
    async def acquire(self, key: str) -> None:
        """
        Acquire permission to make a request.
        
        This method will block until the request can be made without
        violating the rate limit.
        
        Args:
            key: Rate limit key to check
        """
        if key not in self._limits:
            return  # No rate limit configured
        
        async with self._locks[key]:
            limit = self._limits[key]
            requests = self._requests[key]
            current_time = time.time()
            
            # Remove old requests outside the window
            while requests and requests[0] <= current_time - limit.window:
                requests.popleft()
            
            # Check if we need to wait
            if len(requests) >= limit.requests:
                # Calculate wait time until the oldest request expires
                wait_time = requests[0] + limit.window - current_time
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    # Remove the expired request
                    requests.popleft()
            
            # Record this request
            requests.append(current_time)
    
    def get_remaining_requests(self, key: str) -> Optional[int]:
        """
        Get the number of remaining requests for a key.
        
        Args:
            key: Rate limit key to check
            
        Returns:
            Number of remaining requests, or None if no limit is set
        """
        if key not in self._limits:
            return None
        
        limit = self._limits[key]
        requests = self._requests[key]
        current_time = time.time()
        
        # Count requests within the current window
        recent_requests = sum(
            1 for req_time in requests 
            if req_time > current_time - limit.window
        )
        
        return max(0, limit.requests - recent_requests)
    
    def get_reset_time(self, key: str) -> Optional[float]:
        """
        Get the time when the rate limit will reset.
        
        Args:
            key: Rate limit key to check
            
        Returns:
            Unix timestamp when the limit resets, or None if no limit is set
        """
        if key not in self._limits or not self._requests[key]:
            return None
        
        limit = self._limits[key]
        requests = self._requests[key]
        
        if len(requests) < limit.requests:
            return None  # Not at limit
        
        # The limit will reset when the oldest request expires
        return requests[0] + limit.window


# Global rate limiter instance
rate_limiter = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance."""
    return rate_limiter
