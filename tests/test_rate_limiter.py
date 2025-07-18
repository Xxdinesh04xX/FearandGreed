"""
Tests for rate limiting functionality.
"""

import pytest
import asyncio
import time
from src.goquant.utils.rate_limiter import RateLimiter


class TestRateLimiter:
    """Test cases for RateLimiter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.rate_limiter = RateLimiter()
    
    def test_add_limit(self):
        """Test adding rate limits."""
        self.rate_limiter.add_limit("test", 10, 60)
        
        # Should have the limit configured
        assert "test" in self.rate_limiter._limits
        assert self.rate_limiter._limits["test"].requests == 10
        assert self.rate_limiter._limits["test"].window == 60
    
    @pytest.mark.asyncio
    async def test_acquire_no_limit(self):
        """Test acquiring without rate limit."""
        # Should not block when no limit is set
        start_time = time.time()
        await self.rate_limiter.acquire("nonexistent")
        end_time = time.time()
        
        # Should complete immediately
        assert end_time - start_time < 0.1
    
    @pytest.mark.asyncio
    async def test_acquire_within_limit(self):
        """Test acquiring within rate limit."""
        self.rate_limiter.add_limit("test", 5, 10)
        
        # Should allow requests within limit
        for i in range(3):
            start_time = time.time()
            await self.rate_limiter.acquire("test")
            end_time = time.time()
            
            # Should not block
            assert end_time - start_time < 0.1
    
    @pytest.mark.asyncio
    async def test_acquire_at_limit(self):
        """Test acquiring at rate limit."""
        self.rate_limiter.add_limit("test", 2, 2)  # 2 requests per 2 seconds
        
        # First two requests should be immediate
        await self.rate_limiter.acquire("test")
        await self.rate_limiter.acquire("test")
        
        # Third request should block
        start_time = time.time()
        await self.rate_limiter.acquire("test")
        end_time = time.time()
        
        # Should have waited at least 1 second
        assert end_time - start_time >= 1.0
    
    def test_get_remaining_requests(self):
        """Test getting remaining requests."""
        self.rate_limiter.add_limit("test", 5, 60)
        
        # Should start with full limit
        remaining = self.rate_limiter.get_remaining_requests("test")
        assert remaining == 5
        
        # Should return None for non-existent limit
        remaining = self.rate_limiter.get_remaining_requests("nonexistent")
        assert remaining is None
    
    def test_get_reset_time(self):
        """Test getting reset time."""
        self.rate_limiter.add_limit("test", 5, 60)
        
        # Should return None when not at limit
        reset_time = self.rate_limiter.get_reset_time("test")
        assert reset_time is None
        
        # Should return None for non-existent limit
        reset_time = self.rate_limiter.get_reset_time("nonexistent")
        assert reset_time is None


if __name__ == "__main__":
    pytest.main([__file__])
