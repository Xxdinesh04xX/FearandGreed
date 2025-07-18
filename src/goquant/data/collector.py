"""
Main data collector that orchestrates data collection from multiple sources.
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..config import Config
from ..utils.logger import get_logger
from ..utils.rate_limiter import get_rate_limiter
from .sources import TwitterCollector, RedditCollector, NewsCollector, FinancialDataCollector


class DataCollector:
    """
    Main data collector that orchestrates data collection from multiple sources.
    
    Manages rate limiting, error handling, and data storage for all data sources.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the data collector.

        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.rate_limiter = get_rate_limiter()

        # Initialize data source collectors
        self.collectors = {}

        # Database manager will be injected
        self.db_manager: Optional[DatabaseManager] = None

        # Set up rate limits
        self._setup_rate_limits()

        self.logger.info("DataCollector initialized")

    def set_database_manager(self, db_manager) -> None:
        """Set the database manager for data storage."""
        self.db_manager = db_manager
    
    def _setup_rate_limits(self) -> None:
        """Set up rate limits for different data sources."""
        # Twitter rate limits
        if self.config.twitter_bearer_token:
            self.rate_limiter.add_limit(
                'twitter',
                self.config.twitter_rate_limit_requests,
                self.config.twitter_rate_limit_window
            )
        
        # Reddit rate limits
        if self.config.reddit_client_id:
            self.rate_limiter.add_limit(
                'reddit',
                self.config.reddit_rate_limit_requests,
                self.config.reddit_rate_limit_window
            )
        
        # News API rate limits
        if self.config.news_api_key:
            self.rate_limiter.add_limit(
                'news',
                self.config.news_rate_limit_requests,
                self.config.news_rate_limit_window
            )
    
    async def initialize(self) -> None:
        """Initialize all data source collectors."""
        try:
            # Initialize Twitter collector
            if self.config.twitter_bearer_token:
                self.collectors['twitter'] = TwitterCollector(self.config)
                await self.collectors['twitter'].initialize()
                self.logger.info("Twitter collector initialized")
            
            # Initialize Reddit collector
            if self.config.reddit_client_id:
                self.collectors['reddit'] = RedditCollector(self.config)
                await self.collectors['reddit'].initialize()
                self.logger.info("Reddit collector initialized")
            
            # Initialize News collector
            if self.config.news_api_key:
                self.collectors['news'] = NewsCollector(self.config)
                await self.collectors['news'].initialize()
                self.logger.info("News collector initialized")
            
            # Initialize Financial data collector (always initialize - Yahoo Finance doesn't need API key)
            self.collectors['financial'] = FinancialDataCollector(self.config)
            await self.collectors['financial'].initialize()
            self.logger.info("Financial data collector initialized")
            
            self.logger.info(f"Initialized {len(self.collectors)} data collectors")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize data collectors: {e}")
            raise
    
    async def collect_all(self) -> Dict[str, int]:
        """
        Collect data from all available sources.
        
        Returns:
            Dictionary with collection counts per source
        """
        results = {}
        tasks = []
        
        # Create collection tasks for each source
        for source_name, collector in self.collectors.items():
            task = asyncio.create_task(
                self._collect_from_source(source_name, collector),
                name=f"collect_{source_name}"
            )
            tasks.append(task)
        
        # Wait for all collection tasks to complete
        if tasks:
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(completed_results):
                source_name = list(self.collectors.keys())[i]
                if isinstance(result, Exception):
                    self.logger.error(f"Error collecting from {source_name}: {result}")
                    results[source_name] = 0
                else:
                    results[source_name] = result
        
        total_collected = sum(results.values())
        self.logger.info(f"Collected {total_collected} total records: {results}")
        
        return results
    
    async def _collect_from_source(self, source_name: str, collector) -> int:
        """
        Collect data from a specific source with rate limiting.
        
        Args:
            source_name: Name of the data source
            collector: Collector instance
            
        Returns:
            Number of records collected
        """
        try:
            # Apply rate limiting
            await self.rate_limiter.acquire(source_name)
            
            # Collect data
            data = await collector.collect()

            if not data:
                return 0

            # Store data in database
            if self.db_manager:
                try:
                    stored_ids = await self.db_manager.insert_raw_data(data)
                    count = len(stored_ids)
                    self.logger.debug(f"Stored {count} records from {source_name} in database")
                except Exception as e:
                    self.logger.error(f"Failed to store data from {source_name}: {e}")
                    count = 0
            else:
                count = len(data) if isinstance(data, list) else 1
                self.logger.debug(f"Collected {count} records from {source_name} (no database)")

            return count
            
        except Exception as e:
            self.logger.error(f"Error collecting from {source_name}: {e}")
            return 0
    
    async def collect_for_symbols(self, symbols: List[str]) -> Dict[str, Dict[str, int]]:
        """
        Collect data for specific symbols.
        
        Args:
            symbols: List of symbols to collect data for
            
        Returns:
            Nested dictionary with counts per source per symbol
        """
        results = {}
        
        for symbol in symbols:
            results[symbol] = {}
            tasks = []
            
            # Create collection tasks for each source
            for source_name, collector in self.collectors.items():
                if hasattr(collector, 'collect_for_symbol'):
                    task = asyncio.create_task(
                        self._collect_symbol_from_source(source_name, collector, symbol),
                        name=f"collect_{source_name}_{symbol}"
                    )
                    tasks.append(task)
            
            # Wait for all tasks for this symbol
            if tasks:
                completed_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for i, result in enumerate(completed_results):
                    source_name = list(self.collectors.keys())[i]
                    if isinstance(result, Exception):
                        self.logger.error(f"Error collecting {symbol} from {source_name}: {result}")
                        results[symbol][source_name] = 0
                    else:
                        results[symbol][source_name] = result
        
        return results
    
    async def _collect_symbol_from_source(self, source_name: str, collector, symbol: str) -> int:
        """
        Collect data for a specific symbol from a source.
        
        Args:
            source_name: Name of the data source
            collector: Collector instance
            symbol: Symbol to collect data for
            
        Returns:
            Number of records collected
        """
        try:
            # Apply rate limiting
            await self.rate_limiter.acquire(source_name)
            
            # Collect data for symbol
            data = await collector.collect_for_symbol(symbol)
            
            if not data:
                return 0
            
            count = len(data) if isinstance(data, list) else 1
            self.logger.debug(f"Collected {count} records for {symbol} from {source_name}")
            return count
            
        except Exception as e:
            self.logger.error(f"Error collecting {symbol} from {source_name}: {e}")
            return 0
    
    def get_collection_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status information for all collectors.
        
        Returns:
            Dictionary with status information per source
        """
        status = {}
        
        for source_name, collector in self.collectors.items():
            status[source_name] = {
                'initialized': hasattr(collector, '_initialized') and collector._initialized,
                'last_collection': getattr(collector, 'last_collection_time', None),
                'total_collected': getattr(collector, 'total_collected', 0),
                'error_count': getattr(collector, 'error_count', 0),
                'rate_limit_remaining': self.rate_limiter.get_remaining_requests(source_name)
            }
        
        return status
