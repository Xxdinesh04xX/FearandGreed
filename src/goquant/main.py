"""
Main application entry point for GoQuant Sentiment Trader.
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime

from .config import get_config
from .utils.logger import setup_logger
from .database.manager import DatabaseManager
from .data.collector import DataCollector
from .sentiment.analyzer import SentimentAnalyzer
from .signals.generator import SignalGenerator
from .dashboard.app import create_dashboard_app


class SentimentTrader:
    """Main application class for the sentiment trading system."""
    
    def __init__(self):
        """Initialize the sentiment trader."""
        self.config = get_config()
        self.logger = setup_logger(__name__, self.config.log_level)
        
        # Initialize components
        self.db_manager = DatabaseManager(self.config.database_url)
        self.data_collector = DataCollector(self.config)
        self.sentiment_analyzer = SentimentAnalyzer(self.config)
        self.signal_generator = SignalGenerator(self.config)
        
        # Runtime state
        self.is_running = False
        self.tasks: List[asyncio.Task] = []
        
        self.logger.info("SentimentTrader initialized")
    
    async def initialize(self) -> None:
        """Initialize all components and database."""
        try:
            self.logger.info("Initializing SentimentTrader components...")

            # Initialize database
            await self.db_manager.initialize()

            # Initialize sentiment analyzer (load models)
            await self.sentiment_analyzer.initialize()
            self.sentiment_analyzer.set_database_manager(self.db_manager)

            # Initialize signal generator
            self.signal_generator.set_database_manager(self.db_manager)

            # Initialize data collector
            await self.data_collector.initialize()
            self.data_collector.set_database_manager(self.db_manager)

            self.logger.info("All components initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def start(self) -> None:
        """Start the sentiment trading system."""
        if self.is_running:
            self.logger.warning("System is already running")
            return
        
        try:
            await self.initialize()
            
            self.logger.info("Starting SentimentTrader...")
            self.is_running = True
            
            # Start data collection tasks
            data_task = asyncio.create_task(self._run_data_collection())
            self.tasks.append(data_task)
            
            # Start sentiment analysis task
            sentiment_task = asyncio.create_task(self._run_sentiment_analysis())
            self.tasks.append(sentiment_task)
            
            # Start signal generation task
            signal_task = asyncio.create_task(self._run_signal_generation())
            self.tasks.append(signal_task)
            
            self.logger.info("SentimentTrader started successfully")
            
            # Wait for all tasks to complete
            await asyncio.gather(*self.tasks)
            
        except Exception as e:
            self.logger.error(f"Error starting SentimentTrader: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the sentiment trading system."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping SentimentTrader...")
        self.is_running = False
        
        # Cancel all running tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete cancellation
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Close database connections
        await self.db_manager.close()
        
        self.logger.info("SentimentTrader stopped")
    
    async def _run_data_collection(self) -> None:
        """Run the data collection loop."""
        self.logger.info("Starting data collection loop")
        
        while self.is_running:
            try:
                # Collect data from all sources
                await self.data_collector.collect_all()
                
                # Wait for next collection interval
                await asyncio.sleep(self.config.data_collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in data collection: {e}")
                await asyncio.sleep(10)  # Wait before retrying
    
    async def _run_sentiment_analysis(self) -> None:
        """Run the sentiment analysis loop."""
        self.logger.info("Starting sentiment analysis loop")
        
        while self.is_running:
            try:
                # Process unanalyzed data
                await self.sentiment_analyzer.process_pending_data()
                
                # Wait before next analysis cycle
                await asyncio.sleep(30)  # Analyze every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in sentiment analysis: {e}")
                await asyncio.sleep(10)
    
    async def _run_signal_generation(self) -> None:
        """Run the signal generation loop."""
        self.logger.info("Starting signal generation loop")
        
        while self.is_running:
            try:
                # Generate trading signals
                await self.signal_generator.generate_signals()
                
                # Wait before next signal generation
                await asyncio.sleep(60)  # Generate signals every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in signal generation: {e}")
                await asyncio.sleep(10)
    
    def get_sentiment(self, symbol: str) -> Optional[Dict]:
        """Get current sentiment for a symbol."""
        # This will be implemented to query the database
        # for the latest sentiment data
        pass
    
    def get_signals(self) -> List[Dict]:
        """Get current trading signals."""
        # This will be implemented to query the database
        # for the latest signals
        pass
    
    def start_dashboard(self) -> None:
        """Start the web dashboard."""
        app = create_dashboard_app(self)
        app.run_server(
            host=self.config.dashboard_host,
            port=self.config.dashboard_port,
            debug=self.config.dashboard_debug
        )


async def main():
    """Main entry point for the application."""
    trader = SentimentTrader()
    
    try:
        await trader.start()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Application error: {e}")
    finally:
        await trader.stop()


if __name__ == "__main__":
    asyncio.run(main())
