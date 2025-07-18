"""
Database initialization script for GoQuant Sentiment Trader.
"""

import asyncio
import sys
from pathlib import Path

from ..config import get_config
from ..utils.logger import setup_logger
from .manager import DatabaseManager
from .models import Base


async def initialize_database():
    """Initialize the database with all tables and indexes."""
    config = get_config()
    logger = setup_logger(__name__, config.log_level)
    
    logger.info("Initializing GoQuant database...")
    
    try:
        # Create database manager
        db_manager = DatabaseManager(config.database_url)
        
        # Initialize database schema
        await db_manager.initialize()
        
        logger.info("Database initialized successfully!")
        logger.info(f"Database URL: {config.database_url}")
        
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Create data directory if it doesn't exist
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        logger.info("Created necessary directories")
        
        # Close database connections
        await db_manager.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False


def main():
    """Main entry point for database initialization."""
    success = asyncio.run(initialize_database())
    
    if success:
        print("✓ Database initialization completed successfully!")
        sys.exit(0)
    else:
        print("✗ Database initialization failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
