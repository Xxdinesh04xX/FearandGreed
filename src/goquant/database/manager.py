"""
Database manager for GoQuant Sentiment Trader.
"""

import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy import create_engine, select, and_, or_, desc
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from .models import Base, RawData, SentimentData, TradingSignal, AssetPrice, FearGreedIndex
from ..utils.logger import get_logger


class DatabaseManager:
    """
    Manages database connections and operations for the sentiment trading system.
    
    Supports both SQLite (for development) and PostgreSQL (for production).
    Provides async and sync interfaces for database operations.
    """
    
    def __init__(self, database_url: str):
        """
        Initialize the database manager.
        
        Args:
            database_url: Database connection URL
        """
        self.database_url = database_url
        self.logger = get_logger(__name__)
        
        # Determine if we're using SQLite or PostgreSQL
        self.is_sqlite = database_url.startswith('sqlite')
        
        if self.is_sqlite:
            # SQLite configuration
            self.engine = create_engine(
                database_url,
                poolclass=StaticPool,
                connect_args={"check_same_thread": False},
                echo=False
            )
            # For async operations with SQLite, we'll use aiosqlite
            async_url = database_url.replace('sqlite:///', 'sqlite+aiosqlite:///')
            self.async_engine = create_async_engine(
                async_url,
                poolclass=StaticPool,
                connect_args={"check_same_thread": False},
                echo=False
            )
        else:
            # PostgreSQL configuration
            self.engine = create_engine(database_url, echo=False)
            # For async operations with PostgreSQL
            async_url = database_url.replace('postgresql://', 'postgresql+asyncpg://')
            self.async_engine = create_async_engine(async_url, echo=False)
        
        # Session factories
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.AsyncSessionLocal = async_sessionmaker(
            bind=self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        self.logger.info(f"Database manager initialized with {database_url}")
    
    async def initialize(self) -> None:
        """Initialize the database schema."""
        try:
            # Create all tables
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            self.logger.info("Database schema initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def close(self) -> None:
        """Close database connections."""
        await self.async_engine.dispose()
        self.engine.dispose()
        self.logger.info("Database connections closed")
    
    # Context managers for sessions
    def get_session(self) -> Session:
        """Get a synchronous database session."""
        return self.SessionLocal()
    
    async def get_async_session(self) -> AsyncSession:
        """Get an asynchronous database session."""
        return self.AsyncSessionLocal()
    
    # Raw Data Operations
    async def insert_raw_data(self, data_list: List[Dict[str, Any]]) -> List[int]:
        """
        Insert raw data records.
        
        Args:
            data_list: List of raw data dictionaries
            
        Returns:
            List of inserted record IDs
        """
        async with self.get_async_session() as session:
            try:
                records = []
                for data in data_list:
                    record = RawData(**data)
                    records.append(record)
                    session.add(record)
                
                await session.commit()
                
                # Get the IDs of inserted records
                ids = [record.id for record in records]
                self.logger.debug(f"Inserted {len(ids)} raw data records")
                return ids
                
            except Exception as e:
                await session.rollback()
                self.logger.error(f"Failed to insert raw data: {e}")
                raise
    
    async def get_unprocessed_raw_data(self, limit: int = 100) -> List[RawData]:
        """
        Get unprocessed raw data records.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of unprocessed RawData records
        """
        async with self.get_async_session() as session:
            try:
                result = await session.execute(
                    select(RawData)
                    .where(RawData.processed == False)
                    .order_by(RawData.collected_at)
                    .limit(limit)
                )
                return result.scalars().all()
                
            except Exception as e:
                self.logger.error(f"Failed to get unprocessed raw data: {e}")
                raise

    async def mark_raw_data_processed(self, record_ids: List[int]) -> None:
        """
        Mark raw data records as processed.

        Args:
            record_ids: List of record IDs to mark as processed
        """
        async with self.get_async_session() as session:
            try:
                await session.execute(
                    RawData.__table__.update()
                    .where(RawData.id.in_(record_ids))
                    .values(processed=True)
                )
                await session.commit()
                self.logger.debug(f"Marked {len(record_ids)} records as processed")

            except Exception as e:
                await session.rollback()
                self.logger.error(f"Failed to mark records as processed: {e}")
                raise

    # Sentiment Data Operations
    async def insert_sentiment_data(self, sentiment_list: List[Dict[str, Any]]) -> List[int]:
        """
        Insert sentiment analysis results.

        Args:
            sentiment_list: List of sentiment data dictionaries

        Returns:
            List of inserted record IDs
        """
        async with self.get_async_session() as session:
            try:
                records = []
                for data in sentiment_list:
                    record = SentimentData(**data)
                    records.append(record)
                    session.add(record)

                await session.commit()

                ids = [record.id for record in records]
                self.logger.debug(f"Inserted {len(ids)} sentiment records")
                return ids

            except Exception as e:
                await session.rollback()
                self.logger.error(f"Failed to insert sentiment data: {e}")
                raise

    async def get_recent_sentiment(
        self,
        symbol: Optional[str] = None,
        hours: int = 24,
        limit: int = 1000
    ) -> List[SentimentData]:
        """
        Get recent sentiment data for analysis.

        Args:
            symbol: Optional symbol filter
            hours: Number of hours to look back
            limit: Maximum number of records

        Returns:
            List of recent sentiment data
        """
        async with self.get_async_session() as session:
            try:
                cutoff_time = datetime.utcnow() - timedelta(hours=hours)

                query = select(SentimentData).where(
                    SentimentData.processed_at >= cutoff_time
                )

                if symbol:
                    query = query.where(SentimentData.symbol == symbol)

                query = query.order_by(desc(SentimentData.processed_at)).limit(limit)

                result = await session.execute(query)
                return result.scalars().all()

            except Exception as e:
                self.logger.error(f"Failed to get recent sentiment: {e}")
                raise

    # Trading Signal Operations
    async def insert_trading_signal(self, signal_data: Dict[str, Any]) -> int:
        """
        Insert a trading signal.

        Args:
            signal_data: Signal data dictionary

        Returns:
            ID of inserted signal
        """
        async with self.get_async_session() as session:
            try:
                signal = TradingSignal(**signal_data)
                session.add(signal)
                await session.commit()

                self.logger.debug(f"Inserted trading signal for {signal.symbol}")
                return signal.id

            except Exception as e:
                await session.rollback()
                self.logger.error(f"Failed to insert trading signal: {e}")
                raise

    async def get_active_signals(self, symbol: Optional[str] = None) -> List[TradingSignal]:
        """
        Get active trading signals.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of active trading signals
        """
        async with self.get_async_session() as session:
            try:
                query = select(TradingSignal).where(TradingSignal.is_active == True)

                if symbol:
                    query = query.where(TradingSignal.symbol == symbol)

                query = query.order_by(desc(TradingSignal.generated_at))

                result = await session.execute(query)
                return result.scalars().all()

            except Exception as e:
                self.logger.error(f"Failed to get active signals: {e}")
                raise
