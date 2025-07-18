"""
Database models for GoQuant Sentiment Trader.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Text, Boolean, JSON,
    ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class RawData(Base):
    """Raw data collected from various sources."""
    
    __tablename__ = "raw_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(50), nullable=False)  # twitter, reddit, news, etc.
    source_id = Column(String(255), nullable=True)  # Original ID from source
    content = Column(Text, nullable=False)
    author = Column(String(255), nullable=True)
    url = Column(String(500), nullable=True)
    metadata = Column(JSON, nullable=True)  # Additional source-specific data
    collected_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    processed = Column(Boolean, default=False, nullable=False)
    
    # Relationships
    sentiment_data = relationship("SentimentData", back_populates="raw_data")
    
    # Indexes
    __table_args__ = (
        Index('idx_raw_data_source', 'source'),
        Index('idx_raw_data_collected_at', 'collected_at'),
        Index('idx_raw_data_processed', 'processed'),
        UniqueConstraint('source', 'source_id', name='uq_source_source_id'),
    )


class SentimentData(Base):
    """Processed sentiment analysis results."""
    
    __tablename__ = "sentiment_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    raw_data_id = Column(Integer, ForeignKey('raw_data.id'), nullable=False)
    symbol = Column(String(20), nullable=True)  # Financial symbol if detected
    sentiment_score = Column(Float, nullable=False)  # -1 to 1
    confidence = Column(Float, nullable=False)  # 0 to 1
    emotions = Column(JSON, nullable=True)  # Fear, greed, neutral, etc.
    entities = Column(JSON, nullable=True)  # Extracted financial entities
    processed_text = Column(Text, nullable=True)  # Cleaned text used for analysis
    model_version = Column(String(50), nullable=False)
    processed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    raw_data = relationship("RawData", back_populates="sentiment_data")
    
    # Indexes
    __table_args__ = (
        Index('idx_sentiment_symbol', 'symbol'),
        Index('idx_sentiment_processed_at', 'processed_at'),
        Index('idx_sentiment_score', 'sentiment_score'),
    )


class AssetPrice(Base):
    """Asset price data for correlation analysis."""
    
    __tablename__ = "asset_prices"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=True)
    source = Column(String(50), nullable=False)  # yfinance, alpha_vantage, etc.
    
    # Indexes
    __table_args__ = (
        Index('idx_asset_symbol_timestamp', 'symbol', 'timestamp'),
        UniqueConstraint('symbol', 'timestamp', 'source', name='uq_symbol_timestamp_source'),
    )


class TradingSignal(Base):
    """Generated trading signals."""
    
    __tablename__ = "trading_signals"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    signal_type = Column(String(10), nullable=False)  # BUY, SELL, HOLD
    strength = Column(String(10), nullable=False)  # WEAK, MODERATE, STRONG
    confidence = Column(Float, nullable=False)  # 0 to 1
    sentiment_score = Column(Float, nullable=False)  # Current sentiment
    fear_greed_index = Column(Float, nullable=True)  # 0 to 100
    price_at_signal = Column(Float, nullable=True)
    recommended_position_size = Column(Float, nullable=True)  # As percentage
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    expected_duration_hours = Column(Integer, nullable=True)
    metadata = Column(JSON, nullable=True)  # Additional signal data
    generated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_signal_symbol', 'symbol'),
        Index('idx_signal_generated_at', 'generated_at'),
        Index('idx_signal_active', 'is_active'),
        Index('idx_signal_type', 'signal_type'),
    )


class SignalPerformance(Base):
    """Track performance of trading signals."""
    
    __tablename__ = "signal_performance"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(Integer, ForeignKey('trading_signals.id'), nullable=False)
    entry_price = Column(Float, nullable=True)
    exit_price = Column(Float, nullable=True)
    entry_time = Column(DateTime, nullable=True)
    exit_time = Column(DateTime, nullable=True)
    return_percentage = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    hit_stop_loss = Column(Boolean, default=False)
    hit_take_profit = Column(Boolean, default=False)
    status = Column(String(20), default='OPEN')  # OPEN, CLOSED, EXPIRED
    
    # Relationships
    signal = relationship("TradingSignal")
    
    # Indexes
    __table_args__ = (
        Index('idx_performance_signal_id', 'signal_id'),
        Index('idx_performance_status', 'status'),
        Index('idx_performance_return', 'return_percentage'),
    )


class FearGreedIndex(Base):
    """Fear and Greed index calculations."""
    
    __tablename__ = "fear_greed_index"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=True)  # Null for overall market
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    fear_greed_score = Column(Float, nullable=False)  # 0 to 100
    sentiment_component = Column(Float, nullable=False)
    volatility_component = Column(Float, nullable=True)
    momentum_component = Column(Float, nullable=True)
    volume_component = Column(Float, nullable=True)
    data_points_count = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_fear_greed_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_fear_greed_timestamp', 'timestamp'),
    )


class DataQuality(Base):
    """Track data quality metrics."""
    
    __tablename__ = "data_quality"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(50), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    records_collected = Column(Integer, nullable=False)
    records_processed = Column(Integer, nullable=False)
    error_count = Column(Integer, default=0)
    processing_time_seconds = Column(Float, nullable=False)
    success_rate = Column(Float, nullable=False)  # 0 to 1
    
    # Indexes
    __table_args__ = (
        Index('idx_data_quality_source_timestamp', 'source', 'timestamp'),
    )
