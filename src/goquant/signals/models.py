"""
Trading signal models and data structures.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime


class SignalType(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalStrength(Enum):
    """Signal strength levels."""
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"


@dataclass
class TradingSignalData:
    """
    Data structure for trading signals.
    """
    symbol: str
    signal_type: SignalType
    strength: SignalStrength
    confidence: float  # 0 to 1
    sentiment_score: float  # -1 to 1
    fear_greed_index: Optional[float] = None  # 0 to 100
    price_at_signal: Optional[float] = None
    recommended_position_size: Optional[float] = None  # As percentage
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    expected_duration_hours: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    generated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'strength': self.strength.value,
            'confidence': self.confidence,
            'sentiment_score': self.sentiment_score,
            'fear_greed_index': self.fear_greed_index,
            'price_at_signal': self.price_at_signal,
            'recommended_position_size': self.recommended_position_size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'expected_duration_hours': self.expected_duration_hours,
            'metadata': self.metadata,
            'generated_at': self.generated_at or datetime.utcnow(),
            'expires_at': self.expires_at,
            'is_active': True
        }


@dataclass
class FearGreedData:
    """
    Fear and Greed index data.
    """
    symbol: Optional[str]  # None for overall market
    score: float  # 0 to 100
    sentiment_component: float
    volatility_component: Optional[float] = None
    momentum_component: Optional[float] = None
    volume_component: Optional[float] = None
    data_points_count: int = 0
    confidence: float = 0.0
    timestamp: Optional[datetime] = None
    
    def get_fear_greed_label(self) -> str:
        """Get human-readable fear/greed label."""
        if self.score <= 20:
            return "Extreme Fear"
        elif self.score <= 40:
            return "Fear"
        elif self.score <= 60:
            return "Neutral"
        elif self.score <= 80:
            return "Greed"
        else:
            return "Extreme Greed"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp or datetime.utcnow(),
            'fear_greed_score': self.score,
            'sentiment_component': self.sentiment_component,
            'volatility_component': self.volatility_component,
            'momentum_component': self.momentum_component,
            'volume_component': self.volume_component,
            'data_points_count': self.data_points_count,
            'confidence': self.confidence
        }
