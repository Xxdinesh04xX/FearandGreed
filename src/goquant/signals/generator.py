"""
Trading signal generator based on sentiment analysis and market indicators.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

from ..config import Config
from ..utils.logger import get_logger
from ..database.manager import DatabaseManager
from .models import TradingSignalData, SignalType, SignalStrength, FearGreedData


class SignalGenerator:
    """
    Generates trading signals based on sentiment analysis and market indicators.
    
    Combines sentiment scores, fear/greed indicators, momentum analysis,
    and market data to produce actionable trading signals.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the signal generator.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Database manager will be injected
        self.db_manager: Optional[DatabaseManager] = None
        
        # Signal generation parameters
        self.confidence_threshold = config.signal_confidence_threshold
        self.fear_greed_window_hours = config.fear_greed_window_hours
        self.momentum_periods = config.sentiment_momentum_periods
        
        # Tracking
        self.signals_generated = 0
        self.last_generation_time = None
    
    def set_database_manager(self, db_manager: DatabaseManager) -> None:
        """Set the database manager for data operations."""
        self.db_manager = db_manager
    
    async def generate_signals(self) -> List[TradingSignalData]:
        """
        Generate trading signals for all tracked symbols.
        
        Returns:
            List of generated trading signals
        """
        if not self.db_manager:
            self.logger.error("Database manager not set")
            return []
        
        try:
            signals = []
            
            # Generate signals for each tracked symbol
            for symbol in self.config.default_assets:
                try:
                    signal = await self._generate_signal_for_symbol(symbol)
                    if signal:
                        signals.append(signal)
                        
                        # Store signal in database
                        await self.db_manager.insert_trading_signal(signal.to_dict())
                        
                except Exception as e:
                    self.logger.error(f"Error generating signal for {symbol}: {e}")
            
            self.signals_generated += len(signals)
            self.last_generation_time = datetime.utcnow()
            
            if signals:
                self.logger.info(f"Generated {len(signals)} trading signals")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return []
    
    async def _generate_signal_for_symbol(self, symbol: str) -> Optional[TradingSignalData]:
        """
        Generate a trading signal for a specific symbol.
        
        Args:
            symbol: Financial symbol
            
        Returns:
            TradingSignalData or None if no signal should be generated
        """
        try:
            # Get recent sentiment data
            sentiment_records = await self.db_manager.get_recent_sentiment(
                symbol=symbol,
                hours=self.fear_greed_window_hours,
                limit=500
            )
            
            if not sentiment_records or len(sentiment_records) < 5:
                self.logger.debug(f"Insufficient sentiment data for {symbol}")
                return None
            
            # Calculate sentiment metrics
            sentiment_metrics = self._calculate_sentiment_metrics(sentiment_records)
            
            # Calculate fear/greed index
            fear_greed = self._calculate_fear_greed_index(sentiment_records, symbol)
            
            # Generate signal based on metrics
            signal = self._generate_signal_from_metrics(
                symbol, sentiment_metrics, fear_greed
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def _calculate_sentiment_metrics(self, sentiment_records) -> Dict[str, float]:
        """
        Calculate sentiment metrics from historical data.
        
        Args:
            sentiment_records: List of sentiment data records
            
        Returns:
            Dictionary of sentiment metrics
        """
        scores = [record.sentiment_score for record in sentiment_records]
        confidences = [record.confidence for record in sentiment_records]
        
        # Weighted average sentiment
        total_weight = sum(confidences)
        if total_weight > 0:
            weighted_sentiment = sum(s * c for s, c in zip(scores, confidences)) / total_weight
        else:
            weighted_sentiment = sum(scores) / len(scores)
        
        # Calculate momentum (trend over time periods)
        momentum = self._calculate_momentum(sentiment_records)
        
        # Calculate volatility
        volatility = np.std(scores) if len(scores) > 1 else 0.0
        
        # Calculate sentiment strength
        strength = abs(weighted_sentiment)
        
        # Average confidence
        avg_confidence = sum(confidences) / len(confidences)
        
        return {
            'weighted_sentiment': weighted_sentiment,
            'momentum': momentum,
            'volatility': volatility,
            'strength': strength,
            'confidence': avg_confidence,
            'data_points': len(sentiment_records)
        }
    
    def _calculate_momentum(self, sentiment_records) -> float:
        """
        Calculate sentiment momentum using multiple time periods.
        
        Args:
            sentiment_records: List of sentiment records (sorted by time)
            
        Returns:
            Momentum score (-1 to 1)
        """
        if len(sentiment_records) < self.momentum_periods:
            return 0.0
        
        # Sort by time
        sorted_records = sorted(sentiment_records, key=lambda x: x.processed_at)
        
        # Calculate momentum using exponential moving average
        scores = [record.sentiment_score for record in sorted_records]
        
        # Simple momentum: compare recent average to older average
        recent_period = len(scores) // 3  # Last third
        older_period = len(scores) // 3   # First third
        
        if recent_period < 2 or older_period < 2:
            return 0.0
        
        recent_avg = sum(scores[-recent_period:]) / recent_period
        older_avg = sum(scores[:older_period]) / older_period
        
        momentum = recent_avg - older_avg
        
        # Normalize to [-1, 1]
        return max(-1.0, min(1.0, momentum))
    
    def _calculate_fear_greed_index(self, sentiment_records, symbol: str) -> FearGreedData:
        """
        Calculate Fear and Greed index for a symbol.
        
        Args:
            sentiment_records: List of sentiment records
            symbol: Financial symbol
            
        Returns:
            FearGreedData object
        """
        if not sentiment_records:
            return FearGreedData(
                symbol=symbol,
                score=50.0,  # Neutral
                sentiment_component=50.0,
                data_points_count=0,
                confidence=0.0
            )
        
        # Calculate sentiment component (main component)
        scores = [record.sentiment_score for record in sentiment_records]
        confidences = [record.confidence for record in sentiment_records]
        
        # Weighted average sentiment
        total_weight = sum(confidences)
        if total_weight > 0:
            avg_sentiment = sum(s * c for s, c in zip(scores, confidences)) / total_weight
        else:
            avg_sentiment = sum(scores) / len(scores)
        
        # Convert sentiment (-1 to 1) to fear/greed scale (0 to 100)
        # -1 (extreme fear) -> 0, 0 (neutral) -> 50, 1 (extreme greed) -> 100
        sentiment_component = (avg_sentiment + 1) * 50
        
        # Calculate volatility component (high volatility = more fear)
        volatility = np.std(scores) if len(scores) > 1 else 0.0
        volatility_component = max(0, 50 - (volatility * 100))  # High volatility reduces score
        
        # Calculate momentum component
        momentum = self._calculate_momentum(sentiment_records)
        momentum_component = (momentum + 1) * 50  # Convert to 0-100 scale
        
        # Combine components (weighted average)
        fear_greed_score = (
            sentiment_component * 0.6 +
            volatility_component * 0.2 +
            momentum_component * 0.2
        )
        
        # Ensure score is in [0, 100] range
        fear_greed_score = max(0.0, min(100.0, fear_greed_score))
        
        # Calculate overall confidence
        confidence = sum(confidences) / len(confidences)
        
        return FearGreedData(
            symbol=symbol,
            score=fear_greed_score,
            sentiment_component=sentiment_component,
            volatility_component=volatility_component,
            momentum_component=momentum_component,
            data_points_count=len(sentiment_records),
            confidence=confidence,
            timestamp=datetime.utcnow()
        )
    
    def _generate_signal_from_metrics(
        self, 
        symbol: str, 
        metrics: Dict[str, float], 
        fear_greed: FearGreedData
    ) -> Optional[TradingSignalData]:
        """
        Generate trading signal from calculated metrics.
        
        Args:
            symbol: Financial symbol
            metrics: Sentiment metrics
            fear_greed: Fear/Greed index data
            
        Returns:
            TradingSignalData or None
        """
        # Check if we have enough confidence to generate a signal
        if metrics['confidence'] < self.confidence_threshold:
            return None
        
        # Determine signal type based on sentiment and fear/greed
        signal_type = self._determine_signal_type(metrics, fear_greed)
        
        if signal_type == SignalType.HOLD:
            return None  # Don't generate HOLD signals
        
        # Determine signal strength
        strength = self._determine_signal_strength(metrics, fear_greed)
        
        # Calculate position size based on confidence and strength
        position_size = self._calculate_position_size(metrics, strength)
        
        # Calculate stop loss and take profit levels
        stop_loss, take_profit = self._calculate_risk_levels(signal_type, metrics)
        
        # Estimate signal duration
        duration_hours = self._estimate_signal_duration(metrics, strength)
        
        # Create signal metadata
        metadata = {
            'sentiment_metrics': metrics,
            'fear_greed_data': fear_greed.to_dict(),
            'generation_method': 'sentiment_based',
            'model_version': '1.0'
        }
        
        return TradingSignalData(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            confidence=metrics['confidence'],
            sentiment_score=metrics['weighted_sentiment'],
            fear_greed_index=fear_greed.score,
            recommended_position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            expected_duration_hours=duration_hours,
            metadata=metadata,
            generated_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=duration_hours)
        )
    
    def _determine_signal_type(self, metrics: Dict[str, float], fear_greed: FearGreedData) -> SignalType:
        """Determine the type of signal to generate."""
        sentiment = metrics['weighted_sentiment']
        momentum = metrics['momentum']
        fg_score = fear_greed.score
        
        # Strong positive sentiment + positive momentum + greed = BUY
        if sentiment > 0.3 and momentum > 0.2 and fg_score > 60:
            return SignalType.BUY
        
        # Strong negative sentiment + negative momentum + fear = SELL
        if sentiment < -0.3 and momentum < -0.2 and fg_score < 40:
            return SignalType.SELL
        
        # Extreme fear might be a contrarian BUY opportunity
        if fg_score < 20 and sentiment < -0.5:
            return SignalType.BUY
        
        # Extreme greed might be a contrarian SELL opportunity
        if fg_score > 80 and sentiment > 0.5:
            return SignalType.SELL
        
        return SignalType.HOLD
    
    def _determine_signal_strength(self, metrics: Dict[str, float], fear_greed: FearGreedData) -> SignalStrength:
        """Determine the strength of the signal."""
        strength_score = (
            abs(metrics['weighted_sentiment']) * 0.4 +
            abs(metrics['momentum']) * 0.3 +
            metrics['confidence'] * 0.3
        )
        
        if strength_score > 0.7:
            return SignalStrength.STRONG
        elif strength_score > 0.4:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def _calculate_position_size(self, metrics: Dict[str, float], strength: SignalStrength) -> float:
        """Calculate recommended position size as percentage of portfolio."""
        base_size = {
            SignalStrength.WEAK: 0.02,      # 2%
            SignalStrength.MODERATE: 0.05,   # 5%
            SignalStrength.STRONG: 0.10      # 10%
        }[strength]
        
        # Adjust based on confidence
        confidence_multiplier = metrics['confidence']
        
        return base_size * confidence_multiplier
    
    def _calculate_risk_levels(self, signal_type: SignalType, metrics: Dict[str, float]) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels."""
        # These would typically be based on price data and volatility
        # For now, return relative percentages
        
        if signal_type == SignalType.BUY:
            stop_loss = -0.05  # 5% stop loss
            take_profit = 0.10  # 10% take profit
        else:  # SELL
            stop_loss = 0.05   # 5% stop loss (price going up)
            take_profit = -0.10  # 10% take profit (price going down)
        
        # Adjust based on volatility
        volatility_adjustment = min(2.0, 1 + metrics['volatility'])
        stop_loss *= volatility_adjustment
        take_profit *= volatility_adjustment
        
        return stop_loss, take_profit
    
    def _estimate_signal_duration(self, metrics: Dict[str, float], strength: SignalStrength) -> int:
        """Estimate how long the signal should remain active (in hours)."""
        base_duration = {
            SignalStrength.WEAK: 4,      # 4 hours
            SignalStrength.MODERATE: 8,   # 8 hours
            SignalStrength.STRONG: 24     # 24 hours
        }[strength]
        
        # Adjust based on momentum (higher momentum = longer duration)
        momentum_factor = 1 + abs(metrics['momentum']) * 0.5
        
        return int(base_duration * momentum_factor)

    async def calculate_portfolio_signals(self) -> Dict[str, Any]:
        """
        Calculate portfolio-level signals and risk metrics.

        Returns:
            Dictionary with portfolio analysis
        """
        if not self.db_manager:
            return {}

        try:
            portfolio_analysis = {
                'overall_sentiment': 0.0,
                'fear_greed_index': 50.0,
                'risk_level': 'MODERATE',
                'recommended_exposure': 0.5,
                'sector_breakdown': {},
                'signal_distribution': {'BUY': 0, 'SELL': 0, 'HOLD': 0},
                'confidence_score': 0.0
            }

            # Get active signals for all symbols
            active_signals = await self.db_manager.get_active_signals()

            if not active_signals:
                return portfolio_analysis

            # Calculate signal distribution
            for signal in active_signals:
                signal_type = signal.signal_type
                portfolio_analysis['signal_distribution'][signal_type] = \
                    portfolio_analysis['signal_distribution'].get(signal_type, 0) + 1

            # Calculate weighted sentiment
            total_weight = sum(signal.confidence for signal in active_signals)
            if total_weight > 0:
                weighted_sentiment = sum(
                    signal.sentiment_score * signal.confidence
                    for signal in active_signals
                ) / total_weight
                portfolio_analysis['overall_sentiment'] = weighted_sentiment

            # Calculate average fear/greed
            fear_greed_values = [
                signal.fear_greed_index for signal in active_signals
                if signal.fear_greed_index is not None
            ]
            if fear_greed_values:
                portfolio_analysis['fear_greed_index'] = sum(fear_greed_values) / len(fear_greed_values)

            # Calculate confidence score
            portfolio_analysis['confidence_score'] = sum(signal.confidence for signal in active_signals) / len(active_signals)

            # Determine risk level
            portfolio_analysis['risk_level'] = self._calculate_portfolio_risk_level(portfolio_analysis)

            # Calculate recommended exposure
            portfolio_analysis['recommended_exposure'] = self._calculate_recommended_exposure(portfolio_analysis)

            return portfolio_analysis

        except Exception as e:
            self.logger.error(f"Error calculating portfolio signals: {e}")
            return {}

    def _calculate_portfolio_risk_level(self, portfolio_analysis: Dict[str, Any]) -> str:
        """Calculate overall portfolio risk level."""
        fear_greed = portfolio_analysis['fear_greed_index']
        sentiment = abs(portfolio_analysis['overall_sentiment'])
        confidence = portfolio_analysis['confidence_score']

        # High risk conditions
        if fear_greed < 20 or fear_greed > 80:  # Extreme fear or greed
            return 'HIGH'

        if sentiment > 0.7 and confidence > 0.8:  # Very strong sentiment with high confidence
            return 'HIGH'

        # Low risk conditions
        if 40 <= fear_greed <= 60 and sentiment < 0.3:  # Neutral conditions
            return 'LOW'

        return 'MODERATE'

    def _calculate_recommended_exposure(self, portfolio_analysis: Dict[str, Any]) -> float:
        """Calculate recommended portfolio exposure."""
        base_exposure = 0.5  # 50% base exposure

        # Adjust based on fear/greed index
        fear_greed = portfolio_analysis['fear_greed_index']
        if fear_greed < 30:  # Fear - reduce exposure
            fear_greed_adjustment = -0.2
        elif fear_greed > 70:  # Greed - reduce exposure
            fear_greed_adjustment = -0.1
        else:  # Neutral - maintain exposure
            fear_greed_adjustment = 0.0

        # Adjust based on confidence
        confidence = portfolio_analysis['confidence_score']
        confidence_adjustment = (confidence - 0.5) * 0.2  # -0.1 to +0.1

        # Adjust based on signal distribution
        signals = portfolio_analysis['signal_distribution']
        total_signals = sum(signals.values())
        if total_signals > 0:
            buy_ratio = signals.get('BUY', 0) / total_signals
            sell_ratio = signals.get('SELL', 0) / total_signals
            signal_adjustment = (buy_ratio - sell_ratio) * 0.1
        else:
            signal_adjustment = 0.0

        # Calculate final exposure
        recommended_exposure = base_exposure + fear_greed_adjustment + confidence_adjustment + signal_adjustment

        # Clamp to reasonable range
        return max(0.1, min(0.9, recommended_exposure))

    async def get_signal_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for generated signals.

        Returns:
            Dictionary with performance statistics
        """
        # This would query the signal_performance table
        # For now, return mock metrics
        return {
            'total_signals_generated': self.signals_generated,
            'win_rate': 0.65,  # 65% win rate
            'average_return': 0.08,  # 8% average return
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.15,  # 15% max drawdown
            'last_generation_time': self.last_generation_time
        }
