"""
Tests for trading signal generation functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from src.goquant.signals.generator import SignalGenerator
from src.goquant.signals.models import TradingSignalData, SignalType, SignalStrength, FearGreedData
from src.goquant.config import Config
from src.goquant.database.models import SentimentData


class TestSignalGenerator:
    """Test cases for SignalGenerator class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Mock(spec=Config)
        config.signal_confidence_threshold = 0.7
        config.fear_greed_window_hours = 24
        config.sentiment_momentum_periods = 5
        config.default_assets = ['BTC', 'ETH', 'AAPL']
        return config
    
    @pytest.fixture
    def generator(self, config):
        """Create SignalGenerator instance."""
        return SignalGenerator(config)
    
    @pytest.fixture
    def mock_sentiment_records(self):
        """Create mock sentiment data records."""
        base_time = datetime.utcnow()
        records = []
        
        for i in range(20):
            record = Mock(spec=SentimentData)
            record.sentiment_score = 0.1 + (i % 10) * 0.05  # Varying sentiment
            record.confidence = 0.8
            record.processed_at = base_time - timedelta(hours=i)
            record.emotions = {
                'fear': 0.2,
                'greed': 0.6,
                'neutral': 0.2,
                'uncertainty': 0.3
            }
            records.append(record)
        
        return records
    
    def test_calculate_sentiment_metrics(self, generator, mock_sentiment_records):
        """Test sentiment metrics calculation."""
        metrics = generator._calculate_sentiment_metrics(mock_sentiment_records)
        
        assert 'weighted_sentiment' in metrics
        assert 'momentum' in metrics
        assert 'volatility' in metrics
        assert 'strength' in metrics
        assert 'confidence' in metrics
        assert 'data_points' in metrics
        
        assert metrics['data_points'] == len(mock_sentiment_records)
        assert 0 <= metrics['confidence'] <= 1
        assert -1 <= metrics['momentum'] <= 1
    
    def test_calculate_fear_greed_index(self, generator, mock_sentiment_records):
        """Test fear and greed index calculation."""
        fear_greed = generator._calculate_fear_greed_index(mock_sentiment_records, 'BTC')
        
        assert isinstance(fear_greed, FearGreedData)
        assert fear_greed.symbol == 'BTC'
        assert 0 <= fear_greed.score <= 100
        assert 0 <= fear_greed.sentiment_component <= 100
        assert fear_greed.data_points_count == len(mock_sentiment_records)
        assert 0 <= fear_greed.confidence <= 1
    
    def test_determine_signal_type(self, generator):
        """Test signal type determination."""
        # Test BUY signal conditions
        metrics = {
            'weighted_sentiment': 0.4,
            'momentum': 0.3,
            'volatility': 0.1,
            'strength': 0.4,
            'confidence': 0.8
        }
        fear_greed = FearGreedData(
            symbol='BTC',
            score=70,  # Greed
            sentiment_component=70,
            data_points_count=10,
            confidence=0.8
        )
        
        signal_type = generator._determine_signal_type(metrics, fear_greed)
        assert signal_type == SignalType.BUY
        
        # Test SELL signal conditions
        metrics['weighted_sentiment'] = -0.4
        metrics['momentum'] = -0.3
        fear_greed.score = 30  # Fear
        
        signal_type = generator._determine_signal_type(metrics, fear_greed)
        assert signal_type == SignalType.SELL
        
        # Test HOLD conditions
        metrics['weighted_sentiment'] = 0.1
        metrics['momentum'] = 0.05
        fear_greed.score = 50  # Neutral
        
        signal_type = generator._determine_signal_type(metrics, fear_greed)
        assert signal_type == SignalType.HOLD
    
    def test_determine_signal_strength(self, generator):
        """Test signal strength determination."""
        # Strong signal
        metrics = {
            'weighted_sentiment': 0.8,
            'momentum': 0.6,
            'confidence': 0.9
        }
        fear_greed = Mock()
        
        strength = generator._determine_signal_strength(metrics, fear_greed)
        assert strength == SignalStrength.STRONG
        
        # Moderate signal
        metrics['weighted_sentiment'] = 0.5
        metrics['momentum'] = 0.3
        metrics['confidence'] = 0.6
        
        strength = generator._determine_signal_strength(metrics, fear_greed)
        assert strength == SignalStrength.MODERATE
        
        # Weak signal
        metrics['weighted_sentiment'] = 0.2
        metrics['momentum'] = 0.1
        metrics['confidence'] = 0.4
        
        strength = generator._determine_signal_strength(metrics, fear_greed)
        assert strength == SignalStrength.WEAK
    
    def test_calculate_position_size(self, generator):
        """Test position size calculation."""
        metrics = {'confidence': 0.8}
        
        # Test different strengths
        weak_size = generator._calculate_position_size(metrics, SignalStrength.WEAK)
        moderate_size = generator._calculate_position_size(metrics, SignalStrength.MODERATE)
        strong_size = generator._calculate_position_size(metrics, SignalStrength.STRONG)
        
        assert weak_size < moderate_size < strong_size
        assert all(0 < size < 1 for size in [weak_size, moderate_size, strong_size])
    
    def test_calculate_risk_levels(self, generator):
        """Test risk level calculations."""
        metrics = {'volatility': 0.2}
        
        # Test BUY signal
        stop_loss, take_profit = generator._calculate_risk_levels(SignalType.BUY, metrics)
        assert stop_loss < 0  # Stop loss should be negative for BUY
        assert take_profit > 0  # Take profit should be positive for BUY
        
        # Test SELL signal
        stop_loss, take_profit = generator._calculate_risk_levels(SignalType.SELL, metrics)
        assert stop_loss > 0  # Stop loss should be positive for SELL
        assert take_profit < 0  # Take profit should be negative for SELL
    
    def test_estimate_signal_duration(self, generator):
        """Test signal duration estimation."""
        metrics = {'momentum': 0.5}
        
        weak_duration = generator._estimate_signal_duration(metrics, SignalStrength.WEAK)
        moderate_duration = generator._estimate_signal_duration(metrics, SignalStrength.MODERATE)
        strong_duration = generator._estimate_signal_duration(metrics, SignalStrength.STRONG)
        
        assert weak_duration < moderate_duration < strong_duration
        assert all(duration > 0 for duration in [weak_duration, moderate_duration, strong_duration])
    
    @pytest.mark.asyncio
    async def test_generate_signal_from_metrics(self, generator):
        """Test complete signal generation from metrics."""
        metrics = {
            'weighted_sentiment': 0.4,
            'momentum': 0.3,
            'volatility': 0.1,
            'strength': 0.4,
            'confidence': 0.8,
            'data_points': 20
        }
        
        fear_greed = FearGreedData(
            symbol='BTC',
            score=70,
            sentiment_component=70,
            data_points_count=20,
            confidence=0.8
        )
        
        signal = generator._generate_signal_from_metrics('BTC', metrics, fear_greed)
        
        assert isinstance(signal, TradingSignalData)
        assert signal.symbol == 'BTC'
        assert signal.signal_type in [SignalType.BUY, SignalType.SELL]
        assert signal.strength in [SignalStrength.WEAK, SignalStrength.MODERATE, SignalStrength.STRONG]
        assert 0 <= signal.confidence <= 1
        assert -1 <= signal.sentiment_score <= 1
        assert 0 <= signal.fear_greed_index <= 100
        assert signal.recommended_position_size > 0
        assert signal.expected_duration_hours > 0
    
    def test_calculate_portfolio_risk_level(self, generator):
        """Test portfolio risk level calculation."""
        # High risk - extreme fear
        portfolio_analysis = {
            'fear_greed_index': 15,
            'overall_sentiment': 0.3,
            'confidence_score': 0.8
        }
        risk_level = generator._calculate_portfolio_risk_level(portfolio_analysis)
        assert risk_level == 'HIGH'
        
        # Low risk - neutral conditions
        portfolio_analysis = {
            'fear_greed_index': 50,
            'overall_sentiment': 0.1,
            'confidence_score': 0.6
        }
        risk_level = generator._calculate_portfolio_risk_level(portfolio_analysis)
        assert risk_level == 'LOW'
        
        # Moderate risk
        portfolio_analysis = {
            'fear_greed_index': 65,
            'overall_sentiment': 0.4,
            'confidence_score': 0.7
        }
        risk_level = generator._calculate_portfolio_risk_level(portfolio_analysis)
        assert risk_level == 'MODERATE'
    
    def test_calculate_recommended_exposure(self, generator):
        """Test recommended exposure calculation."""
        portfolio_analysis = {
            'fear_greed_index': 50,
            'confidence_score': 0.7,
            'signal_distribution': {'BUY': 3, 'SELL': 1, 'HOLD': 2}
        }
        
        exposure = generator._calculate_recommended_exposure(portfolio_analysis)
        
        assert 0.1 <= exposure <= 0.9  # Should be within reasonable bounds
        assert isinstance(exposure, float)


if __name__ == "__main__":
    pytest.main([__file__])
