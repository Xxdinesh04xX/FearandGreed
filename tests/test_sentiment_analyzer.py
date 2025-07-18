"""
Tests for sentiment analysis functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from src.goquant.sentiment.analyzer import SentimentAnalyzer
from src.goquant.sentiment.models import SentimentResult
from src.goquant.config import Config
from src.goquant.database.models import RawData


class TestSentimentAnalyzer:
    """Test cases for SentimentAnalyzer class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Mock(spec=Config)
        config.sentiment_model = "distilbert-base-uncased-finetuned-sst-2-english"
        config.device = "cpu"
        config.sentiment_analysis_batch_size = 10
        return config
    
    @pytest.fixture
    def analyzer(self, config):
        """Create SentimentAnalyzer instance."""
        return SentimentAnalyzer(config)
    
    @pytest.mark.asyncio
    async def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        # Mock the sentiment model initialization
        with patch.object(analyzer.sentiment_model, 'initialize', new_callable=AsyncMock):
            await analyzer.initialize()
            assert analyzer._initialized is True
    
    @pytest.mark.asyncio
    async def test_analyze_text(self, analyzer):
        """Test single text analysis."""
        # Mock the sentiment model
        mock_result = SentimentResult(
            score=0.5,
            confidence=0.8,
            label='positive',
            probabilities={'positive': 0.7, 'negative': 0.3}
        )
        
        with patch.object(analyzer.sentiment_model, 'analyze_sentiment', return_value=mock_result):
            analyzer._initialized = True
            result = await analyzer.analyze_text("Great stock performance today!")
            
            assert isinstance(result, SentimentResult)
            assert result.score == 0.5
            assert result.confidence == 0.8
            assert result.label == 'positive'
    
    @pytest.mark.asyncio
    async def test_process_batch(self, analyzer):
        """Test batch processing of raw data."""
        # Create mock raw data
        raw_data = [
            Mock(spec=RawData, id=1, content="Bullish on $AAPL today!"),
            Mock(spec=RawData, id=2, content="Market crash incoming, sell everything!"),
            Mock(spec=RawData, id=3, content="Neutral market conditions.")
        ]
        
        # Mock sentiment results
        mock_results = [
            SentimentResult(score=0.6, confidence=0.8, label='positive', probabilities={}),
            SentimentResult(score=-0.7, confidence=0.9, label='negative', probabilities={}),
            SentimentResult(score=0.0, confidence=0.5, label='neutral', probabilities={})
        ]
        
        with patch.object(analyzer.sentiment_model, 'analyze_batch', return_value=mock_results):
            analyzer._initialized = True
            result = await analyzer._process_batch(raw_data)
            
            assert len(result) == 3
            assert all('raw_data_id' in item for item in result)
            assert all('sentiment_score' in item for item in result)
            assert all('confidence' in item for item in result)
    
    @pytest.mark.asyncio
    async def test_sentiment_momentum_calculation(self, analyzer):
        """Test sentiment momentum calculation."""
        from datetime import datetime, timedelta
        
        # Create mock sentiment records with time progression
        base_time = datetime.utcnow()
        sentiment_records = []
        
        for i in range(10):
            record = Mock()
            record.sentiment_score = 0.1 * i - 0.5  # Increasing sentiment
            record.processed_at = base_time + timedelta(hours=i)
            sentiment_records.append(record)
        
        momentum = analyzer._calculate_sentiment_momentum(sentiment_records)
        
        # Should be positive since sentiment is increasing
        assert momentum > 0
        assert -1 <= momentum <= 1
    
    @pytest.mark.asyncio
    async def test_aggregate_emotions(self, analyzer):
        """Test emotion aggregation."""
        # Create mock sentiment records with emotions
        sentiment_records = []
        
        for i in range(5):
            record = Mock()
            record.emotions = {
                'fear': 0.2 + i * 0.1,
                'greed': 0.8 - i * 0.1,
                'neutral': 0.3,
                'uncertainty': 0.4
            }
            sentiment_records.append(record)
        
        aggregated = analyzer._aggregate_emotions(sentiment_records)
        
        assert 'fear' in aggregated
        assert 'greed' in aggregated
        assert 'neutral' in aggregated
        assert 'uncertainty' in aggregated
        
        # Check that values are averages
        assert 0 <= aggregated['fear'] <= 1
        assert 0 <= aggregated['greed'] <= 1
    
    def test_processing_stats(self, analyzer):
        """Test processing statistics."""
        analyzer._initialized = True
        analyzer.processed_count = 100
        analyzer.error_count = 5
        
        stats = analyzer.get_processing_stats()
        
        assert stats['initialized'] is True
        assert stats['processed_count'] == 100
        assert stats['error_count'] == 5
        assert 'model_name' in stats
        assert 'device' in stats


if __name__ == "__main__":
    pytest.main([__file__])
