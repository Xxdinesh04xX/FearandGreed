"""
Integration tests for GoQuant Sentiment Trader.
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch

from src.goquant.main import SentimentTrader
from src.goquant.config import Config
from src.goquant.database.manager import DatabaseManager


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.fixture
    async def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        db_url = f"sqlite:///{db_path}"
        db_manager = DatabaseManager(db_url)
        await db_manager.initialize()
        
        yield db_manager
        
        await db_manager.close()
        os.unlink(db_path)
    
    @pytest.fixture
    def test_config(self, temp_db):
        """Create test configuration."""
        config = Mock(spec=Config)
        config.database_url = temp_db.database_url
        config.sentiment_model = "distilbert-base-uncased-finetuned-sst-2-english"
        config.device = "cpu"
        config.log_level = "INFO"
        config.data_collection_interval = 60
        config.sentiment_analysis_batch_size = 10
        config.signal_confidence_threshold = 0.7
        config.fear_greed_window_hours = 24
        config.sentiment_momentum_periods = 5
        config.default_assets = ['BTC', 'ETH']
        
        # Mock API keys (not real)
        config.twitter_bearer_token = None
        config.reddit_client_id = None
        config.news_api_key = None
        config.alpha_vantage_api_key = None
        
        return config
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_sentiment_trader_initialization(self, test_config):
        """Test SentimentTrader initialization."""
        with patch('src.goquant.main.get_config', return_value=test_config):
            trader = SentimentTrader()
            
            # Mock the sentiment model initialization to avoid downloading models
            with patch.object(trader.sentiment_analyzer.sentiment_model, 'initialize'):
                await trader.initialize()
                
                assert trader.db_manager is not None
                assert trader.data_collector is not None
                assert trader.sentiment_analyzer is not None
                assert trader.signal_generator is not None
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_data_flow_pipeline(self, temp_db):
        """Test the complete data flow pipeline."""
        # Insert mock raw data
        raw_data = [
            {
                'source': 'test',
                'source_id': 'test_1',
                'content': 'Bitcoin is performing very well today! $BTC to the moon!',
                'author': 'test_user',
                'url': 'https://test.com/1',
                'metadata': {'test': True}
            },
            {
                'source': 'test',
                'source_id': 'test_2',
                'content': 'Market crash incoming! Sell everything now! $BTC $ETH',
                'author': 'test_user_2',
                'url': 'https://test.com/2',
                'metadata': {'test': True}
            }
        ]
        
        # Insert raw data
        inserted_ids = await temp_db.insert_raw_data(raw_data)
        assert len(inserted_ids) == 2
        
        # Get unprocessed data
        unprocessed = await temp_db.get_unprocessed_raw_data(limit=10)
        assert len(unprocessed) == 2
        
        # Mock sentiment analysis results
        sentiment_data = [
            {
                'raw_data_id': inserted_ids[0],
                'symbol': 'BTC',
                'sentiment_score': 0.8,
                'confidence': 0.9,
                'emotions': {'fear': 0.1, 'greed': 0.8, 'neutral': 0.1},
                'entities': {'symbols': ['BTC']},
                'processed_text': 'bitcoin performing well today btc moon',
                'model_version': 'test_model'
            },
            {
                'raw_data_id': inserted_ids[1],
                'symbol': 'BTC',
                'sentiment_score': -0.7,
                'confidence': 0.85,
                'emotions': {'fear': 0.9, 'greed': 0.05, 'neutral': 0.05},
                'entities': {'symbols': ['BTC', 'ETH']},
                'processed_text': 'market crash incoming sell everything btc eth',
                'model_version': 'test_model'
            }
        ]
        
        # Insert sentiment data
        sentiment_ids = await temp_db.insert_sentiment_data(sentiment_data)
        assert len(sentiment_ids) == 2
        
        # Mark raw data as processed
        await temp_db.mark_raw_data_processed(inserted_ids)
        
        # Verify no unprocessed data remains
        unprocessed_after = await temp_db.get_unprocessed_raw_data(limit=10)
        assert len(unprocessed_after) == 0
        
        # Get recent sentiment data
        recent_sentiment = await temp_db.get_recent_sentiment(symbol='BTC', hours=24)
        assert len(recent_sentiment) == 2
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_signal_generation_pipeline(self, temp_db):
        """Test signal generation from sentiment data."""
        from datetime import datetime, timedelta
        
        # Create sentiment data for signal generation
        base_time = datetime.utcnow()
        sentiment_data_list = []
        
        # Create a trend of increasing positive sentiment
        for i in range(10):
            sentiment_data = {
                'raw_data_id': i + 1,
                'symbol': 'BTC',
                'sentiment_score': 0.1 + i * 0.05,  # Increasing sentiment
                'confidence': 0.8,
                'emotions': {'fear': 0.2, 'greed': 0.6, 'neutral': 0.2},
                'entities': {'symbols': ['BTC']},
                'processed_text': f'test sentiment {i}',
                'model_version': 'test_model',
                'processed_at': base_time - timedelta(hours=i)
            }
            sentiment_data_list.append(sentiment_data)
        
        # Insert sentiment data
        await temp_db.insert_sentiment_data(sentiment_data_list)
        
        # Test signal generation
        from src.goquant.signals.generator import SignalGenerator
        from src.goquant.config import Config
        
        config = Mock(spec=Config)
        config.signal_confidence_threshold = 0.7
        config.fear_greed_window_hours = 24
        config.sentiment_momentum_periods = 5
        
        generator = SignalGenerator(config)
        generator.set_database_manager(temp_db)
        
        # Generate signal for BTC
        signal = await generator._generate_signal_for_symbol('BTC')
        
        if signal:  # Signal might not be generated if confidence is too low
            assert signal.symbol == 'BTC'
            assert signal.signal_type in ['BUY', 'SELL', 'HOLD']
            assert 0 <= signal.confidence <= 1
            assert -1 <= signal.sentiment_score <= 1
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_database_operations(self, temp_db):
        """Test various database operations."""
        # Test raw data operations
        raw_data = [{
            'source': 'test',
            'source_id': 'db_test_1',
            'content': 'Test content for database operations',
            'author': 'test_author',
            'metadata': {'test_key': 'test_value'}
        }]
        
        inserted_ids = await temp_db.insert_raw_data(raw_data)
        assert len(inserted_ids) == 1
        
        # Test sentiment data operations
        sentiment_data = [{
            'raw_data_id': inserted_ids[0],
            'symbol': 'TEST',
            'sentiment_score': 0.5,
            'confidence': 0.8,
            'emotions': {'test_emotion': 0.5},
            'entities': {'test_entity': ['TEST']},
            'processed_text': 'processed test content',
            'model_version': 'test_model'
        }]
        
        sentiment_ids = await temp_db.insert_sentiment_data(sentiment_data)
        assert len(sentiment_ids) == 1
        
        # Test signal operations
        signal_data = {
            'symbol': 'TEST',
            'signal_type': 'BUY',
            'strength': 'MODERATE',
            'confidence': 0.8,
            'sentiment_score': 0.5,
            'fear_greed_index': 65.0,
            'recommended_position_size': 0.05,
            'metadata': {'test': True}
        }
        
        signal_id = await temp_db.insert_trading_signal(signal_data)
        assert signal_id is not None
        
        # Test getting active signals
        active_signals = await temp_db.get_active_signals()
        assert len(active_signals) >= 1
        
        # Test getting recent sentiment
        recent_sentiment = await temp_db.get_recent_sentiment(symbol='TEST', hours=1)
        assert len(recent_sentiment) >= 1
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_error_handling(self, temp_db):
        """Test error handling in various scenarios."""
        # Test invalid raw data
        invalid_data = [{'invalid': 'data'}]
        
        with pytest.raises(Exception):
            await temp_db.insert_raw_data(invalid_data)
        
        # Test getting data for non-existent symbol
        no_data = await temp_db.get_recent_sentiment(symbol='NONEXISTENT', hours=24)
        assert len(no_data) == 0
        
        # Test empty data operations
        empty_result = await temp_db.get_unprocessed_raw_data(limit=0)
        assert len(empty_result) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
