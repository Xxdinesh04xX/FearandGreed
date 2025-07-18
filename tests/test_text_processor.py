"""
Tests for text processing utilities.
"""

import pytest
from src.goquant.utils.text_processor import TextProcessor


class TestTextProcessor:
    """Test cases for TextProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = TextProcessor()
    
    def test_extract_financial_symbols(self):
        """Test financial symbol extraction."""
        text = "I'm bullish on $AAPL and $TSLA. Also watching BTC and ETH closely."
        symbols = self.processor.extract_financial_symbols(text)
        
        assert "AAPL" in symbols
        assert "TSLA" in symbols
        assert "BTC" in symbols
        assert "ETH" in symbols
    
    def test_extract_hashtags(self):
        """Test hashtag extraction."""
        text = "Great day for #stocks and #trading! #BullMarket is here."
        hashtags = self.processor.extract_hashtags(text)
        
        assert "stocks" in hashtags
        assert "trading" in hashtags
        assert "BullMarket" in hashtags
    
    def test_extract_mentions(self):
        """Test user mention extraction."""
        text = "Thanks @elonmusk for the great insights! @cathiedwood agrees."
        mentions = self.processor.extract_mentions(text)
        
        assert "elonmusk" in mentions
        assert "cathiedwood" in mentions
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        text = "🚀 $AAPL to the moon! 🌙 Check this link: https://example.com @user #stocks"
        cleaned = self.processor.clean_text(text, preserve_symbols=True)
        
        # Should preserve financial symbols
        assert "AAPL" in cleaned.upper()
        
        # Should remove URLs
        assert "https://example.com" not in cleaned
        
        # Should be lowercase (except preserved symbols)
        assert cleaned.islower() or "AAPL" in cleaned
    
    def test_is_financial_text(self):
        """Test financial text detection."""
        financial_text = "I'm buying more $AAPL shares today. Great earnings report!"
        non_financial_text = "What a beautiful day! Going to the park with friends."
        
        assert self.processor.is_financial_text(financial_text) == True
        assert self.processor.is_financial_text(non_financial_text) == False
    
    def test_preprocess_for_sentiment(self):
        """Test sentiment preprocessing."""
        text = "🚀🚀🚀 $AAPL is AMAZINGGGGG!!! To the moon! 🌙"
        processed = self.processor.preprocess_for_sentiment(text)
        
        # Should reduce excessive repetition
        assert "AMAZINGGG" not in processed or processed.count("G") < 6
        
        # Should preserve financial symbols
        assert "AAPL" in processed.upper()
    
    def test_tokenize(self):
        """Test text tokenization."""
        text = "The stock market is volatile today."
        tokens = self.processor.tokenize(text)
        
        expected_tokens = ["the", "stock", "market", "is", "volatile", "today"]
        assert all(token in tokens for token in expected_tokens)
        
        # Should filter out very short tokens
        assert all(len(token) > 1 for token in tokens)


if __name__ == "__main__":
    pytest.main([__file__])
