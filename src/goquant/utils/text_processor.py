"""
Text processing utilities for sentiment analysis.
"""

import re
import string
from typing import List, Set, Optional
from urllib.parse import urlparse


class TextProcessor:
    """
    Text preprocessing utilities for financial sentiment analysis.
    
    Handles cleaning, normalization, and extraction of relevant information
    from social media posts, news articles, and other text sources.
    """
    
    def __init__(self):
        """Initialize the text processor."""
        # Common financial symbols and patterns
        self.stock_pattern = re.compile(r'\$[A-Z]{1,5}\b')
        self.crypto_pattern = re.compile(r'\b(?:BTC|ETH|ADA|SOL|MATIC|DOGE|SHIB|AVAX|DOT|LINK)\b', re.IGNORECASE)
        self.hashtag_pattern = re.compile(r'#\w+')
        self.mention_pattern = re.compile(r'@\w+')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        
        # Emoji pattern (basic)
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        
        # Financial stopwords (in addition to standard stopwords)
        self.financial_stopwords = {
            'stock', 'stocks', 'share', 'shares', 'market', 'markets',
            'trading', 'trader', 'invest', 'investment', 'portfolio',
            'price', 'prices', 'chart', 'charts', 'analysis', 'technical'
        }
    
    def clean_text(self, text: str, preserve_symbols: bool = True) -> str:
        """
        Clean and normalize text for sentiment analysis.
        
        Args:
            text: Raw text to clean
            preserve_symbols: Whether to preserve financial symbols ($AAPL, BTC, etc.)
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = self.url_pattern.sub(' ', text)
        
        # Handle financial symbols
        if preserve_symbols:
            # Extract and preserve financial symbols
            symbols = self.extract_financial_symbols(text)
            # Replace symbols with placeholders
            for i, symbol in enumerate(symbols):
                text = text.replace(symbol.lower(), f' SYMBOL_{i} ')
        
        # Remove mentions and hashtags (but keep the text)
        text = self.mention_pattern.sub(' ', text)
        text = self.hashtag_pattern.sub(' ', text)
        
        # Remove emojis (optional - they might contain sentiment info)
        text = self.emoji_pattern.sub(' ', text)
        
        # Remove extra whitespace and punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Restore financial symbols
        if preserve_symbols:
            for i, symbol in enumerate(symbols):
                text = text.replace(f'symbol_{i}', symbol.upper())
        
        return text.strip()
    
    def extract_financial_symbols(self, text: str) -> List[str]:
        """
        Extract financial symbols from text.
        
        Args:
            text: Text to extract symbols from
            
        Returns:
            List of financial symbols found
        """
        symbols = []
        
        # Extract stock symbols ($AAPL, $TSLA, etc.)
        stock_symbols = self.stock_pattern.findall(text)
        symbols.extend([s[1:] for s in stock_symbols])  # Remove $ prefix
        
        # Extract crypto symbols
        crypto_symbols = self.crypto_pattern.findall(text)
        symbols.extend(crypto_symbols)
        
        return list(set(symbols))  # Remove duplicates
    
    def extract_hashtags(self, text: str) -> List[str]:
        """
        Extract hashtags from text.
        
        Args:
            text: Text to extract hashtags from
            
        Returns:
            List of hashtags found
        """
        hashtags = self.hashtag_pattern.findall(text)
        return [tag[1:] for tag in hashtags]  # Remove # prefix
    
    def extract_mentions(self, text: str) -> List[str]:
        """
        Extract user mentions from text.
        
        Args:
            text: Text to extract mentions from
            
        Returns:
            List of mentions found
        """
        mentions = self.mention_pattern.findall(text)
        return [mention[1:] for mention in mentions]  # Remove @ prefix
    
    def is_financial_text(self, text: str, threshold: float = 0.1) -> bool:
        """
        Determine if text is likely financial/trading related.
        
        Args:
            text: Text to analyze
            threshold: Minimum ratio of financial terms to consider financial
            
        Returns:
            True if text appears to be financial-related
        """
        if not text:
            return False
        
        words = text.lower().split()
        if not words:
            return False
        
        # Check for financial symbols
        if self.extract_financial_symbols(text):
            return True
        
        # Check for financial keywords
        financial_words = sum(1 for word in words if word in self.financial_stopwords)
        ratio = financial_words / len(words)
        
        return ratio >= threshold
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Clean text first
        cleaned = self.clean_text(text)
        
        # Split into tokens
        tokens = cleaned.split()
        
        # Filter out very short tokens
        tokens = [token for token in tokens if len(token) > 1]
        
        return tokens
    
    def preprocess_for_sentiment(self, text: str) -> str:
        """
        Preprocess text specifically for sentiment analysis.
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text ready for sentiment analysis
        """
        # Clean text while preserving financial symbols
        cleaned = self.clean_text(text, preserve_symbols=True)
        
        # Additional preprocessing for sentiment models
        # Remove excessive repetition
        cleaned = re.sub(r'(.)\1{2,}', r'\1\1', cleaned)
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip()


# Global text processor instance
text_processor = TextProcessor()


def get_text_processor() -> TextProcessor:
    """Get the global text processor instance."""
    return text_processor
