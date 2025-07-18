"""
Data collection and ingestion for GoQuant Sentiment Trader.
"""

from .collector import DataCollector
from .sources import TwitterCollector, RedditCollector, NewsCollector, FinancialDataCollector

__all__ = [
    "DataCollector", 
    "TwitterCollector", 
    "RedditCollector", 
    "NewsCollector", 
    "FinancialDataCollector"
]
