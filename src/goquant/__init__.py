"""
GoQuant Sentiment Trader

A comprehensive sentiment analysis and trade signal generation system that processes 
real-time data from social media, news feeds, and financial markets.
"""

__version__ = "0.1.0"
__author__ = "GoQuant Team"
__email__ = "team@goquant.com"

from .main import SentimentTrader
from .config import Config

__all__ = ["SentimentTrader", "Config"]
