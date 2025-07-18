"""
Trading signal generation module for GoQuant Sentiment Trader.
"""

from .generator import SignalGenerator
from .models import TradingSignalData, SignalStrength, SignalType

__all__ = ["SignalGenerator", "TradingSignalData", "SignalStrength", "SignalType"]
