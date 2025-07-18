"""
Basic test to check if core functionality works.
"""

import os
import asyncio


def test_imports():
    """Test if we can import basic modules."""
    print("🧪 Testing Basic Imports")
    print("=" * 30)
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
    
    try:
        import requests
        print("✅ requests imported successfully")
    except ImportError as e:
        print(f"❌ requests import failed: {e}")
    
    try:
        import aiohttp
        print("✅ aiohttp imported successfully")
    except ImportError as e:
        print(f"❌ aiohttp import failed: {e}")
    
    try:
        import sqlalchemy
        print("✅ sqlalchemy imported successfully")
    except ImportError as e:
        print(f"❌ sqlalchemy import failed: {e}")


def test_config():
    """Test configuration loading."""
    print("\n🔧 Testing Configuration")
    print("=" * 25)
    
    try:
        # Check if .env file exists
        if os.path.exists('.env'):
            print("✅ .env file found")
            
            # Read some config values
            with open('.env', 'r') as f:
                content = f.read()
                
            if 'TWITTER_BEARER_TOKEN=' in content:
                print("✅ Twitter API configured")
            if 'REDDIT_CLIENT_ID=' in content:
                print("✅ Reddit API configured")
            if 'NEWS_API_KEY=' in content:
                print("✅ News API configured")
            if 'ALPHA_VANTAGE_API_KEY=' in content:
                print("✅ Alpha Vantage API configured")
            if 'FINNHUB_API_KEY=' in content:
                print("✅ Finnhub API configured")
                
        else:
            print("❌ .env file not found")
            
    except Exception as e:
        print(f"❌ Config test failed: {e}")


async def test_yahoo_finance():
    """Test Yahoo Finance data collection (no API key needed)."""
    print("\n📈 Testing Yahoo Finance")
    print("=" * 25)
    
    try:
        import yfinance as yf
        print("✅ yfinance imported successfully")
        
        # Test getting data for AAPL
        ticker = yf.Ticker("AAPL")
        info = ticker.info
        
        if info and 'regularMarketPrice' in info:
            price = info['regularMarketPrice']
            print(f"✅ AAPL current price: ${price}")
        else:
            print("⚠️ Could not get AAPL price data")
            
    except Exception as e:
        print(f"❌ Yahoo Finance test failed: {e}")


async def test_basic_text_processing():
    """Test basic text processing without ML models."""
    print("\n📝 Testing Text Processing")
    print("=" * 30)
    
    try:
        # Simple text processing without ML
        test_text = "Apple stock $AAPL is performing great today! Bitcoin $BTC to the moon!"
        
        # Extract symbols manually
        import re
        symbols = re.findall(r'\$([A-Z]{1,5})', test_text)
        print(f"✅ Extracted symbols: {symbols}")
        
        # Basic sentiment (simple keyword matching)
        positive_words = ['great', 'good', 'moon', 'bull', 'up', 'rise']
        negative_words = ['bad', 'crash', 'down', 'bear', 'fall']
        
        text_lower = test_text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = "Positive"
        elif negative_count > positive_count:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
            
        print(f"✅ Basic sentiment: {sentiment}")
        
    except Exception as e:
        print(f"❌ Text processing test failed: {e}")


async def test_database():
    """Test basic database functionality."""
    print("\n🗄️ Testing Database")
    print("=" * 20)
    
    try:
        import sqlite3
        print("✅ sqlite3 available (built-in)")
        
        # Test creating a simple database
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute("INSERT INTO test_table (content) VALUES (?)", ("Test data",))
        conn.commit()
        
        cursor.execute("SELECT * FROM test_table")
        result = cursor.fetchone()
        
        if result:
            print(f"✅ Database test successful: {result}")
        else:
            print("❌ No data retrieved from database")
            
        conn.close()
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")


async def main():
    """Run all basic tests."""
    print("🚀 GoQuant Basic System Test")
    print("=" * 35)
    print("Testing core functionality without advanced ML...")
    print()
    
    # Run tests
    test_imports()
    test_config()
    await test_yahoo_finance()
    await test_basic_text_processing()
    await test_database()
    
    print("\n" + "=" * 35)
    print("🎯 Basic Test Summary")
    print("=" * 20)
    print("If most tests passed, your system is ready for basic functionality!")
    print()
    print("Next steps:")
    print("1. If tests passed: Try running a simple data collection")
    print("2. Install additional ML packages later if needed")
    print("3. Start with basic sentiment analysis using keyword matching")


if __name__ == "__main__":
    asyncio.run(main())
