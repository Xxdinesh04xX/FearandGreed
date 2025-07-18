"""
Standalone test that doesn't import the main package.
"""

import os
import asyncio
import sys
from dotenv import load_dotenv

# Add src to path so we can import directly
sys.path.insert(0, 'src')

def test_environment():
    """Test environment loading."""
    print("🔧 Testing Environment Variables")
    print("=" * 35)
    
    # Load environment variables
    load_dotenv()
    
    # Test basic environment variables
    twitter_token = os.getenv('TWITTER_BEARER_TOKEN')
    reddit_id = os.getenv('REDDIT_CLIENT_ID')
    news_key = os.getenv('NEWS_API_KEY')
    av_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    finnhub_key = os.getenv('FINNHUB_API_KEY')
    
    print(f"✅ Twitter: {'Configured' if twitter_token else 'Missing'}")
    print(f"✅ Reddit: {'Configured' if reddit_id else 'Missing'}")
    print(f"✅ News API: {'Configured' if news_key else 'Missing'}")
    print(f"✅ Alpha Vantage: {'Configured' if av_key else 'Missing'}")
    print(f"✅ Finnhub: {'Configured' if finnhub_key else 'Missing'}")
    
    # Asset configuration
    default_assets_str = os.getenv('DEFAULT_ASSETS', 'BTC,ETH,SPY,AAPL,TSLA,NVDA')
    default_assets = [asset.strip() for asset in default_assets_str.split(',')]
    print(f"✅ Default Assets: {default_assets}")
    
    # Count configured APIs
    configured_apis = sum([
        bool(twitter_token),
        bool(reddit_id),
        bool(news_key),
        bool(av_key),
        bool(finnhub_key)
    ])
    
    print(f"\n🎯 APIs Configured: {configured_apis}/5")
    return configured_apis >= 3


async def test_yahoo_finance():
    """Test Yahoo Finance without importing main package."""
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
            return True
        else:
            print("⚠️ Could not get AAPL price data")
            return False
            
    except Exception as e:
        print(f"❌ Yahoo Finance test failed: {e}")
        return False


async def test_basic_functionality():
    """Test basic functionality without complex imports."""
    print("\n🧠 Testing Basic Text Processing")
    print("=" * 35)
    
    try:
        import re
        
        # Test text processing
        test_text = "Apple stock $AAPL is performing great today! Bitcoin $BTC to the moon!"
        
        # Extract symbols
        symbols = re.findall(r'\$([A-Z]{1,5})', test_text)
        print(f"✅ Extracted symbols: {symbols}")
        
        # Basic sentiment analysis
        positive_words = ['great', 'good', 'moon', 'bull', 'up', 'rise', 'excellent', 'amazing']
        negative_words = ['bad', 'crash', 'down', 'bear', 'fall', 'terrible', 'awful']
        
        text_lower = test_text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = "Positive"
            score = 0.5
        elif negative_count > positive_count:
            sentiment = "Negative"
            score = -0.5
        else:
            sentiment = "Neutral"
            score = 0.0
            
        print(f"✅ Basic sentiment: {sentiment} (score: {score})")
        
        # Test multiple texts
        test_texts = [
            "Great earnings report! $AAPL stock is rising!",
            "Market crash incoming! Sell everything now!",
            "Bitcoin showing strong bullish momentum $BTC",
            "Neutral market conditions today"
        ]
        
        print("\nTesting multiple texts:")
        for text in test_texts:
            symbols = re.findall(r'\$([A-Z]{1,5})', text)
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                sentiment = "📈 Positive"
            elif neg_count > pos_count:
                sentiment = "📉 Negative"
            else:
                sentiment = "➡️ Neutral"
            
            symbols_str = f" ({', '.join(symbols)})" if symbols else ""
            print(f"  {sentiment}{symbols_str}: {text[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Text processing test failed: {e}")
        return False


async def test_database():
    """Test basic database functionality."""
    print("\n🗄️ Testing Database")
    print("=" * 20)
    
    try:
        import sqlite3
        print("✅ sqlite3 available")
        
        # Test creating a database
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        
        # Create tables similar to what we'll need
        cursor.execute('''
            CREATE TABLE raw_data (
                id INTEGER PRIMARY KEY,
                source TEXT,
                content TEXT,
                symbols TEXT,
                collected_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE sentiment_data (
                id INTEGER PRIMARY KEY,
                raw_data_id INTEGER,
                symbol TEXT,
                sentiment_score REAL,
                confidence REAL,
                processed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (raw_data_id) REFERENCES raw_data (id)
            )
        ''')
        
        # Insert test data
        cursor.execute("""
            INSERT INTO raw_data (source, content, symbols) 
            VALUES (?, ?, ?)
        """, ('test', 'Apple stock is performing great! $AAPL', 'AAPL'))
        
        raw_data_id = cursor.lastrowid
        
        cursor.execute("""
            INSERT INTO sentiment_data (raw_data_id, symbol, sentiment_score, confidence)
            VALUES (?, ?, ?, ?)
        """, (raw_data_id, 'AAPL', 0.8, 0.9))
        
        conn.commit()
        
        # Query data
        cursor.execute("""
            SELECT rd.content, sd.sentiment_score, sd.confidence 
            FROM raw_data rd 
            JOIN sentiment_data sd ON rd.id = sd.raw_data_id
        """)
        
        result = cursor.fetchone()
        if result:
            print(f"✅ Database test successful: {result}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False


async def main():
    """Run all standalone tests."""
    print("🚀 GoQuant Standalone System Test")
    print("=" * 40)
    print("Testing core functionality without complex imports...")
    print()
    
    # Run tests
    env_ok = test_environment()
    yahoo_ok = await test_yahoo_finance()
    text_ok = await test_basic_functionality()
    db_ok = await test_database()
    
    print("\n" + "=" * 40)
    print("🎯 Test Results Summary")
    print("=" * 25)
    
    results = {
        "Environment": env_ok,
        "Yahoo Finance": yahoo_ok,
        "Text Processing": text_ok,
        "Database": db_ok
    }
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name:<15} {status}")
    
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests >= 3:
        print("\n🎉 Core functionality is working!")
        print("✅ Ready to build a simplified trading system")
        print("\nNext steps:")
        print("1. Create simplified data collectors")
        print("2. Implement basic sentiment analysis")
        print("3. Build simple trading signals")
        print("4. Create basic dashboard")
    else:
        print("\n⚠️ Some core functionality needs attention")
        print("Please check the failed tests above")


if __name__ == "__main__":
    asyncio.run(main())
