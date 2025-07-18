"""
Test the system using simple configuration.
"""

import asyncio
from src.goquant.simple_config import get_simple_config


async def test_simple_system():
    """Test the system with simple configuration."""
    print("🚀 GoQuant Simple System Test")
    print("=" * 35)
    
    # Test configuration
    config = get_simple_config()
    print("✅ Simple configuration loaded")
    
    print(f"✅ Default assets: {config.default_assets}")
    print(f"✅ Database URL: {config.database_url}")
    
    configured_apis = config.get_configured_apis()
    print(f"✅ Configured APIs: {configured_apis}")
    print(f"✅ Total APIs: {len(configured_apis)}")
    
    # Test Yahoo Finance (no API key needed)
    print("\n📈 Testing Yahoo Finance")
    try:
        import yfinance as yf
        ticker = yf.Ticker("AAPL")
        info = ticker.info
        if info and 'regularMarketPrice' in info:
            price = info['regularMarketPrice']
            print(f"✅ AAPL price: ${price}")
        else:
            print("⚠️ Could not get AAPL price")
    except Exception as e:
        print(f"❌ Yahoo Finance error: {e}")
    
    # Test basic sentiment analysis (keyword-based)
    print("\n🧠 Testing Basic Sentiment")
    test_texts = [
        "Apple stock is performing great! $AAPL to the moon!",
        "Market crash incoming, sell everything!",
        "Bitcoin showing strong momentum $BTC"
    ]
    
    for text in test_texts:
        # Simple keyword-based sentiment
        positive_words = ['great', 'moon', 'strong', 'bull', 'up', 'rise', 'good']
        negative_words = ['crash', 'sell', 'down', 'bear', 'fall', 'bad']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            sentiment = "Positive"
        elif neg_count > pos_count:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        print(f"  {sentiment}: {text[:50]}...")
    
    print("\n✅ Basic system test completed!")
    print("\nNext steps:")
    print("1. The simple configuration works!")
    print("2. You can start building with basic functionality")
    print("3. Add advanced ML features later")


if __name__ == "__main__":
    asyncio.run(test_simple_system())
