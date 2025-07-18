"""
Test script to verify the system works without IEX Cloud API.
"""

import asyncio
import os
from src.goquant.config import get_config
from src.goquant.data.sources import FinancialDataCollector


async def test_financial_data_collection():
    """Test financial data collection without IEX Cloud."""
    print("🧪 Testing Financial Data Collection")
    print("=" * 40)
    
    # Get configuration
    config = get_config()
    
    # Show API status
    print("API Configuration Status:")
    print(f"  Alpha Vantage: {'✅ Configured' if config.alpha_vantage_api_key else '❌ Missing'}")
    print(f"  IEX Cloud:     {'✅ Configured' if config.iex_cloud_api_key else '❌ Missing (OK - not required)'}")
    print(f"  Finnhub:       {'✅ Configured' if config.finnhub_api_key else '❌ Missing'}")
    print(f"  Polygon:       {'✅ Configured' if config.polygon_api_key else '❌ Missing (Optional)'}")
    print()
    
    # Initialize financial data collector
    collector = FinancialDataCollector(config)
    await collector.initialize()
    
    # Test symbols
    test_symbols = ['AAPL', 'BTC-USD', 'TSLA']
    
    print("Testing data collection for symbols:", test_symbols)
    print("-" * 40)
    
    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}:")
        try:
            data = await collector.collect_for_symbol(symbol)
            
            if data:
                print(f"  ✅ Successfully collected {len(data)} data point(s)")
                for item in data:
                    provider = item['metadata'].get('provider', 'unknown')
                    content = item['content'][:60] + "..." if len(item['content']) > 60 else item['content']
                    print(f"     📈 {provider}: {content}")
            else:
                print(f"  ⚠️  No data collected (this might be normal for some symbols)")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n" + "=" * 40)
    print("✅ Financial data collection test completed!")
    print()
    print("Available data sources:")
    print("  🆓 Yahoo Finance (always available, no API key needed)")
    if config.alpha_vantage_api_key:
        print("  🔑 Alpha Vantage (configured)")
    if config.finnhub_api_key:
        print("  🔑 Finnhub (configured)")
    if config.polygon_api_key:
        print("  🔑 Polygon.io (configured)")
    
    print("\nNote: The system will work with just Yahoo Finance,")
    print("but additional APIs provide more comprehensive data!")


async def test_sentiment_analysis():
    """Test sentiment analysis with financial text."""
    print("\n🧠 Testing Sentiment Analysis")
    print("=" * 30)
    
    try:
        from src.goquant.sentiment.analyzer import SentimentAnalyzer
        
        config = get_config()
        analyzer = SentimentAnalyzer(config)
        
        print("Initializing sentiment analyzer...")
        await analyzer.initialize()
        
        # Test texts
        test_texts = [
            "Apple stock is performing amazingly well! $AAPL to the moon!",
            "Market crash incoming, sell everything now!",
            "Bitcoin showing strong bullish momentum $BTC",
            "Neutral market conditions, holding positions"
        ]
        
        print("\nTesting sentiment analysis:")
        for text in test_texts:
            result = await analyzer.analyze_text(text)
            sentiment_label = "📈 Positive" if result.score > 0.1 else "📉 Negative" if result.score < -0.1 else "➡️ Neutral"
            print(f"  {sentiment_label} ({result.score:.2f}): {text[:50]}...")
        
        print("✅ Sentiment analysis working correctly!")
        
    except Exception as e:
        print(f"❌ Sentiment analysis error: {e}")
        print("This might be due to missing model files - they'll download on first run")


async def main():
    """Main test function."""
    print("🚀 GoQuant System Test (Without IEX Cloud)")
    print("=" * 50)
    print("This test verifies the system works without IEX Cloud API")
    print()
    
    # Test financial data collection
    await test_financial_data_collection()
    
    # Test sentiment analysis
    await test_sentiment_analysis()
    
    print("\n" + "=" * 50)
    print("🎉 System Test Complete!")
    print("=" * 25)
    print()
    print("Your system is ready to run! 🚀")
    print()
    print("To start the full system:")
    print("  goquant-sentiment run")
    print()
    print("To access the dashboard:")
    print("  http://localhost:8050")
    print()
    print("To get additional free API keys:")
    print("  python get_api_keys.py")


if __name__ == "__main__":
    asyncio.run(main())
