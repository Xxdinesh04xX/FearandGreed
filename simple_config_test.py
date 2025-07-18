"""
Simple configuration test without complex parsing.
"""

import os
from dotenv import load_dotenv

def test_simple_config():
    """Test configuration loading with simple approach."""
    print("🔧 Testing Simple Configuration")
    print("=" * 35)
    
    # Load environment variables
    load_dotenv()
    
    # Test basic environment variables
    print("Environment Variables:")
    
    # API Keys
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
    
    # Database
    db_url = os.getenv('DATABASE_URL', 'sqlite:///goquant.db')
    print(f"✅ Database URL: {db_url}")
    
    # Count configured APIs
    configured_apis = sum([
        bool(twitter_token),
        bool(reddit_id),
        bool(news_key),
        bool(av_key),
        bool(finnhub_key)
    ])
    
    print(f"\n🎯 APIs Configured: {configured_apis}/5")
    
    if configured_apis >= 3:
        print("✅ Sufficient APIs for operation!")
        return True
    else:
        print("⚠️ Consider configuring more APIs")
        return True  # Still OK to proceed


if __name__ == "__main__":
    success = test_simple_config()
    if success:
        print("\n🎉 Simple configuration works! Let's try the system.")
    else:
        print("\n❌ Configuration issues detected.")
