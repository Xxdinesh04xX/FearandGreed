"""
Test configuration loading.
"""

def test_config():
    """Test if configuration loads properly."""
    print("🔧 Testing Configuration Loading")
    print("=" * 35)
    
    try:
        from src.goquant.config import get_config
        
        config = get_config()
        print("✅ Configuration loaded successfully!")
        
        print(f"✅ Default assets: {config.default_assets}")
        print(f"✅ Database URL: {config.database_url}")
        print(f"✅ Log level: {config.log_level}")
        
        # Check API keys
        apis_configured = 0
        if config.twitter_bearer_token:
            print("✅ Twitter API configured")
            apis_configured += 1
        if config.reddit_client_id:
            print("✅ Reddit API configured")
            apis_configured += 1
        if config.news_api_key:
            print("✅ News API configured")
            apis_configured += 1
        if config.alpha_vantage_api_key:
            print("✅ Alpha Vantage API configured")
            apis_configured += 1
        if config.finnhub_api_key:
            print("✅ Finnhub API configured")
            apis_configured += 1
            
        print(f"\n🎯 Total APIs configured: {apis_configured}/5")
        
        if apis_configured >= 3:
            print("✅ Sufficient APIs configured for operation!")
        else:
            print("⚠️ Consider configuring more APIs for better data coverage")
            
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False


if __name__ == "__main__":
    success = test_config()
    if success:
        print("\n🎉 Configuration test passed! Ready to test the full system.")
    else:
        print("\n❌ Configuration test failed. Please check your .env file.")
