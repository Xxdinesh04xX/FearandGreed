"""
Helper script to guide users through getting free API keys.
"""

import webbrowser
import time


def get_finnhub_api_key():
    """Guide user to get free Finnhub API key."""
    print("🔑 Getting Free Finnhub API Key")
    print("=" * 40)
    print("Finnhub provides free real-time stock data with 60 API calls/minute")
    print("This is perfect for our sentiment trading system!")
    print()
    
    print("Steps to get your FREE Finnhub API key:")
    print("1. Visit https://finnhub.io")
    print("2. Click 'Get free API key' or 'Sign up'")
    print("3. Create a free account with your email")
    print("4. Verify your email address")
    print("5. Go to your dashboard to find your API key")
    print()
    
    response = input("Would you like me to open the Finnhub website? (y/n): ")
    if response.lower() in ['y', 'yes']:
        print("Opening Finnhub website...")
        webbrowser.open("https://finnhub.io")
        time.sleep(2)
    
    print("\nOnce you have your API key:")
    print("1. Copy the API key")
    print("2. Open your .env file")
    print("3. Replace 'your_finnhub_key_here' with your actual key")
    print("   Example: FINNHUB_API_KEY=c123abc456def789")


def get_polygon_api_key():
    """Guide user to get free Polygon API key."""
    print("\n🔑 Getting Free Polygon.io API Key (Optional)")
    print("=" * 50)
    print("Polygon.io offers free tier with 5 API calls/minute")
    print("This can serve as an additional data source")
    print()
    
    print("Steps to get your FREE Polygon API key:")
    print("1. Visit https://polygon.io")
    print("2. Click 'Get API Key' or 'Sign up'")
    print("3. Choose the 'Free' plan")
    print("4. Create account and verify email")
    print("5. Find your API key in the dashboard")
    print()
    
    response = input("Would you like me to open the Polygon website? (y/n): ")
    if response.lower() in ['y', 'yes']:
        print("Opening Polygon website...")
        webbrowser.open("https://polygon.io")


def check_current_apis():
    """Check what APIs are already configured."""
    print("🔍 Checking Current API Configuration")
    print("=" * 40)
    
    try:
        with open('.env', 'r') as f:
            env_content = f.read()
        
        apis_status = {
            'Twitter': 'TWITTER_BEARER_TOKEN=' in env_content and 'your_twitter_bearer_token_here' not in env_content,
            'Reddit': 'REDDIT_CLIENT_ID=' in env_content and 'your_reddit_client_id_here' not in env_content,
            'NewsAPI': 'NEWS_API_KEY=' in env_content and 'your_newsapi_key_here' not in env_content,
            'Alpha Vantage': 'ALPHA_VANTAGE_API_KEY=' in env_content and 'your_alpha_vantage_key_here' not in env_content,
            'Finnhub': 'FINNHUB_API_KEY=' in env_content and 'your_finnhub_key_here' not in env_content,
        }
        
        print("Current API Status:")
        for api, configured in apis_status.items():
            status = "✅ Configured" if configured else "❌ Not configured"
            print(f"  {api:<15} {status}")
        
        configured_count = sum(apis_status.values())
        print(f"\nTotal configured: {configured_count}/5 APIs")
        
        if configured_count >= 3:
            print("✅ You have enough APIs configured to run the system!")
        else:
            print("⚠️  You should configure at least 3 APIs for best results")
            
    except FileNotFoundError:
        print("❌ .env file not found. Please copy .env.example to .env first")


def show_minimal_setup():
    """Show minimal API setup needed."""
    print("\n🎯 Minimal Setup for Testing")
    print("=" * 30)
    print("You can start with just these FREE APIs:")
    print("1. ✅ Twitter (you already have this)")
    print("2. ✅ Reddit (you already have this)")  
    print("3. ✅ NewsAPI (you already have this)")
    print("4. ✅ Alpha Vantage (you already have this)")
    print("5. 🆓 Finnhub (free - get this one!)")
    print()
    print("This gives you:")
    print("- Social media sentiment (Twitter + Reddit)")
    print("- News sentiment (NewsAPI)")
    print("- Financial data (Alpha Vantage + Finnhub + Yahoo Finance)")
    print()
    print("Yahoo Finance works without any API key, so you already have")
    print("financial data even without additional APIs!")


def main():
    """Main function."""
    print("🚀 GoQuant API Key Helper")
    print("=" * 30)
    print("This script will help you get the remaining free API keys")
    print("for your sentiment trading system.")
    print()
    
    # Check current status
    check_current_apis()
    
    # Show what's needed
    show_minimal_setup()
    
    # Get Finnhub key
    print("\n" + "="*60)
    get_finnhub_api_key()
    
    # Optionally get Polygon key
    print("\n" + "="*60)
    response = input("\nWould you like to also set up Polygon.io API? (y/n): ")
    if response.lower() in ['y', 'yes']:
        get_polygon_api_key()
    
    print("\n" + "="*60)
    print("🎉 Setup Complete!")
    print("=" * 15)
    print("After getting your API keys:")
    print("1. Update your .env file with the new keys")
    print("2. Run: goquant-sentiment run")
    print("3. Open: http://localhost:8050")
    print()
    print("The system will work with Yahoo Finance even without")
    print("additional API keys, but more sources = better signals!")


if __name__ == "__main__":
    main()
