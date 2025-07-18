"""
Twitter-friendly GoQuant Trader with proper rate limiting.
"""

import asyncio
import sqlite3
import time
import re
import aiohttp
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import from our existing modules
from simple_trader import SimpleConfig, SimpleDatabase, SimpleSentimentAnalyzer, SimpleSignalGenerator


class TwitterFriendlyDataCollector:
    """Data collector with Twitter-friendly rate limiting."""
    
    def __init__(self, config):
        self.config = config
        self.session = None
        self.last_twitter_call = 0
        self.twitter_calls_count = 0
        self.twitter_window_start = time.time()
    
    async def initialize(self):
        """Initialize HTTP session."""
        self.session = aiohttp.ClientSession()
    
    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
    
    def can_call_twitter(self):
        """Check if we can make a Twitter API call."""
        current_time = time.time()
        
        # Reset counter every 15 minutes (900 seconds)
        if current_time - self.twitter_window_start > 900:
            self.twitter_calls_count = 0
            self.twitter_window_start = current_time
        
        # Twitter allows 300 requests per 15 minutes
        # We'll be conservative and use max 50 per 15 minutes
        if self.twitter_calls_count >= 50:
            return False
        
        # Minimum 5 seconds between calls
        if current_time - self.last_twitter_call < 5:
            return False
        
        return True
    
    async def wait_for_twitter_rate_limit(self):
        """Wait if we need to respect Twitter rate limits."""
        current_time = time.time()
        
        # If we've hit our limit, wait until next window
        if self.twitter_calls_count >= 50:
            wait_time = 900 - (current_time - self.twitter_window_start)
            if wait_time > 0:
                print(f"⏳ Twitter rate limit reached. Waiting {wait_time/60:.1f} minutes...")
                await asyncio.sleep(wait_time)
                self.twitter_calls_count = 0
                self.twitter_window_start = time.time()
        
        # Minimum delay between calls
        time_since_last = current_time - self.last_twitter_call
        if time_since_last < 5:
            await asyncio.sleep(5 - time_since_last)
    
    async def collect_twitter_data(self) -> List[Dict[str, Any]]:
        """Collect Twitter data with proper rate limiting."""
        if not self.config.twitter_bearer_token:
            print("⚠️ Twitter Bearer Token not configured")
            return []
        
        if not self.can_call_twitter():
            print("⏳ Skipping Twitter due to rate limits")
            return []
        
        data = []
        try:
            await self.wait_for_twitter_rate_limit()
            
            # Use a simple, focused search
            query = "$BTC OR $ETH OR $AAPL OR $TSLA"
            
            url = "https://api.twitter.com/2/tweets/search/recent"
            headers = {
                'Authorization': f'Bearer {self.config.twitter_bearer_token}',
                'Content-Type': 'application/json'
            }
            params = {
                'query': query,
                'max_results': 15,  # Small number to conserve API calls
                'tweet.fields': 'created_at,author_id,public_metrics'
            }
            
            print("🐦 Calling Twitter API...")
            self.last_twitter_call = time.time()
            self.twitter_calls_count += 1
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    if 'data' in result:
                        for tweet in result['data']:
                            # Extract symbols from tweet
                            symbols = self._extract_symbols(tweet['text'])
                            
                            if symbols:  # Only include tweets with financial symbols
                                data.append({
                                    'source': 'twitter',
                                    'content': tweet['text'],
                                    'symbols': symbols,
                                    'author': f"user_{tweet['author_id']}",
                                    'url': f"https://twitter.com/i/status/{tweet['id']}",
                                    'metadata': {
                                        'retweet_count': tweet.get('public_metrics', {}).get('retweet_count', 0),
                                        'like_count': tweet.get('public_metrics', {}).get('like_count', 0),
                                        'created_at': tweet.get('created_at')
                                    }
                                })
                        
                        print(f"🐦 Collected {len(data)} relevant tweets")
                        print(f"📊 Twitter API calls used: {self.twitter_calls_count}/50 this window")
                    else:
                        print("🐦 No tweets found in response")
                        
                elif response.status == 429:
                    print("⚠️ Twitter rate limit hit! Will wait before next call.")
                    self.twitter_calls_count = 50  # Force wait
                    
                elif response.status == 401:
                    print("❌ Twitter authentication failed. Check your Bearer Token.")
                    
                elif response.status == 403:
                    print("❌ Twitter access forbidden. Check your API permissions.")
                    
                else:
                    print(f"❌ Twitter API error: {response.status}")
                    try:
                        error_text = await response.text()
                        print(f"   Error details: {error_text[:200]}...")
                    except:
                        pass
                    
        except Exception as e:
            print(f"❌ Twitter collection error: {e}")
        
        return data
    
    async def collect_reddit_data(self) -> List[Dict[str, Any]]:
        """Collect Reddit data."""
        if not self.config.reddit_client_id:
            return []
        
        data = []
        try:
            # Get Reddit access token
            auth_url = "https://www.reddit.com/api/v1/access_token"
            auth_data = {'grant_type': 'client_credentials'}
            auth_headers = {'User-Agent': 'GoQuant/1.0'}
            
            async with self.session.post(
                auth_url, 
                data=auth_data, 
                headers=auth_headers,
                auth=aiohttp.BasicAuth(self.config.reddit_client_id, self.config.reddit_client_secret)
            ) as response:
                if response.status == 200:
                    auth_result = await response.json()
                    access_token = auth_result['access_token']
                    
                    # Search financial subreddits
                    subreddits = ['investing', 'stocks', 'cryptocurrency']
                    
                    for subreddit in subreddits:
                        try:
                            url = f"https://oauth.reddit.com/r/{subreddit}/hot"
                            headers = {
                                'Authorization': f'Bearer {access_token}',
                                'User-Agent': 'GoQuant/1.0'
                            }
                            params = {'limit': 10}
                            
                            async with self.session.get(url, headers=headers, params=params) as sub_response:
                                if sub_response.status == 200:
                                    result = await sub_response.json()
                                    
                                    for post in result['data']['children']:
                                        post_data = post['data']
                                        content = f"{post_data['title']} {post_data.get('selftext', '')}"
                                        symbols = self._extract_symbols(content)
                                        
                                        if symbols:
                                            data.append({
                                                'source': 'reddit',
                                                'content': content,
                                                'symbols': symbols,
                                                'author': post_data['author'],
                                                'url': f"https://reddit.com{post_data['permalink']}",
                                                'metadata': {
                                                    'subreddit': subreddit,
                                                    'score': post_data['score'],
                                                    'num_comments': post_data['num_comments']
                                                }
                                            })
                            
                            await asyncio.sleep(1)  # Rate limiting
                            
                        except Exception as e:
                            print(f"❌ Error collecting from r/{subreddit}: {e}")
                    
                    print(f"🔴 Collected {len(data)} relevant Reddit posts")
                    
        except Exception as e:
            print(f"❌ Reddit collection error: {e}")
        
        return data
    
    async def collect_news_data(self) -> List[Dict[str, Any]]:
        """Collect news data."""
        if not self.config.news_api_key:
            return []
        
        data = []
        try:
            # Search for financial news
            keywords = ['bitcoin', 'apple stock', 'tesla stock']
            
            for keyword in keywords:
                url = "https://newsapi.org/v2/everything"
                params = {
                    'q': keyword,
                    'apiKey': self.config.news_api_key,
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': 10,
                    'from': (datetime.now() - timedelta(hours=12)).isoformat()
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        for article in result.get('articles', []):
                            content = f"{article['title']} {article.get('description', '')}"
                            symbols = self._extract_symbols(content)
                            
                            if symbols:
                                data.append({
                                    'source': 'news',
                                    'content': content,
                                    'symbols': symbols,
                                    'author': article.get('source', {}).get('name', 'Unknown'),
                                    'url': article['url'],
                                    'metadata': {
                                        'published_at': article['publishedAt'],
                                        'source_name': article.get('source', {}).get('name')
                                    }
                                })
                
                await asyncio.sleep(0.5)  # Rate limiting
            
            print(f"📰 Collected {len(data)} relevant news articles")
            
        except Exception as e:
            print(f"❌ News collection error: {e}")
        
        return data
    
    async def collect_financial_data(self) -> List[Dict[str, Any]]:
        """Collect financial data."""
        data = []
        
        # Yahoo Finance
        try:
            import yfinance as yf
            
            assets = ['BTC-USD', 'ETH-USD', 'AAPL', 'TSLA', 'SPY', 'NVDA']
            
            for symbol in assets:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    if info and 'regularMarketPrice' in info:
                        price = info['regularMarketPrice']
                        change = info.get('regularMarketChange', 0)
                        change_percent = info.get('regularMarketChangePercent', 0)
                        
                        # Create sentiment-like content based on price movement
                        if change_percent > 2:
                            sentiment_text = f"{symbol} is surging with strong bullish momentum!"
                        elif change_percent > 0:
                            sentiment_text = f"{symbol} showing positive gains today."
                        elif change_percent < -2:
                            sentiment_text = f"{symbol} experiencing significant decline."
                        else:
                            sentiment_text = f"{symbol} trading relatively stable."
                        
                        content = f"{sentiment_text} Price: ${price:.2f}, Change: {change:+.2f} ({change_percent:+.2f}%)"
                        
                        # Convert symbol for consistency
                        clean_symbol = symbol.replace('-USD', '')
                        
                        data.append({
                            'source': 'yahoo_finance',
                            'content': content,
                            'symbols': [clean_symbol],
                            'author': 'yahoo_finance',
                            'url': f'https://finance.yahoo.com/quote/{symbol}',
                            'metadata': {
                                'price': price,
                                'change': change,
                                'change_percent': change_percent
                            }
                        })
                        
                except Exception as e:
                    print(f"❌ Error collecting Yahoo Finance data for {symbol}: {e}")
                    
        except ImportError:
            print("❌ yfinance not available")
        
        print(f"📈 Collected {len(data)} financial data points")
        return data
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract financial symbols from text."""
        symbols = []
        
        # Find $SYMBOL patterns
        dollar_symbols = re.findall(r'\$([A-Z]{1,5})', text)
        symbols.extend(dollar_symbols)
        
        # Look for common symbols and keywords
        symbol_keywords = {
            'bitcoin': 'BTC',
            'btc': 'BTC',
            'ethereum': 'ETH',
            'eth': 'ETH',
            'apple': 'AAPL',
            'tesla': 'TSLA',
            'nvidia': 'NVDA',
            'spy': 'SPY'
        }
        
        text_lower = text.lower()
        for keyword, symbol in symbol_keywords.items():
            if keyword in text_lower and symbol not in symbols:
                symbols.append(symbol)
        
        return list(set(symbols))  # Remove duplicates


class TwitterFriendlyTrader:
    """Trading system with Twitter-friendly rate limiting."""
    
    def __init__(self):
        self.config = SimpleConfig()
        self.db = SimpleDatabase(self.config.database_path)
        self.sentiment_analyzer = SimpleSentimentAnalyzer()
        self.data_collector = TwitterFriendlyDataCollector(self.config)
        self.signal_generator = SimpleSignalGenerator(self.db)
        
        print("🚀 Twitter-Friendly GoQuant Trader initialized")
        print("🐦 Includes Twitter with proper rate limiting")
        
        # Show configured APIs
        apis = []
        if self.config.twitter_bearer_token: apis.append("Twitter")
        if self.config.reddit_client_id: apis.append("Reddit")
        if self.config.news_api_key: apis.append("NewsAPI")
        if self.config.alpha_vantage_api_key: apis.append("Alpha Vantage")
        if self.config.finnhub_api_key: apis.append("Finnhub")
        apis.append("Yahoo Finance")
        
        print(f"🔑 Active APIs: {', '.join(apis)} ({len(apis)} sources)")
    
    async def run_cycle(self):
        """Run trading cycle with Twitter included."""
        print(f"\n⏰ Running trading cycle at {datetime.now().strftime('%H:%M:%S')}")
        
        await self.data_collector.initialize()
        
        try:
            all_data = []
            
            # 1. Collect from all sources
            print("📊 Collecting data from all sources...")
            
            # Financial data (always works)
            financial_data = await self.data_collector.collect_financial_data()
            all_data.extend(financial_data)
            
            # Twitter data (with rate limiting)
            twitter_data = await self.data_collector.collect_twitter_data()
            all_data.extend(twitter_data)
            
            # Reddit data
            reddit_data = await self.data_collector.collect_reddit_data()
            all_data.extend(reddit_data)
            
            # News data
            news_data = await self.data_collector.collect_news_data()
            all_data.extend(news_data)
            
            print(f"📈 Total data collected: {len(all_data)} items")
            
            # 2. Process and store data
            if all_data:
                print("🧠 Processing sentiment analysis...")
                for data in all_data:
                    # Store raw data
                    raw_data_id = self.db.insert_raw_data(
                        source=data['source'],
                        content=data['content'],
                        symbols=data['symbols'],
                        author=data['author'],
                        url=data['url']
                    )
                    
                    # Analyze sentiment
                    sentiment_result = self.sentiment_analyzer.analyze_text(data['content'])
                    
                    # Store sentiment for each symbol
                    for symbol in data['symbols']:
                        self.db.insert_sentiment_data(
                            raw_data_id=raw_data_id,
                            symbol=symbol,
                            sentiment_score=sentiment_result['score'],
                            confidence=sentiment_result['confidence']
                        )
                
                # 3. Generate trading signals
                print("🎯 Generating trading signals...")
                signals = self.signal_generator.generate_signals()
                
                # Store signals
                for signal in signals:
                    self.db.insert_trading_signal(
                        symbol=signal['symbol'],
                        signal_type=signal['signal_type'],
                        strength=signal['strength'],
                        confidence=signal['confidence'],
                        sentiment_score=signal['sentiment_score']
                    )
                
                print(f"✅ Cycle complete: {len(all_data)} data points, {len(signals)} signals")
                return len(signals)
            else:
                print("⚠️ No data collected this cycle")
                return 0
                
        finally:
            await self.data_collector.close()
    
    async def run_continuous(self, interval_minutes: int = 20):
        """Run the trader continuously with Twitter-friendly intervals."""
        print(f"🔄 Starting continuous trading (every {interval_minutes} minutes)")
        print("🐦 Using Twitter-friendly intervals to respect rate limits")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                signals_generated = await self.run_cycle()
                
                print(f"💤 Waiting {interval_minutes} minutes until next cycle...")
                await asyncio.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping trader...")
        except Exception as e:
            print(f"❌ Error in continuous run: {e}")


async def main():
    """Main function for Twitter-friendly trader."""
    print("🚀 Twitter-Friendly GoQuant Sentiment Trader")
    print("=" * 50)
    print("🐦 Includes Twitter with proper rate limiting!")
    print()
    
    trader = TwitterFriendlyTrader()
    
    # Run one cycle first
    await trader.run_cycle()
    
    # Ask user if they want to run continuously
    print("\n" + "=" * 50)
    response = input("Run continuously with Twitter? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        await trader.run_continuous()
    else:
        print("✅ Cycle completed. Check simple_goquant.db for results!")
        print("💡 Run 'python dashboard.py' to view results in web interface")


if __name__ == "__main__":
    asyncio.run(main())
