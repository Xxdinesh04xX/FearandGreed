"""
Enhanced GoQuant Sentiment Trader with real data sources.
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


class EnhancedDataCollector:
    """Enhanced data collector with real APIs."""
    
    def __init__(self, config):
        self.config = config
        self.session = None
    
    async def initialize(self):
        """Initialize HTTP session."""
        self.session = aiohttp.ClientSession()
    
    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
    
    async def collect_twitter_data(self) -> List[Dict[str, Any]]:
        """Collect Twitter data using Bearer Token."""
        if not self.config.twitter_bearer_token:
            return []
        
        data = []
        try:
            # Search for financial keywords (reduced to avoid rate limits)
            keywords = ['$BTC', '$ETH', '$AAPL', '$TSLA']  # Reduced keywords
            query = ' OR '.join(keywords)

            url = "https://api.twitter.com/2/tweets/search/recent"
            headers = {
                'Authorization': f'Bearer {self.config.twitter_bearer_token}',
                'Content-Type': 'application/json'
            }
            params = {
                'query': query,
                'max_results': 20,  # Reduced from 50 to 20
                'tweet.fields': 'created_at,author_id,public_metrics'
            }
            
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
                elif response.status == 429:
                    print("⚠️ Twitter rate limit reached. Skipping Twitter for this cycle.")
                    print("💡 Twitter allows 300 requests per 15 minutes. Try again later.")
                elif response.status == 401:
                    print("❌ Twitter authentication failed. Check your Bearer Token.")
                elif response.status == 403:
                    print("❌ Twitter access forbidden. Check your API permissions.")
                else:
                    print(f"❌ Twitter API error: {response.status}")
                    if response.status != 200:
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
            auth_data = {
                'grant_type': 'client_credentials'
            }
            auth_headers = {
                'User-Agent': 'GoQuant/1.0'
            }
            
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
                    subreddits = ['investing', 'stocks', 'cryptocurrency', 'wallstreetbets']
                    
                    for subreddit in subreddits:
                        try:
                            url = f"https://oauth.reddit.com/r/{subreddit}/hot"
                            headers = {
                                'Authorization': f'Bearer {access_token}',
                                'User-Agent': 'GoQuant/1.0'
                            }
                            params = {'limit': 25}
                            
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
                                                    'num_comments': post_data['num_comments'],
                                                    'created_utc': post_data['created_utc']
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
            keywords = ['bitcoin', 'ethereum', 'apple', 'tesla', 'nvidia', 'stock market']
            
            for keyword in keywords:
                url = "https://newsapi.org/v2/everything"
                params = {
                    'q': keyword,
                    'apiKey': self.config.news_api_key,
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': 20,
                    'from': (datetime.now() - timedelta(hours=24)).isoformat()
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
        """Collect financial data from multiple sources."""
        data = []
        
        # Yahoo Finance
        try:
            import yfinance as yf
            
            assets_str = os.getenv('DEFAULT_ASSETS', 'BTC-USD,ETH-USD,SPY,AAPL,TSLA,NVDA')
            assets = [asset.strip() for asset in assets_str.split(',')]
            
            for symbol in assets:
                try:
                    # Convert crypto symbols for Yahoo Finance
                    yf_symbol = symbol
                    if symbol in ['BTC', 'BITCOIN']:
                        yf_symbol = 'BTC-USD'
                    elif symbol in ['ETH', 'ETHEREUM']:
                        yf_symbol = 'ETH-USD'
                    
                    ticker = yf.Ticker(yf_symbol)
                    info = ticker.info
                    
                    if info and 'regularMarketPrice' in info:
                        price = info['regularMarketPrice']
                        change = info.get('regularMarketChange', 0)
                        change_percent = info.get('regularMarketChangePercent', 0)
                        
                        content = f"{symbol} Price: ${price:.2f}, Change: {change:+.2f} ({change_percent:+.2f}%)"
                        
                        data.append({
                            'source': 'yahoo_finance',
                            'content': content,
                            'symbols': [symbol],
                            'author': 'yahoo_finance',
                            'url': f'https://finance.yahoo.com/quote/{yf_symbol}',
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
        
        # Finnhub API
        if self.config.finnhub_api_key:
            try:
                assets_str = os.getenv('DEFAULT_ASSETS', 'AAPL,TSLA,NVDA,SPY')
                stock_symbols = [s.strip() for s in assets_str.split(',') if s.strip() not in ['BTC', 'ETH']]
                
                for symbol in stock_symbols:
                    url = "https://finnhub.io/api/v1/quote"
                    params = {
                        'symbol': symbol,
                        'token': self.config.finnhub_api_key
                    }
                    
                    async with self.session.get(url, params=params) as response:
                        if response.status == 200:
                            result = await response.json()
                            
                            if 'c' in result and result['c']:
                                price = result['c']
                                change = price - result.get('pc', price)
                                change_percent = (change / result.get('pc', price)) * 100 if result.get('pc') else 0
                                
                                content = f"{symbol} (Finnhub) Price: ${price:.2f}, Change: {change:+.2f} ({change_percent:+.2f}%)"
                                
                                data.append({
                                    'source': 'finnhub',
                                    'content': content,
                                    'symbols': [symbol],
                                    'author': 'finnhub',
                                    'url': f'https://finnhub.io/quote/{symbol}',
                                    'metadata': {
                                        'price': price,
                                        'change': change,
                                        'change_percent': change_percent
                                    }
                                })
                    
                    await asyncio.sleep(0.1)  # Rate limiting
                    
            except Exception as e:
                print(f"❌ Finnhub collection error: {e}")
        
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


# Import the rest from simple_trader.py
from simple_trader import SimpleConfig, SimpleDatabase, SimpleSentimentAnalyzer, SimpleSignalGenerator


class EnhancedTrader:
    """Enhanced trading system with real data sources."""
    
    def __init__(self):
        self.config = SimpleConfig()
        self.db = SimpleDatabase(self.config.database_path)
        self.sentiment_analyzer = SimpleSentimentAnalyzer()
        self.data_collector = EnhancedDataCollector(self.config)
        self.signal_generator = SimpleSignalGenerator(self.db)
        
        print("🚀 Enhanced GoQuant Trader initialized")
        print(f"📊 Tracking assets: {self.config.default_assets}")
        
        # Show configured APIs
        apis = []
        if self.config.twitter_bearer_token: apis.append("Twitter")
        if self.config.reddit_client_id: apis.append("Reddit")
        if self.config.news_api_key: apis.append("NewsAPI")
        if self.config.alpha_vantage_api_key: apis.append("Alpha Vantage")
        if self.config.finnhub_api_key: apis.append("Finnhub")
        
        print(f"🔑 APIs configured: {', '.join(apis)} ({len(apis)}/5)")
    
    async def run_enhanced_cycle(self):
        """Run enhanced trading cycle with real data sources."""
        print(f"\n⏰ Running enhanced trading cycle at {datetime.now().strftime('%H:%M:%S')}")
        
        await self.data_collector.initialize()
        
        try:
            all_data = []
            
            # 1. Collect from all sources
            print("📊 Collecting data from all sources...")
            
            # Financial data
            financial_data = await self.data_collector.collect_financial_data()
            all_data.extend(financial_data)
            
            # Social media data
            twitter_data = await self.data_collector.collect_twitter_data()
            all_data.extend(twitter_data)
            
            reddit_data = await self.data_collector.collect_reddit_data()
            all_data.extend(reddit_data)
            
            # News data
            news_data = await self.data_collector.collect_news_data()
            all_data.extend(news_data)
            
            print(f"📈 Total data collected: {len(all_data)} items")
            
            # 2. Process and store data
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
            
            print(f"✅ Enhanced cycle complete: {len(all_data)} data points, {len(signals)} signals")
            return len(signals)
            
        finally:
            await self.data_collector.close()
    
    async def run_continuous(self, interval_minutes: int = 10):
        """Run the enhanced trader continuously."""
        print(f"🔄 Starting enhanced continuous trading (every {interval_minutes} minutes)")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                signals_generated = await self.run_enhanced_cycle()
                
                print(f"💤 Waiting {interval_minutes} minutes until next cycle...")
                await asyncio.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping enhanced trader...")
        except Exception as e:
            print(f"❌ Error in enhanced continuous run: {e}")


async def main():
    """Main function for enhanced trader."""
    print("🚀 Enhanced GoQuant Sentiment Trader")
    print("=" * 45)
    print("Now with real Twitter, Reddit, and News data!")
    print()
    
    trader = EnhancedTrader()
    
    # Run one enhanced cycle first
    await trader.run_enhanced_cycle()
    
    # Ask user if they want to run continuously
    print("\n" + "=" * 45)
    response = input("Run enhanced trader continuously? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        await trader.run_continuous()
    else:
        print("✅ Enhanced cycle completed. Check simple_goquant.db for results!")


if __name__ == "__main__":
    asyncio.run(main())
