"""
GoQuant Trader that works without Twitter (when rate limited).
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


class NoTwitterDataCollector:
    """Data collector that works without Twitter."""
    
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
    
    async def collect_reddit_data(self) -> List[Dict[str, Any]]:
        """Collect Reddit data."""
        if not self.config.reddit_client_id:
            print("⚠️ Reddit API not configured")
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
                            params = {'limit': 15}  # Reduced limit
                            
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
                                else:
                                    print(f"⚠️ Reddit error for r/{subreddit}: {sub_response.status}")
                            
                            await asyncio.sleep(2)  # Longer delay for Reddit
                            
                        except Exception as e:
                            print(f"❌ Error collecting from r/{subreddit}: {e}")
                    
                    print(f"🔴 Collected {len(data)} relevant Reddit posts")
                else:
                    print(f"❌ Reddit authentication failed: {response.status}")
                    
        except Exception as e:
            print(f"❌ Reddit collection error: {e}")
        
        return data
    
    async def collect_news_data(self) -> List[Dict[str, Any]]:
        """Collect news data."""
        if not self.config.news_api_key:
            print("⚠️ News API not configured")
            return []
        
        data = []
        try:
            # Search for financial news with fewer requests
            keywords = ['bitcoin', 'apple stock', 'tesla stock']  # Reduced keywords
            
            for keyword in keywords:
                url = "https://newsapi.org/v2/everything"
                params = {
                    'q': keyword,
                    'apiKey': self.config.news_api_key,
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': 10,  # Reduced page size
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
                    else:
                        print(f"⚠️ News API error for '{keyword}': {response.status}")
                
                await asyncio.sleep(1)  # Rate limiting
            
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
                stock_symbols = ['AAPL', 'TSLA', 'NVDA', 'SPY']
                
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
                                
                                # Create sentiment-like content
                                if change_percent > 1:
                                    sentiment_text = f"{symbol} showing strong performance"
                                elif change_percent < -1:
                                    sentiment_text = f"{symbol} under pressure"
                                else:
                                    sentiment_text = f"{symbol} trading steady"
                                
                                content = f"{sentiment_text} (Finnhub) Price: ${price:.2f}, Change: {change:+.2f} ({change_percent:+.2f}%)"
                                
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
                        else:
                            print(f"⚠️ Finnhub error for {symbol}: {response.status}")
                    
                    await asyncio.sleep(0.2)  # Rate limiting
                    
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


class NoTwitterTrader:
    """Trading system that works without Twitter."""
    
    def __init__(self):
        self.config = SimpleConfig()
        self.db = SimpleDatabase(self.config.database_path)
        self.sentiment_analyzer = SimpleSentimentAnalyzer()
        self.data_collector = NoTwitterDataCollector(self.config)
        self.signal_generator = SimpleSignalGenerator(self.db)
        
        print("🚀 GoQuant Trader (No Twitter) initialized")
        print("📊 This version works without Twitter API limits")
        
        # Show configured APIs
        apis = []
        if self.config.reddit_client_id: apis.append("Reddit")
        if self.config.news_api_key: apis.append("NewsAPI")
        if self.config.alpha_vantage_api_key: apis.append("Alpha Vantage")
        if self.config.finnhub_api_key: apis.append("Finnhub")
        apis.append("Yahoo Finance")  # Always available
        
        print(f"🔑 Active APIs: {', '.join(apis)} ({len(apis)} sources)")
    
    async def run_cycle(self):
        """Run trading cycle without Twitter."""
        print(f"\n⏰ Running trading cycle at {datetime.now().strftime('%H:%M:%S')}")
        
        await self.data_collector.initialize()
        
        try:
            all_data = []
            
            # 1. Collect from available sources
            print("📊 Collecting data from available sources...")
            
            # Financial data (always works)
            financial_data = await self.data_collector.collect_financial_data()
            all_data.extend(financial_data)
            
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
    
    async def run_continuous(self, interval_minutes: int = 15):
        """Run the trader continuously with longer intervals."""
        print(f"🔄 Starting continuous trading (every {interval_minutes} minutes)")
        print("💡 Using longer intervals to avoid rate limits")
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
    """Main function for no-Twitter trader."""
    print("🚀 GoQuant Sentiment Trader (No Twitter)")
    print("=" * 45)
    print("This version works around Twitter rate limits!")
    print()
    
    trader = NoTwitterTrader()
    
    # Run one cycle first
    await trader.run_cycle()
    
    # Ask user if they want to run continuously
    print("\n" + "=" * 45)
    response = input("Run continuously? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        await trader.run_continuous()
    else:
        print("✅ Cycle completed. Check simple_goquant.db for results!")
        print("💡 Run 'python dashboard.py' to view results in web interface")


if __name__ == "__main__":
    asyncio.run(main())
