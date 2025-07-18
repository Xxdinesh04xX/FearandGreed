"""
Data source collectors for various APIs.
"""

import asyncio
import aiohttp
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from ..config import Config
from ..utils.logger import get_logger
from ..utils.text_processor import get_text_processor


class BaseCollector(ABC):
    """Base class for all data collectors."""
    
    def __init__(self, config: Config):
        """Initialize the base collector."""
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.text_processor = get_text_processor()
        self._initialized = False
        self.last_collection_time = None
        self.total_collected = 0
        self.error_count = 0
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the collector."""
        pass
    
    @abstractmethod
    async def collect(self) -> List[Dict[str, Any]]:
        """Collect data from the source."""
        pass
    
    async def collect_for_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """Collect data for a specific symbol (optional implementation)."""
        return []


class TwitterCollector(BaseCollector):
    """
    Twitter data collector using Twitter API v2.
    
    Collects tweets mentioning financial symbols and relevant hashtags.
    """
    
    def __init__(self, config: Config):
        """Initialize the Twitter collector."""
        super().__init__(config)
        self.bearer_token = config.twitter_bearer_token
        self.base_url = "https://api.twitter.com/2"
        self.session = None
        
        # Search queries for financial content
        self.financial_queries = [
            "($BTC OR $ETH OR $AAPL OR $TSLA OR $SPY) -is:retweet lang:en",
            "(bitcoin OR ethereum OR stocks OR trading) -is:retweet lang:en",
            "(#stocks OR #trading OR #crypto OR #bitcoin) -is:retweet lang:en"
        ]
    
    async def initialize(self) -> None:
        """Initialize the Twitter collector."""
        if not self.bearer_token:
            raise ValueError("Twitter bearer token is required")
        
        # Create HTTP session with authentication
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json"
        }
        
        self.session = aiohttp.ClientSession(headers=headers)
        self._initialized = True
        self.logger.info("Twitter collector initialized")
    
    async def collect(self) -> List[Dict[str, Any]]:
        """Collect recent tweets from all financial queries."""
        if not self._initialized:
            await self.initialize()
        
        all_tweets = []
        
        for query in self.financial_queries:
            try:
                tweets = await self._search_tweets(query, max_results=100)
                all_tweets.extend(tweets)
                
                # Small delay between queries
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error searching tweets for query '{query}': {e}")
                self.error_count += 1
        
        self.last_collection_time = datetime.utcnow()
        self.total_collected += len(all_tweets)
        
        return all_tweets
    
    async def collect_for_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """Collect tweets for a specific symbol."""
        if not self._initialized:
            await self.initialize()
        
        # Create symbol-specific query
        query = f"${symbol} -is:retweet lang:en"
        
        try:
            tweets = await self._search_tweets(query, max_results=50)
            return tweets
        except Exception as e:
            self.logger.error(f"Error collecting tweets for {symbol}: {e}")
            self.error_count += 1
            return []
    
    async def _search_tweets(self, query: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Search for tweets using the Twitter API."""
        url = f"{self.base_url}/tweets/search/recent"
        
        params = {
            "query": query,
            "max_results": min(max_results, 100),  # API limit
            "tweet.fields": "created_at,author_id,public_metrics,context_annotations",
            "user.fields": "username,verified,public_metrics",
            "expansions": "author_id"
        }
        
        async with self.session.get(url, params=params) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Twitter API error {response.status}: {error_text}")
            
            data = await response.json()
            
            # Process tweets
            tweets = []
            tweet_data = data.get('data', [])
            users_data = {user['id']: user for user in data.get('includes', {}).get('users', [])}
            
            for tweet in tweet_data:
                # Extract financial symbols
                symbols = self.text_processor.extract_financial_symbols(tweet['text'])
                
                # Only include tweets with financial content
                if symbols or self.text_processor.is_financial_text(tweet['text']):
                    user = users_data.get(tweet['author_id'], {})
                    
                    processed_tweet = {
                        'source': 'twitter',
                        'source_id': tweet['id'],
                        'content': tweet['text'],
                        'author': user.get('username'),
                        'url': f"https://twitter.com/i/status/{tweet['id']}",
                        'metadata': {
                            'created_at': tweet['created_at'],
                            'author_id': tweet['author_id'],
                            'public_metrics': tweet.get('public_metrics', {}),
                            'symbols': symbols,
                            'verified': user.get('verified', False)
                        },
                        'collected_at': datetime.utcnow()
                    }
                    tweets.append(processed_tweet)
            
            return tweets


class RedditCollector(BaseCollector):
    """
    Reddit data collector using PRAW (Python Reddit API Wrapper).
    
    Collects posts and comments from financial subreddits.
    """
    
    def __init__(self, config: Config):
        """Initialize the Reddit collector."""
        super().__init__(config)
        self.client_id = config.reddit_client_id
        self.client_secret = config.reddit_client_secret
        self.user_agent = config.reddit_user_agent
        self.reddit = None
        
        # Financial subreddits to monitor
        self.subreddits = [
            'wallstreetbets', 'investing', 'stocks', 'SecurityAnalysis',
            'ValueInvesting', 'cryptocurrency', 'Bitcoin', 'ethereum'
        ]
    
    async def initialize(self) -> None:
        """Initialize the Reddit collector."""
        try:
            import praw
            
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent
            )
            
            # Test the connection
            self.reddit.user.me()
            
            self._initialized = True
            self.logger.info("Reddit collector initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Reddit collector: {e}")
            raise
    
    async def collect(self) -> List[Dict[str, Any]]:
        """Collect recent posts from financial subreddits."""
        if not self._initialized:
            await self.initialize()
        
        all_posts = []
        
        for subreddit_name in self.subreddits:
            try:
                posts = await self._collect_subreddit_posts(subreddit_name, limit=50)
                all_posts.extend(posts)
                
                # Small delay between subreddits
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error collecting from r/{subreddit_name}: {e}")
                self.error_count += 1
        
        self.last_collection_time = datetime.utcnow()
        self.total_collected += len(all_posts)
        
        return all_posts
    
    async def _collect_subreddit_posts(self, subreddit_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Collect posts from a specific subreddit."""
        posts = []

        try:
            # Run Reddit API calls in executor to avoid blocking
            loop = asyncio.get_event_loop()
            subreddit = await loop.run_in_executor(None, self.reddit.subreddit, subreddit_name)

            # Get hot posts (run in executor)
            hot_posts = await loop.run_in_executor(None, lambda: list(subreddit.hot(limit=limit)))

            for submission in hot_posts:
                # Skip stickied posts
                if submission.stickied:
                    continue

                # Get submission content safely
                try:
                    title = submission.title or ""
                    selftext = submission.selftext or ""
                    content = f"{title}\n\n{selftext}".strip()

                    if not content:
                        continue

                    # Extract financial symbols
                    symbols = self.text_processor.extract_financial_symbols(content)

                    # Only include posts with financial content or symbols
                    if symbols or self.text_processor.is_financial_text(content):
                        post_data = {
                            'source': 'reddit',
                            'source_id': submission.id,
                            'content': content,
                            'author': str(submission.author) if submission.author else '[deleted]',
                            'url': f"https://reddit.com{submission.permalink}",
                            'metadata': {
                                'subreddit': subreddit_name,
                                'score': getattr(submission, 'score', 0),
                                'upvote_ratio': getattr(submission, 'upvote_ratio', 0.5),
                                'num_comments': getattr(submission, 'num_comments', 0),
                                'created_utc': getattr(submission, 'created_utc', time.time()),
                                'symbols': symbols,
                                'title': title,
                                'is_self': getattr(submission, 'is_self', False)
                            },
                            'collected_at': datetime.utcnow()
                        }
                        posts.append(post_data)

                except Exception as e:
                    self.logger.warning(f"Error processing Reddit post {submission.id}: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error collecting from r/{subreddit_name}: {e}")
            raise

        return posts


class NewsCollector(BaseCollector):
    """
    News data collector using NewsAPI.

    Collects financial news articles from various sources.
    """

    def __init__(self, config: Config):
        """Initialize the News collector."""
        super().__init__(config)
        self.api_key = config.news_api_key
        self.base_url = "https://newsapi.org/v2"
        self.session = None

        # Financial keywords for news search
        self.financial_keywords = [
            "stock market", "trading", "investment", "bitcoin", "cryptocurrency",
            "earnings", "financial", "economy", "fed", "interest rates"
        ]

    async def initialize(self) -> None:
        """Initialize the News collector."""
        if not self.api_key:
            raise ValueError("NewsAPI key is required")

        self.session = aiohttp.ClientSession()
        self._initialized = True
        self.logger.info("News collector initialized")

    async def collect(self) -> List[Dict[str, Any]]:
        """Collect recent financial news articles."""
        if not self._initialized:
            await self.initialize()

        all_articles = []

        # Collect from different categories
        for keyword in self.financial_keywords[:3]:  # Limit to avoid rate limits
            try:
                articles = await self._search_news(keyword, page_size=20)
                all_articles.extend(articles)

                # Delay between requests
                await asyncio.sleep(1)

            except Exception as e:
                self.logger.error(f"Error searching news for '{keyword}': {e}")
                self.error_count += 1

        self.last_collection_time = datetime.utcnow()
        self.total_collected += len(all_articles)

        return all_articles

    async def _search_news(self, query: str, page_size: int = 20) -> List[Dict[str, Any]]:
        """Search for news articles using NewsAPI."""
        url = f"{self.base_url}/everything"

        # Search for articles from the last 24 hours
        from_date = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')

        params = {
            'q': query,
            'from': from_date,
            'sortBy': 'publishedAt',
            'pageSize': page_size,
            'language': 'en',
            'apiKey': self.api_key
        }

        async with self.session.get(url, params=params) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"NewsAPI error {response.status}: {error_text}")

            data = await response.json()

            articles = []
            for article in data.get('articles', []):
                # Skip articles without content
                if not article.get('content') or article['content'] == '[Removed]':
                    continue

                # Extract financial symbols from title and description
                text_content = f"{article.get('title', '')} {article.get('description', '')}"
                symbols = self.text_processor.extract_financial_symbols(text_content)

                processed_article = {
                    'source': 'news',
                    'source_id': article.get('url', ''),
                    'content': f"{article.get('title', '')}\n\n{article.get('description', '')}\n\n{article.get('content', '')}",
                    'author': article.get('author'),
                    'url': article.get('url'),
                    'metadata': {
                        'published_at': article.get('publishedAt'),
                        'source_name': article.get('source', {}).get('name'),
                        'symbols': symbols,
                        'query': query
                    },
                    'collected_at': datetime.utcnow()
                }
                articles.append(processed_article)

            return articles


class FinancialDataCollector(BaseCollector):
    """
    Financial data collector for price and market data.

    Collects OHLCV data and market indicators from financial APIs.
    """

    def __init__(self, config: Config):
        """Initialize the Financial data collector."""
        super().__init__(config)
        self.alpha_vantage_key = config.alpha_vantage_api_key
        self.iex_cloud_key = config.iex_cloud_api_key
        self.finnhub_key = config.finnhub_api_key
        self.polygon_key = config.polygon_api_key
        self.session = None

        # Default symbols to track
        self.default_symbols = config.default_assets

    async def initialize(self) -> None:
        """Initialize the Financial data collector."""
        self.session = aiohttp.ClientSession()
        self._initialized = True
        self.logger.info("Financial data collector initialized")

    async def collect(self) -> List[Dict[str, Any]]:
        """Collect price data for default symbols."""
        if not self._initialized:
            await self.initialize()

        all_data = []

        for symbol in self.default_symbols:
            try:
                price_data = await self.collect_for_symbol(symbol)
                all_data.extend(price_data)

                # Delay between requests
                await asyncio.sleep(1)

            except Exception as e:
                self.logger.error(f"Error collecting price data for {symbol}: {e}")
                self.error_count += 1

        self.last_collection_time = datetime.utcnow()
        self.total_collected += len(all_data)

        return all_data

    async def collect_for_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """Collect price data for a specific symbol."""
        if not self._initialized:
            await self.initialize()

        all_data = []

        # Try Yahoo Finance first (free and reliable)
        try:
            yahoo_data = await self._collect_yfinance_data(symbol)
            all_data.extend(yahoo_data)
        except Exception as e:
            self.logger.error(f"Error collecting Yahoo Finance data for {symbol}: {e}")

        # Try Alpha Vantage as backup/additional source
        if self.alpha_vantage_key:
            try:
                av_data = await self._collect_alpha_vantage_data(symbol)
                all_data.extend(av_data)
                # Small delay to respect Alpha Vantage rate limits
                await asyncio.sleep(0.2)
            except Exception as e:
                self.logger.error(f"Error collecting Alpha Vantage data for {symbol}: {e}")

        # Try Finnhub as additional free source
        if self.finnhub_key:
            try:
                finnhub_data = await self._collect_finnhub_data(symbol)
                all_data.extend(finnhub_data)
                # Small delay to respect rate limits
                await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error collecting Finnhub data for {symbol}: {e}")

        # If no data from any source, log a warning
        if not all_data:
            self.logger.warning(f"No financial data collected for {symbol} from any source")

        return all_data

    async def _collect_yfinance_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Collect data using Yahoo Finance API."""
        try:
            # Yahoo Finance API endpoint
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                'interval': '1m',
                'range': '1d',
                'includePrePost': 'true'
            }

            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    self.logger.error(f"Yahoo Finance API error {response.status} for {symbol}")
                    return []

                data = await response.json()

                if 'chart' not in data or not data['chart']['result']:
                    self.logger.warning(f"No data returned for {symbol}")
                    return []

                result = data['chart']['result'][0]
                meta = result['meta']
                timestamps = result['timestamp']
                indicators = result['indicators']['quote'][0]

                # Get the latest data point
                if not timestamps:
                    return []

                latest_idx = -1
                latest_timestamp = timestamps[latest_idx]

                price_data = {
                    'source': 'financial',
                    'source_id': f"{symbol}_{latest_timestamp}",
                    'content': f"Price: ${indicators['close'][latest_idx]:.2f}, Volume: {indicators['volume'][latest_idx] or 0:,}",
                    'author': 'yahoo_finance',
                    'url': f"https://finance.yahoo.com/quote/{symbol}",
                    'metadata': {
                        'symbol': symbol,
                        'timestamp': datetime.fromtimestamp(latest_timestamp).isoformat(),
                        'open': indicators['open'][latest_idx],
                        'high': indicators['high'][latest_idx],
                        'low': indicators['low'][latest_idx],
                        'close': indicators['close'][latest_idx],
                        'volume': indicators['volume'][latest_idx],
                        'currency': meta.get('currency', 'USD'),
                        'exchange': meta.get('exchangeName', 'Unknown'),
                        'data_type': 'price',
                        'provider': 'yahoo_finance'
                    },
                    'collected_at': datetime.utcnow()
                }

                return [price_data]

        except Exception as e:
            self.logger.error(f"Error collecting Yahoo Finance data for {symbol}: {e}")
            return []

    async def _collect_alpha_vantage_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Collect data using Alpha Vantage API."""
        if not self.alpha_vantage_key:
            return []

        try:
            # Alpha Vantage real-time quote endpoint
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key
            }

            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    self.logger.error(f"Alpha Vantage API error {response.status} for {symbol}")
                    return []

                data = await response.json()

                if 'Global Quote' not in data:
                    self.logger.warning(f"No quote data returned for {symbol} from Alpha Vantage")
                    return []

                quote = data['Global Quote']

                if not quote:
                    return []

                price_data = {
                    'source': 'financial',
                    'source_id': f"{symbol}_av_{datetime.utcnow().isoformat()}",
                    'content': f"Price: ${float(quote['05. price']):.2f}, Change: {quote['09. change']} ({quote['10. change percent']})",
                    'author': 'alpha_vantage',
                    'url': f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}",
                    'metadata': {
                        'symbol': quote['01. symbol'],
                        'timestamp': quote['07. latest trading day'],
                        'open': float(quote['02. open']),
                        'high': float(quote['03. high']),
                        'low': float(quote['04. low']),
                        'close': float(quote['05. price']),
                        'volume': int(quote['06. volume']),
                        'change': float(quote['09. change']),
                        'change_percent': quote['10. change percent'],
                        'data_type': 'price',
                        'provider': 'alpha_vantage'
                    },
                    'collected_at': datetime.utcnow()
                }

                return [price_data]

        except Exception as e:
            self.logger.error(f"Error collecting Alpha Vantage data for {symbol}: {e}")
            return []

    async def _collect_finnhub_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Collect data using Finnhub API (free tier available)."""
        if not self.finnhub_key:
            return []

        try:
            # Finnhub real-time quote endpoint
            url = "https://finnhub.io/api/v1/quote"
            params = {
                'symbol': symbol,
                'token': self.finnhub_key
            }

            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    self.logger.error(f"Finnhub API error {response.status} for {symbol}")
                    return []

                data = await response.json()

                if 'error' in data:
                    self.logger.warning(f"Finnhub API error for {symbol}: {data['error']}")
                    return []

                # Finnhub returns: c (current price), h (high), l (low), o (open), pc (previous close)
                if 'c' not in data or data['c'] is None:
                    return []

                current_price = data['c']
                change = current_price - data.get('pc', current_price)
                change_percent = (change / data.get('pc', current_price)) * 100 if data.get('pc') else 0

                price_data = {
                    'source': 'financial',
                    'source_id': f"{symbol}_finnhub_{datetime.utcnow().isoformat()}",
                    'content': f"Price: ${current_price:.2f}, Change: {change:+.2f} ({change_percent:+.2f}%)",
                    'author': 'finnhub',
                    'url': f"https://finnhub.io/quote/{symbol}",
                    'metadata': {
                        'symbol': symbol,
                        'timestamp': datetime.utcnow().isoformat(),
                        'current_price': current_price,
                        'high': data.get('h'),
                        'low': data.get('l'),
                        'open': data.get('o'),
                        'previous_close': data.get('pc'),
                        'change': change,
                        'change_percent': change_percent,
                        'data_type': 'price',
                        'provider': 'finnhub'
                    },
                    'collected_at': datetime.utcnow()
                }

                return [price_data]

        except Exception as e:
            self.logger.error(f"Error collecting Finnhub data for {symbol}: {e}")
            return []
