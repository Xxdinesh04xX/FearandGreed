"""
Simplified GoQuant Sentiment Trader that works with Python 3.13.
"""

import asyncio
import sqlite3
import time
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class SimpleConfig:
    """Simple configuration class."""
    def __init__(self):
        self.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.alpha_vantage_api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.finnhub_api_key = os.getenv('FINNHUB_API_KEY')
        
        # Assets to track
        assets_str = os.getenv('DEFAULT_ASSETS', 'BTC,ETH,SPY,AAPL,TSLA,NVDA')
        self.default_assets = [asset.strip() for asset in assets_str.split(',')]
        
        # Database
        self.database_path = 'simple_goquant.db'


class SimpleDatabase:
    """Simple database manager."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Raw data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS raw_data (
                id INTEGER PRIMARY KEY,
                source TEXT,
                content TEXT,
                symbols TEXT,
                author TEXT,
                url TEXT,
                collected_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Sentiment data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_data (
                id INTEGER PRIMARY KEY,
                raw_data_id INTEGER,
                symbol TEXT,
                sentiment_score REAL,
                confidence REAL,
                processed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (raw_data_id) REFERENCES raw_data (id)
            )
        ''')
        
        # Trading signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_signals (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                signal_type TEXT,
                strength TEXT,
                confidence REAL,
                sentiment_score REAL,
                price_at_signal REAL,
                generated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Database initialized")
    
    def insert_raw_data(self, source: str, content: str, symbols: List[str], author: str = "", url: str = ""):
        """Insert raw data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        symbols_str = ','.join(symbols)
        cursor.execute('''
            INSERT INTO raw_data (source, content, symbols, author, url)
            VALUES (?, ?, ?, ?, ?)
        ''', (source, content, symbols_str, author, url))
        
        raw_data_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return raw_data_id
    
    def insert_sentiment_data(self, raw_data_id: int, symbol: str, sentiment_score: float, confidence: float):
        """Insert sentiment data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sentiment_data (raw_data_id, symbol, sentiment_score, confidence)
            VALUES (?, ?, ?, ?)
        ''', (raw_data_id, symbol, sentiment_score, confidence))
        
        conn.commit()
        conn.close()
    
    def insert_trading_signal(self, symbol: str, signal_type: str, strength: str, 
                            confidence: float, sentiment_score: float, price: float = None):
        """Insert trading signal."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trading_signals (symbol, signal_type, strength, confidence, sentiment_score, price_at_signal)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (symbol, signal_type, strength, confidence, sentiment_score, price))
        
        conn.commit()
        conn.close()
    
    def get_recent_sentiment(self, symbol: str, hours: int = 24) -> List[Dict]:
        """Get recent sentiment data for a symbol."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        cursor.execute('''
            SELECT sentiment_score, confidence, processed_at
            FROM sentiment_data
            WHERE symbol = ? AND processed_at > ?
            ORDER BY processed_at DESC
        ''', (symbol, cutoff_time))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'sentiment_score': row[0],
                'confidence': row[1],
                'processed_at': row[2]
            })
        
        conn.close()
        return results


class SimpleSentimentAnalyzer:
    """Simple keyword-based sentiment analyzer."""
    
    def __init__(self):
        self.positive_words = [
            'great', 'good', 'excellent', 'amazing', 'awesome', 'fantastic',
            'bull', 'bullish', 'up', 'rise', 'rising', 'moon', 'rocket',
            'buy', 'strong', 'momentum', 'growth', 'profit', 'gain'
        ]
        
        self.negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'crash', 'fall',
            'bear', 'bearish', 'down', 'drop', 'sell', 'weak',
            'loss', 'decline', 'dump', 'fear', 'panic'
        ]
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text."""
        text_lower = text.lower()
        
        # Count positive and negative words
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        # Calculate sentiment score
        total_words = len(text.split())
        if total_words == 0:
            return {'score': 0.0, 'confidence': 0.0, 'label': 'neutral'}
        
        # Normalize by text length
        positive_ratio = positive_count / total_words
        negative_ratio = negative_count / total_words
        
        # Calculate final score (-1 to 1)
        sentiment_score = (positive_ratio - negative_ratio) * 2
        sentiment_score = max(-1.0, min(1.0, sentiment_score))
        
        # Calculate confidence based on word count
        confidence = min(1.0, (positive_count + negative_count) / 5)
        
        # Determine label
        if sentiment_score > 0.1:
            label = 'positive'
        elif sentiment_score < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'score': sentiment_score,
            'confidence': confidence,
            'label': label
        }
    
    def extract_symbols(self, text: str) -> List[str]:
        """Extract financial symbols from text."""
        # Find $SYMBOL patterns
        symbols = re.findall(r'\$([A-Z]{1,5})', text)
        
        # Also look for common symbols without $
        common_symbols = ['BTC', 'ETH', 'AAPL', 'TSLA', 'SPY', 'NVDA', 'MSFT', 'GOOGL']
        for symbol in common_symbols:
            if symbol.lower() in text.lower() and symbol not in symbols:
                symbols.append(symbol)
        
        return list(set(symbols))  # Remove duplicates


class SimpleDataCollector:
    """Simple data collector using Yahoo Finance."""
    
    def __init__(self, config: SimpleConfig):
        self.config = config
    
    async def collect_financial_data(self) -> List[Dict[str, Any]]:
        """Collect financial data from Yahoo Finance."""
        data = []
        
        try:
            import yfinance as yf
            
            for symbol in self.config.default_assets:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    if info and 'regularMarketPrice' in info:
                        price = info['regularMarketPrice']
                        change = info.get('regularMarketChange', 0)
                        change_percent = info.get('regularMarketChangePercent', 0)
                        
                        content = f"${symbol} Price: ${price:.2f}, Change: {change:+.2f} ({change_percent:+.2f}%)"
                        
                        data.append({
                            'source': 'yahoo_finance',
                            'content': content,
                            'symbols': [symbol],
                            'author': 'yahoo_finance',
                            'url': f'https://finance.yahoo.com/quote/{symbol}',
                            'price': price
                        })
                        
                        print(f"📈 {symbol}: ${price:.2f} ({change_percent:+.2f}%)")
                        
                except Exception as e:
                    print(f"❌ Error collecting data for {symbol}: {e}")
                    
                # Small delay to be respectful
                await asyncio.sleep(0.1)
                
        except ImportError:
            print("❌ yfinance not available")
        
        return data


class SimpleSignalGenerator:
    """Simple trading signal generator."""
    
    def __init__(self, db: SimpleDatabase):
        self.db = db
    
    def generate_signals(self) -> List[Dict[str, Any]]:
        """Generate trading signals based on recent sentiment."""
        signals = []
        
        # Get configured assets
        load_dotenv()
        assets_str = os.getenv('DEFAULT_ASSETS', 'BTC,ETH,SPY,AAPL,TSLA,NVDA')
        assets = [asset.strip() for asset in assets_str.split(',')]
        
        for symbol in assets:
            try:
                # Get recent sentiment data
                recent_sentiment = self.db.get_recent_sentiment(symbol, hours=24)
                
                if len(recent_sentiment) < 3:
                    continue  # Need at least 3 data points
                
                # Calculate average sentiment
                avg_sentiment = sum(s['sentiment_score'] for s in recent_sentiment) / len(recent_sentiment)
                avg_confidence = sum(s['confidence'] for s in recent_sentiment) / len(recent_sentiment)
                
                # Generate signal based on sentiment
                if avg_sentiment > 0.3 and avg_confidence > 0.5:
                    signal_type = "BUY"
                    strength = "STRONG" if avg_sentiment > 0.6 else "MODERATE"
                elif avg_sentiment < -0.3 and avg_confidence > 0.5:
                    signal_type = "SELL"
                    strength = "STRONG" if avg_sentiment < -0.6 else "MODERATE"
                else:
                    signal_type = "HOLD"
                    strength = "WEAK"
                
                if signal_type != "HOLD":
                    signals.append({
                        'symbol': symbol,
                        'signal_type': signal_type,
                        'strength': strength,
                        'confidence': avg_confidence,
                        'sentiment_score': avg_sentiment,
                        'data_points': len(recent_sentiment)
                    })
                    
                    print(f"🎯 {signal_type} signal for {symbol} (strength: {strength}, confidence: {avg_confidence:.2f})")
                
            except Exception as e:
                print(f"❌ Error generating signal for {symbol}: {e}")
        
        return signals


class SimpleTrader:
    """Main simple trading system."""
    
    def __init__(self):
        self.config = SimpleConfig()
        self.db = SimpleDatabase(self.config.database_path)
        self.sentiment_analyzer = SimpleSentimentAnalyzer()
        self.data_collector = SimpleDataCollector(self.config)
        self.signal_generator = SimpleSignalGenerator(self.db)
        
        print("🚀 Simple GoQuant Trader initialized")
        print(f"📊 Tracking assets: {self.config.default_assets}")
        print(f"🔑 APIs configured: {len([x for x in [self.config.twitter_bearer_token, self.config.reddit_client_id, self.config.news_api_key, self.config.alpha_vantage_api_key, self.config.finnhub_api_key] if x])}/5")
    
    async def run_cycle(self):
        """Run one complete trading cycle."""
        print(f"\n⏰ Running trading cycle at {datetime.now().strftime('%H:%M:%S')}")
        
        # 1. Collect financial data
        print("📈 Collecting financial data...")
        financial_data = await self.data_collector.collect_financial_data()
        
        # 2. Process and store data
        for data in financial_data:
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
        
        print(f"✅ Cycle complete: {len(financial_data)} data points, {len(signals)} signals")
        return len(signals)
    
    async def run_continuous(self, interval_minutes: int = 5):
        """Run the trader continuously."""
        print(f"🔄 Starting continuous trading (every {interval_minutes} minutes)")
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
    """Main function."""
    print("🚀 Simple GoQuant Sentiment Trader")
    print("=" * 40)
    
    trader = SimpleTrader()
    
    # Run one cycle first
    await trader.run_cycle()
    
    # Ask user if they want to run continuously
    print("\n" + "=" * 40)
    response = input("Run continuously? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        await trader.run_continuous()
    else:
        print("✅ Single cycle completed. Check simple_goquant.db for results!")


if __name__ == "__main__":
    asyncio.run(main())
