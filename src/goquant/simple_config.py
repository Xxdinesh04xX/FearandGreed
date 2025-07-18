"""
Simplified configuration that works with Python 3.13.
"""

import os
from typing import List, Optional
from dotenv import load_dotenv


class SimpleConfig:
    """Simplified configuration class that avoids pydantic-settings issues."""
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        # Load environment variables
        load_dotenv()
        
        # Database Configuration
        self.database_url = os.getenv('DATABASE_URL', 'sqlite:///goquant.db')
        
        # Logging Configuration
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.log_file = os.getenv('LOG_FILE', 'logs/goquant.log')
        
        # API Keys
        self.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        self.twitter_api_key = os.getenv('TWITTER_API_KEY')
        self.twitter_api_secret = os.getenv('TWITTER_API_SECRET')
        self.twitter_access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        self.twitter_access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.reddit_user_agent = os.getenv('REDDIT_USER_AGENT', 'GoQuant Sentiment Trader v1.0')
        
        self.news_api_key = os.getenv('NEWS_API_KEY')
        
        self.alpha_vantage_api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.iex_cloud_api_key = os.getenv('IEX_CLOUD_API_KEY')
        self.finnhub_api_key = os.getenv('FINNHUB_API_KEY')
        self.polygon_api_key = os.getenv('POLYGON_API_KEY')
        
        # Asset Configuration
        self.default_assets = self._parse_asset_list('DEFAULT_ASSETS', 'BTC,ETH,SPY,AAPL,TSLA,NVDA')
        self.crypto_assets = self._parse_asset_list('CRYPTO_ASSETS', 'BTC,ETH,ADA,SOL,MATIC')
        self.stock_assets = self._parse_asset_list('STOCK_ASSETS', 'SPY,QQQ,AAPL,MSFT,GOOGL,TSLA,NVDA,AMZN')
        self.forex_assets = self._parse_asset_list('FOREX_ASSETS', 'EURUSD,GBPUSD,USDJPY')
        
        # System Configuration
        self.environment = os.getenv('ENVIRONMENT', 'development')
        self.debug = os.getenv('DEBUG', 'false').lower() == 'true'
        
        # Data Collection Configuration
        self.data_collection_interval = int(os.getenv('DATA_COLLECTION_INTERVAL', '300'))  # 5 minutes
        self.max_data_age_hours = int(os.getenv('MAX_DATA_AGE_HOURS', '24'))
        
        # Sentiment Analysis Configuration
        self.sentiment_model = os.getenv('SENTIMENT_MODEL', 'distilbert-base-uncased-finetuned-sst-2-english')
        self.device = os.getenv('DEVICE', 'cpu')
        self.sentiment_analysis_batch_size = int(os.getenv('SENTIMENT_ANALYSIS_BATCH_SIZE', '32'))
        
        # Signal Generation Configuration
        self.signal_confidence_threshold = float(os.getenv('SIGNAL_CONFIDENCE_THRESHOLD', '0.7'))
        self.fear_greed_window_hours = int(os.getenv('FEAR_GREED_WINDOW_HOURS', '24'))
        self.sentiment_momentum_periods = int(os.getenv('SENTIMENT_MOMENTUM_PERIODS', '5'))
        
        # Dashboard Configuration
        self.dashboard_host = os.getenv('DASHBOARD_HOST', 'localhost')
        self.dashboard_port = int(os.getenv('DASHBOARD_PORT', '8050'))
        self.dashboard_debug = os.getenv('DASHBOARD_DEBUG', 'false').lower() == 'true'
        
        # Backtesting Configuration
        self.backtest_initial_capital = float(os.getenv('BACKTEST_INITIAL_CAPITAL', '100000'))
        self.backtest_commission_rate = float(os.getenv('BACKTEST_COMMISSION_RATE', '0.001'))
        
        # Rate Limiting Configuration
        self.twitter_rate_limit = int(os.getenv('TWITTER_RATE_LIMIT', '300'))
        self.reddit_rate_limit = int(os.getenv('REDDIT_RATE_LIMIT', '60'))
        self.news_rate_limit = int(os.getenv('NEWS_RATE_LIMIT', '100'))
        self.alpha_vantage_rate_limit = int(os.getenv('ALPHA_VANTAGE_RATE_LIMIT', '5'))
        self.finnhub_rate_limit = int(os.getenv('FINNHUB_RATE_LIMIT', '60'))
    
    def _parse_asset_list(self, env_var: str, default: str) -> List[str]:
        """Parse comma-separated asset list from environment variable."""
        asset_string = os.getenv(env_var, default)
        return [asset.strip() for asset in asset_string.split(',') if asset.strip()]
    
    def get_configured_apis(self) -> List[str]:
        """Get list of configured APIs."""
        apis = []
        if self.twitter_bearer_token:
            apis.append('Twitter')
        if self.reddit_client_id:
            apis.append('Reddit')
        if self.news_api_key:
            apis.append('NewsAPI')
        if self.alpha_vantage_api_key:
            apis.append('Alpha Vantage')
        if self.finnhub_api_key:
            apis.append('Finnhub')
        if self.polygon_api_key:
            apis.append('Polygon')
        return apis
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == 'production'


# Global config instance
_config = None


def get_simple_config() -> SimpleConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = SimpleConfig()
    return _config
