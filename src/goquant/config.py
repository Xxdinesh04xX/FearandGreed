"""
Configuration management for GoQuant Sentiment Trader.
"""

import os
from typing import List, Optional
try:
    from pydantic import BaseSettings, Field, validator
except ImportError:
    from pydantic_settings import BaseSettings
    from pydantic import Field, validator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config(BaseSettings):
    """Application configuration using Pydantic BaseSettings."""
    
    # Twitter API Configuration
    twitter_bearer_token: Optional[str] = Field(None, env="TWITTER_BEARER_TOKEN")
    twitter_api_key: Optional[str] = Field(None, env="TWITTER_API_KEY")
    twitter_api_secret: Optional[str] = Field(None, env="TWITTER_API_SECRET")
    twitter_access_token: Optional[str] = Field(None, env="TWITTER_ACCESS_TOKEN")
    twitter_access_token_secret: Optional[str] = Field(None, env="TWITTER_ACCESS_TOKEN_SECRET")
    
    # Reddit API Configuration
    reddit_client_id: Optional[str] = Field(None, env="REDDIT_CLIENT_ID")
    reddit_client_secret: Optional[str] = Field(None, env="REDDIT_CLIENT_SECRET")
    reddit_user_agent: str = Field("GoQuant-SentimentTrader/1.0", env="REDDIT_USER_AGENT")
    
    # News API Configuration
    news_api_key: Optional[str] = Field(None, env="NEWS_API_KEY")
    
    # Financial Data APIs
    alpha_vantage_api_key: Optional[str] = Field(None, env="ALPHA_VANTAGE_API_KEY")
    iex_cloud_api_key: Optional[str] = Field(None, env="IEX_CLOUD_API_KEY")
    finnhub_api_key: Optional[str] = Field(None, env="FINNHUB_API_KEY")
    polygon_api_key: Optional[str] = Field(None, env="POLYGON_API_KEY")
    
    # Database Configuration
    database_url: str = Field("sqlite:///goquant.db", env="DATABASE_URL")
    redis_url: Optional[str] = Field(None, env="REDIS_URL")
    
    # Application Settings
    log_level: str = Field("INFO", env="LOG_LEVEL")
    debug: bool = Field(False, env="DEBUG")
    environment: str = Field("development", env="ENVIRONMENT")
    
    # Data Collection Settings
    data_collection_interval: int = Field(60, env="DATA_COLLECTION_INTERVAL")
    sentiment_analysis_batch_size: int = Field(100, env="SENTIMENT_ANALYSIS_BATCH_SIZE")
    max_concurrent_requests: int = Field(10, env="MAX_CONCURRENT_REQUESTS")
    
    # Trading Signal Settings
    signal_confidence_threshold: float = Field(0.7, env="SIGNAL_CONFIDENCE_THRESHOLD")
    fear_greed_window_hours: int = Field(24, env="FEAR_GREED_WINDOW_HOURS")
    sentiment_momentum_periods: int = Field(5, env="SENTIMENT_MOMENTUM_PERIODS")
    
    # Rate Limiting
    twitter_rate_limit_requests: int = Field(300, env="TWITTER_RATE_LIMIT_REQUESTS")
    twitter_rate_limit_window: int = Field(900, env="TWITTER_RATE_LIMIT_WINDOW")
    reddit_rate_limit_requests: int = Field(60, env="REDDIT_RATE_LIMIT_REQUESTS")
    reddit_rate_limit_window: int = Field(60, env="REDDIT_RATE_LIMIT_WINDOW")
    news_rate_limit_requests: int = Field(1000, env="NEWS_RATE_LIMIT_REQUESTS")
    news_rate_limit_window: int = Field(86400, env="NEWS_RATE_LIMIT_WINDOW")
    
    # Dashboard Settings
    dashboard_host: str = Field("localhost", env="DASHBOARD_HOST")
    dashboard_port: int = Field(8050, env="DASHBOARD_PORT")
    dashboard_debug: bool = Field(True, env="DASHBOARD_DEBUG")
    
    # Backtesting Settings
    backtest_start_date: str = Field("2023-01-01", env="BACKTEST_START_DATE")
    backtest_end_date: str = Field("2024-01-01", env="BACKTEST_END_DATE")
    backtest_initial_capital: float = Field(10000.0, env="BACKTEST_INITIAL_CAPITAL")
    
    # Asset Configuration (will be parsed from environment)
    default_assets: List[str] = Field(default=["BTC", "ETH", "SPY", "AAPL", "TSLA", "NVDA"])
    crypto_assets: List[str] = Field(default=["BTC", "ETH", "ADA", "SOL", "MATIC"])
    stock_assets: List[str] = Field(default=["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN"])
    forex_assets: List[str] = Field(default=["EURUSD", "GBPUSD", "USDJPY"])

    def __init__(self, **kwargs):
        """Initialize config and parse comma-separated asset lists from environment."""
        super().__init__(**kwargs)

        # Parse asset lists from environment variables
        import os

        if os.getenv('DEFAULT_ASSETS'):
            self.default_assets = [asset.strip() for asset in os.getenv('DEFAULT_ASSETS').split(',') if asset.strip()]

        if os.getenv('CRYPTO_ASSETS'):
            self.crypto_assets = [asset.strip() for asset in os.getenv('CRYPTO_ASSETS').split(',') if asset.strip()]

        if os.getenv('STOCK_ASSETS'):
            self.stock_assets = [asset.strip() for asset in os.getenv('STOCK_ASSETS').split(',') if asset.strip()]

        if os.getenv('FOREX_ASSETS'):
            self.forex_assets = [asset.strip() for asset in os.getenv('FOREX_ASSETS').split(',') if asset.strip()]
    
    # Model Configuration
    sentiment_model: str = Field("ProsusAI/finbert", env="SENTIMENT_MODEL")
    embedding_model: str = Field("sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    device: str = Field("cpu", env="DEVICE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str) -> any:
            if field_name.endswith('_assets'):
                return [asset.strip() for asset in raw_val.split(',')]
            return cls.json_loads(raw_val)


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def reload_config() -> Config:
    """Reload configuration from environment variables."""
    global config
    config = Config()
    return config
