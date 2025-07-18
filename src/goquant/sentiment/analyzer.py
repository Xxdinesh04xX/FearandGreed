"""
Main sentiment analyzer that processes raw data and generates sentiment scores.
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from ..config import Config
from ..utils.logger import get_logger
from ..utils.text_processor import get_text_processor
from ..database.manager import DatabaseManager
from ..database.models import RawData, SentimentData
from .models import FinBERTSentimentModel, SentimentResult


class SentimentAnalyzer:
    """
    Main sentiment analyzer that processes raw data and generates sentiment insights.
    
    Handles text preprocessing, sentiment analysis using FinBERT, and aggregation
    of sentiment scores across different data sources and time periods.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the sentiment analyzer.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.text_processor = get_text_processor()
        
        # Initialize sentiment model
        self.sentiment_model = FinBERTSentimentModel(
            model_name=config.sentiment_model,
            device=config.device
        )
        
        # Database manager will be injected
        self.db_manager: Optional[DatabaseManager] = None
        
        self._initialized = False
        self.processed_count = 0
        self.error_count = 0
    
    async def initialize(self) -> None:
        """Initialize the sentiment analyzer."""
        try:
            # Initialize the sentiment model
            await self.sentiment_model.initialize()
            
            self._initialized = True
            self.logger.info("SentimentAnalyzer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SentimentAnalyzer: {e}")
            raise
    
    def set_database_manager(self, db_manager: DatabaseManager) -> None:
        """Set the database manager for data operations."""
        self.db_manager = db_manager
    
    async def process_pending_data(self) -> int:
        """
        Process all pending raw data for sentiment analysis.
        
        Returns:
            Number of records processed
        """
        if not self._initialized:
            await self.initialize()
        
        if not self.db_manager:
            self.logger.error("Database manager not set")
            return 0
        
        try:
            # Get unprocessed raw data
            raw_data_records = await self.db_manager.get_unprocessed_raw_data(
                limit=self.config.sentiment_analysis_batch_size
            )
            
            if not raw_data_records:
                return 0
            
            self.logger.info(f"Processing {len(raw_data_records)} raw data records")
            
            # Process in batches
            batch_size = 32  # Optimal batch size for BERT models
            processed_count = 0
            
            for i in range(0, len(raw_data_records), batch_size):
                batch = raw_data_records[i:i + batch_size]
                batch_results = await self._process_batch(batch)
                
                if batch_results:
                    # Store sentiment results
                    await self.db_manager.insert_sentiment_data(batch_results)
                    
                    # Mark raw data as processed
                    processed_ids = [record.id for record in batch]
                    await self.db_manager.mark_raw_data_processed(processed_ids)
                    
                    processed_count += len(batch)
            
            self.processed_count += processed_count
            self.logger.info(f"Successfully processed {processed_count} records")
            
            return processed_count
            
        except Exception as e:
            self.logger.error(f"Error processing pending data: {e}")
            self.error_count += 1
            return 0
    
    async def _process_batch(self, raw_data_batch: List[RawData]) -> List[Dict[str, Any]]:
        """
        Process a batch of raw data records.
        
        Args:
            raw_data_batch: List of RawData records
            
        Returns:
            List of sentiment data dictionaries
        """
        try:
            # Preprocess texts
            texts = []
            valid_records = []
            
            for record in raw_data_batch:
                if record.content and record.content.strip():
                    # Preprocess text for sentiment analysis
                    processed_text = self.text_processor.preprocess_for_sentiment(record.content)
                    
                    if processed_text:
                        texts.append(processed_text)
                        valid_records.append(record)
            
            if not texts:
                return []
            
            # Analyze sentiment for all texts
            sentiment_results = self.sentiment_model.analyze_batch(texts)
            
            # Create sentiment data records
            sentiment_data_list = []
            
            for record, processed_text, sentiment_result in zip(valid_records, texts, sentiment_results):
                # Extract financial symbols
                symbols = self.text_processor.extract_financial_symbols(record.content)
                primary_symbol = symbols[0] if symbols else None
                
                sentiment_data = {
                    'raw_data_id': record.id,
                    'symbol': primary_symbol,
                    'sentiment_score': sentiment_result.score,
                    'confidence': sentiment_result.confidence,
                    'emotions': sentiment_result.emotions,
                    'entities': {
                        'symbols': symbols,
                        'hashtags': self.text_processor.extract_hashtags(record.content),
                        'mentions': self.text_processor.extract_mentions(record.content)
                    },
                    'processed_text': processed_text,
                    'model_version': self.sentiment_model.model_name,
                    'processed_at': datetime.utcnow()
                }
                
                sentiment_data_list.append(sentiment_data)
            
            return sentiment_data_list
            
        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")
            return []
    
    async def analyze_text(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            SentimentResult
        """
        if not self._initialized:
            await self.initialize()
        
        # Preprocess text
        processed_text = self.text_processor.preprocess_for_sentiment(text)
        
        # Analyze sentiment
        return self.sentiment_model.analyze_sentiment(processed_text)
    
    async def get_symbol_sentiment_summary(
        self, 
        symbol: str, 
        hours: int = 24
    ) -> Optional[Dict[str, Any]]:
        """
        Get aggregated sentiment summary for a symbol.
        
        Args:
            symbol: Financial symbol
            hours: Number of hours to look back
            
        Returns:
            Sentiment summary dictionary
        """
        if not self.db_manager:
            self.logger.error("Database manager not set")
            return None
        
        try:
            # Get recent sentiment data for the symbol
            sentiment_records = await self.db_manager.get_recent_sentiment(
                symbol=symbol,
                hours=hours,
                limit=1000
            )
            
            if not sentiment_records:
                return None
            
            # Calculate aggregated metrics
            scores = [record.sentiment_score for record in sentiment_records]
            confidences = [record.confidence for record in sentiment_records]
            
            # Weighted average (weight by confidence)
            total_weight = sum(confidences)
            if total_weight > 0:
                weighted_score = sum(score * conf for score, conf in zip(scores, confidences)) / total_weight
            else:
                weighted_score = sum(scores) / len(scores)
            
            # Calculate sentiment distribution
            positive_count = sum(1 for score in scores if score > 0.1)
            negative_count = sum(1 for score in scores if score < -0.1)
            neutral_count = len(scores) - positive_count - negative_count
            
            # Calculate momentum (trend over time)
            momentum = self._calculate_sentiment_momentum(sentiment_records)
            
            # Aggregate emotions
            emotions = self._aggregate_emotions(sentiment_records)
            
            summary = {
                'symbol': symbol,
                'time_period_hours': hours,
                'data_points': len(sentiment_records),
                'weighted_sentiment_score': weighted_score,
                'average_confidence': sum(confidences) / len(confidences),
                'sentiment_distribution': {
                    'positive': positive_count / len(scores),
                    'negative': negative_count / len(scores),
                    'neutral': neutral_count / len(scores)
                },
                'momentum': momentum,
                'emotions': emotions,
                'last_updated': datetime.utcnow()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting sentiment summary for {symbol}: {e}")
            return None
    
    def _calculate_sentiment_momentum(self, sentiment_records: List[SentimentData]) -> float:
        """
        Calculate sentiment momentum (trend direction).
        
        Args:
            sentiment_records: List of sentiment records (ordered by time)
            
        Returns:
            Momentum score (-1 to 1)
        """
        if len(sentiment_records) < 2:
            return 0.0
        
        # Sort by processed_at time
        sorted_records = sorted(sentiment_records, key=lambda x: x.processed_at)
        
        # Split into two halves and compare average sentiment
        mid_point = len(sorted_records) // 2
        first_half = sorted_records[:mid_point]
        second_half = sorted_records[mid_point:]
        
        if not first_half or not second_half:
            return 0.0
        
        first_avg = sum(record.sentiment_score for record in first_half) / len(first_half)
        second_avg = sum(record.sentiment_score for record in second_half) / len(second_half)
        
        # Calculate momentum as the difference
        momentum = second_avg - first_avg
        
        # Normalize to [-1, 1] range
        return max(-1.0, min(1.0, momentum))
    
    def _aggregate_emotions(self, sentiment_records: List[SentimentData]) -> Dict[str, float]:
        """
        Aggregate emotion scores across multiple records.
        
        Args:
            sentiment_records: List of sentiment records
            
        Returns:
            Aggregated emotion scores
        """
        if not sentiment_records:
            return {}
        
        # Collect all emotion data
        all_emotions = {}
        valid_records = 0
        
        for record in sentiment_records:
            if record.emotions:
                valid_records += 1
                for emotion, score in record.emotions.items():
                    if emotion not in all_emotions:
                        all_emotions[emotion] = []
                    all_emotions[emotion].append(score)
        
        if not all_emotions:
            return {}
        
        # Calculate averages
        aggregated = {}
        for emotion, scores in all_emotions.items():
            aggregated[emotion] = sum(scores) / len(scores)
        
        return aggregated
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'initialized': self._initialized,
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'model_name': self.sentiment_model.model_name,
            'device': self.sentiment_model.device
        }
