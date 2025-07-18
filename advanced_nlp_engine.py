"""
Advanced NLP Engine with Transformer Models and Deep Learning Integration.
Implements FinBERT, sarcasm detection, and multi-language analysis.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, BertTokenizer, BertForSequenceClassification
)
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import asyncio
import aiohttp
import time
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import gc
from functools import lru_cache
import warnings
warnings.filterwarnings("ignore")


@dataclass
class SentimentResult:
    """Structured sentiment analysis result."""
    score: float
    confidence: float
    label: str
    model_name: str
    processing_time: float
    features: Dict[str, Any]
    metadata: Dict[str, Any]


class FinBERTAnalyzer:
    """Financial BERT model for domain-specific sentiment analysis."""
    
    def __init__(self, model_name: str = "ProsusAI/finbert", device: str = "auto"):
        """Initialize FinBERT model."""
        self.device = self._get_device(device)
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._load_model()
        
        # Performance tracking
        self.inference_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        
        logging.info(f"FinBERT initialized on {self.device}")
    
    def _get_device(self, device: str) -> str:
        """Determine optimal device for inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
            else:
                return "cpu"
        return device
    
    def _load_model(self):
        """Load FinBERT model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
            
        except Exception as e:
            logging.error(f"Failed to load FinBERT model: {e}")
            # Fallback to basic model
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load fallback model if FinBERT fails."""
        try:
            self.pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
            self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            logging.info("Loaded fallback sentiment model")
        except Exception as e:
            logging.error(f"Failed to load fallback model: {e}")
            raise
    
    @lru_cache(maxsize=1000)
    def _cached_inference(self, text: str) -> Tuple[float, float, str]:
        """Cached inference for repeated texts."""
        self.cache_hits += 1
        return self._raw_inference(text)
    
    def _raw_inference(self, text: str) -> Tuple[float, float, str]:
        """Raw model inference without caching."""
        start_time = time.time()
        
        try:
            # Truncate text to model's max length
            max_length = self.tokenizer.model_max_length
            if len(text) > max_length:
                text = text[:max_length-10] + "..."
            
            # Get predictions
            results = self.pipeline(text)
            
            # Process results based on model type
            if isinstance(results[0], list):
                # Multiple scores returned
                scores = {r['label'].lower(): r['score'] for r in results[0]}
                
                # Map labels to sentiment
                if 'positive' in scores:
                    positive_score = scores.get('positive', 0)
                    negative_score = scores.get('negative', 0)
                    neutral_score = scores.get('neutral', 0)
                elif 'bullish' in scores:
                    positive_score = scores.get('bullish', 0)
                    negative_score = scores.get('bearish', 0)
                    neutral_score = scores.get('neutral', 0)
                else:
                    # Generic mapping
                    positive_score = max([s for l, s in scores.items() if 'pos' in l.lower()], default=0)
                    negative_score = max([s for l, s in scores.items() if 'neg' in l.lower()], default=0)
                    neutral_score = max([s for l, s in scores.items() if 'neu' in l.lower()], default=0)
                
                # Calculate final sentiment score (-1 to 1)
                sentiment_score = positive_score - negative_score
                confidence = max(positive_score, negative_score, neutral_score)
                
                # Determine label
                if positive_score > negative_score and positive_score > neutral_score:
                    label = "positive"
                elif negative_score > positive_score and negative_score > neutral_score:
                    label = "negative"
                else:
                    label = "neutral"
            
            else:
                # Single result
                result = results[0]
                label = result['label'].lower()
                confidence = result['score']
                
                # Map to sentiment score
                if 'pos' in label or 'bull' in label:
                    sentiment_score = confidence
                elif 'neg' in label or 'bear' in label:
                    sentiment_score = -confidence
                else:
                    sentiment_score = 0
            
            processing_time = time.time() - start_time
            self.inference_times.append(processing_time)
            
            return sentiment_score, confidence, label
            
        except Exception as e:
            logging.error(f"FinBERT inference error: {e}")
            return 0.0, 0.0, "neutral"
    
    def analyze(self, text: str, use_cache: bool = True) -> SentimentResult:
        """Analyze sentiment using FinBERT."""
        if not text or not text.strip():
            return SentimentResult(0.0, 0.0, "neutral", self.model_name, 0.0, {}, {})
        
        text = text.strip()
        
        if use_cache:
            sentiment_score, confidence, label = self._cached_inference(text)
        else:
            self.cache_misses += 1
            sentiment_score, confidence, label = self._raw_inference(text)
        
        # Extract additional features
        features = self._extract_features(text)
        
        # Metadata
        metadata = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'device': self.device,
            'cache_hit': use_cache and text in self._cached_inference.cache_info().currsize > 0
        }
        
        processing_time = self.inference_times[-1] if self.inference_times else 0.0
        
        return SentimentResult(
            score=sentiment_score,
            confidence=confidence,
            label=label,
            model_name=self.model_name,
            processing_time=processing_time,
            features=features,
            metadata=metadata
        )
    
    def _extract_features(self, text: str) -> Dict[str, Any]:
        """Extract additional features from text."""
        features = {}
        
        # Financial keywords
        financial_keywords = [
            'earnings', 'revenue', 'profit', 'loss', 'growth', 'decline',
            'bullish', 'bearish', 'buy', 'sell', 'hold', 'target', 'price',
            'upgrade', 'downgrade', 'analyst', 'forecast', 'guidance'
        ]
        
        text_lower = text.lower()
        features['financial_keyword_count'] = sum(1 for kw in financial_keywords if kw in text_lower)
        features['financial_keyword_density'] = features['financial_keyword_count'] / max(len(text.split()), 1)
        
        # Urgency indicators
        urgency_words = ['breaking', 'urgent', 'alert', 'now', 'immediately', 'asap']
        features['urgency_score'] = sum(1 for word in urgency_words if word in text_lower)
        
        # Certainty indicators
        certainty_words = ['definitely', 'certainly', 'confirmed', 'official', 'announced']
        uncertainty_words = ['maybe', 'possibly', 'rumor', 'speculation', 'unconfirmed']
        features['certainty_score'] = sum(1 for word in certainty_words if word in text_lower)
        features['uncertainty_score'] = sum(1 for word in uncertainty_words if word in text_lower)
        
        # Emotional intensity
        intense_words = ['amazing', 'terrible', 'incredible', 'disaster', 'fantastic', 'awful']
        features['emotional_intensity'] = sum(1 for word in intense_words if word in text_lower)
        
        return features
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        cache_info = self._cached_inference.cache_info()
        
        return {
            'model_name': self.model_name,
            'device': self.device,
            'total_inferences': len(self.inference_times),
            'avg_inference_time': np.mean(self.inference_times) if self.inference_times else 0,
            'cache_hits': cache_info.hits,
            'cache_misses': cache_info.misses,
            'cache_hit_rate': cache_info.hits / (cache_info.hits + cache_info.misses) if (cache_info.hits + cache_info.misses) > 0 else 0,
            'cache_size': cache_info.currsize,
            'max_cache_size': cache_info.maxsize
        }


class SarcasmDetector:
    """Detect sarcasm and irony in text."""
    
    def __init__(self):
        """Initialize sarcasm detection model."""
        try:
            self.pipeline = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-irony",
                return_all_scores=True
            )
            self.available = True
        except Exception as e:
            logging.warning(f"Sarcasm detector not available: {e}")
            self.available = False
    
    def detect(self, text: str) -> Dict[str, float]:
        """Detect sarcasm/irony in text."""
        if not self.available or not text:
            return {'sarcasm_score': 0.0, 'confidence': 0.0}
        
        try:
            results = self.pipeline(text)
            
            # Extract sarcasm score
            scores = {r['label'].lower(): r['score'] for r in results[0]}
            
            sarcasm_score = scores.get('irony', scores.get('sarcasm', 0.0))
            confidence = max(scores.values())
            
            return {
                'sarcasm_score': sarcasm_score,
                'confidence': confidence,
                'is_sarcastic': sarcasm_score > 0.5
            }
            
        except Exception as e:
            logging.error(f"Sarcasm detection error: {e}")
            return {'sarcasm_score': 0.0, 'confidence': 0.0}


class MultiLanguageAnalyzer:
    """Multi-language sentiment analysis."""
    
    def __init__(self):
        """Initialize multi-language models."""
        self.models = {}
        self.supported_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'zh', 'ja']
        
        # Load language detection
        try:
            from langdetect import detect
            self.detect_language = detect
            self.lang_detection_available = True
        except ImportError:
            logging.warning("Language detection not available. Install langdetect.")
            self.lang_detection_available = False
    
    def _get_model_for_language(self, lang: str):
        """Get or load model for specific language."""
        if lang in self.models:
            return self.models[lang]
        
        try:
            if lang == 'en':
                model = pipeline("sentiment-analysis", 
                               model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            else:
                # Use multilingual model for other languages
                model = pipeline("sentiment-analysis",
                               model="cardiffnlp/twitter-xlm-roberta-base-sentiment")
            
            self.models[lang] = model
            return model
            
        except Exception as e:
            logging.error(f"Failed to load model for language {lang}: {e}")
            return None
    
    def analyze(self, text: str, language: str = None) -> Dict[str, Any]:
        """Analyze sentiment in multiple languages."""
        if not text:
            return {'score': 0.0, 'confidence': 0.0, 'language': 'unknown'}
        
        # Detect language if not provided
        if language is None and self.lang_detection_available:
            try:
                language = self.detect_language(text)
            except:
                language = 'en'  # Default to English
        elif language is None:
            language = 'en'
        
        # Get appropriate model
        model = self._get_model_for_language(language)
        if model is None:
            return {'score': 0.0, 'confidence': 0.0, 'language': language}
        
        try:
            results = model(text)
            
            if isinstance(results, list) and len(results) > 0:
                result = results[0]
                label = result['label'].lower()
                confidence = result['score']
                
                # Map to sentiment score
                if 'pos' in label:
                    score = confidence
                elif 'neg' in label:
                    score = -confidence
                else:
                    score = 0.0
                
                return {
                    'score': score,
                    'confidence': confidence,
                    'label': label,
                    'language': language
                }
            
        except Exception as e:
            logging.error(f"Multi-language analysis error: {e}")
        
        return {'score': 0.0, 'confidence': 0.0, 'language': language}


class AdvancedNLPEngine:
    """Advanced NLP engine combining multiple models and techniques."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize advanced NLP engine."""
        self.config = config or {}
        
        # Initialize components
        self.finbert = FinBERTAnalyzer()
        self.sarcasm_detector = SarcasmDetector()
        self.multilang_analyzer = MultiLanguageAnalyzer()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Performance tracking
        self.total_analyses = 0
        self.total_processing_time = 0.0
        
        logging.info("Advanced NLP Engine initialized")
    
    async def analyze_comprehensive(self, text: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Comprehensive analysis using all available models."""
        if not text:
            return self._empty_result()
        
        start_time = time.time()
        metadata = metadata or {}
        
        # Run analyses in parallel
        loop = asyncio.get_event_loop()
        
        # Primary sentiment analysis
        finbert_task = loop.run_in_executor(self.executor, self.finbert.analyze, text)
        
        # Sarcasm detection
        sarcasm_task = loop.run_in_executor(self.executor, self.sarcasm_detector.detect, text)
        
        # Multi-language analysis
        multilang_task = loop.run_in_executor(self.executor, self.multilang_analyzer.analyze, text)
        
        # Wait for all analyses
        finbert_result, sarcasm_result, multilang_result = await asyncio.gather(
            finbert_task, sarcasm_task, multilang_task
        )
        
        # Combine results
        combined_result = self._combine_results(
            finbert_result, sarcasm_result, multilang_result, metadata
        )
        
        # Update performance tracking
        processing_time = time.time() - start_time
        self.total_analyses += 1
        self.total_processing_time += processing_time
        
        combined_result['processing_time'] = processing_time
        combined_result['engine_stats'] = self.get_performance_stats()
        
        return combined_result
    
    def _combine_results(self, finbert_result: SentimentResult, sarcasm_result: Dict,
                        multilang_result: Dict, metadata: Dict) -> Dict[str, Any]:
        """Combine results from multiple models."""
        
        # Base sentiment from FinBERT
        base_sentiment = finbert_result.score
        base_confidence = finbert_result.confidence
        
        # Adjust for sarcasm
        if sarcasm_result.get('is_sarcastic', False):
            # Flip sentiment if sarcastic
            adjusted_sentiment = -base_sentiment * 0.8  # Reduce magnitude slightly
            sarcasm_adjustment = True
        else:
            adjusted_sentiment = base_sentiment
            sarcasm_adjustment = False
        
        # Weight by multi-language confidence
        multilang_confidence = multilang_result.get('confidence', 0.5)
        final_confidence = (base_confidence + multilang_confidence) / 2
        
        # Ensemble scoring
        ensemble_score = self._calculate_ensemble_score(
            finbert_result, multilang_result, sarcasm_result
        )
        
        return {
            'sentiment_score': adjusted_sentiment,
            'confidence': final_confidence,
            'ensemble_score': ensemble_score,
            'label': self._determine_label(adjusted_sentiment),
            'sarcasm_detected': sarcasm_adjustment,
            'language': multilang_result.get('language', 'en'),
            'models_used': {
                'finbert': finbert_result.model_name,
                'sarcasm': 'cardiffnlp/twitter-roberta-base-irony' if sarcasm_result else None,
                'multilang': 'cardiffnlp/twitter-xlm-roberta-base-sentiment'
            },
            'features': finbert_result.features,
            'detailed_results': {
                'finbert': finbert_result.__dict__,
                'sarcasm': sarcasm_result,
                'multilang': multilang_result
            }
        }
    
    def _calculate_ensemble_score(self, finbert_result: SentimentResult,
                                 multilang_result: Dict, sarcasm_result: Dict) -> float:
        """Calculate ensemble score from multiple models."""
        scores = []
        weights = []
        
        # FinBERT score (highest weight for financial text)
        scores.append(finbert_result.score)
        weights.append(0.6)
        
        # Multi-language score
        if multilang_result.get('score') is not None:
            scores.append(multilang_result['score'])
            weights.append(0.3)
        
        # Sarcasm adjustment
        if sarcasm_result.get('is_sarcastic', False):
            # Add negative weight for sarcasm
            scores.append(-abs(finbert_result.score) * 0.5)
            weights.append(0.1)
        
        # Weighted average
        if scores and weights:
            ensemble_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        else:
            ensemble_score = 0.0
        
        return ensemble_score
    
    def _determine_label(self, score: float) -> str:
        """Determine sentiment label from score."""
        if score > 0.2:
            return "positive"
        elif score < -0.2:
            return "negative"
        else:
            return "neutral"
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'sentiment_score': 0.0,
            'confidence': 0.0,
            'ensemble_score': 0.0,
            'label': 'neutral',
            'sarcasm_detected': False,
            'language': 'unknown',
            'models_used': {},
            'features': {},
            'detailed_results': {},
            'processing_time': 0.0
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        finbert_stats = self.finbert.get_performance_stats()
        
        return {
            'total_analyses': self.total_analyses,
            'avg_processing_time': self.total_processing_time / max(self.total_analyses, 1),
            'finbert_stats': finbert_stats,
            'thread_pool_size': self.executor._max_workers,
            'models_loaded': {
                'finbert': True,
                'sarcasm': self.sarcasm_detector.available,
                'multilang': len(self.multilang_analyzer.models)
            }
        }
    
    def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=True)
        
        # Clear model caches
        if hasattr(self.finbert, '_cached_inference'):
            self.finbert._cached_inference.cache_clear()
        
        # Force garbage collection
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Test the advanced NLP engine
if __name__ == "__main__":
    async def test_advanced_nlp():
        """Test the advanced NLP engine."""
        engine = AdvancedNLPEngine()
        
        test_texts = [
            "Apple's earnings beat expectations! $AAPL to the moon! 🚀",
            "Oh great, another market crash. Just what we needed... 🙄",
            "Bitcoin price stable today, normal trading volume.",
            "Tesla announces record deliveries! Bullish news for $TSLA investors."
        ]
        
        print("🧠 Testing Advanced NLP Engine")
        print("=" * 50)
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n📝 Test {i}: {text}")
            
            result = await engine.analyze_comprehensive(text)
            
            print(f"   Sentiment: {result['sentiment_score']:.3f}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Label: {result['label']}")
            print(f"   Sarcasm: {result['sarcasm_detected']}")
            print(f"   Language: {result['language']}")
            print(f"   Processing Time: {result['processing_time']:.3f}s")
        
        # Performance stats
        stats = engine.get_performance_stats()
        print(f"\n📊 Performance Statistics:")
        print(f"   Total Analyses: {stats['total_analyses']}")
        print(f"   Avg Processing Time: {stats['avg_processing_time']:.3f}s")
        print(f"   FinBERT Cache Hit Rate: {stats['finbert_stats']['cache_hit_rate']:.2%}")
        
        engine.cleanup()
    
    # Run test
    asyncio.run(test_advanced_nlp())
