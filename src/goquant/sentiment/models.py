"""
Sentiment analysis models for financial text.
"""

import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

from ..utils.logger import get_logger


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    score: float  # -1 to 1 (negative to positive)
    confidence: float  # 0 to 1
    label: str  # 'positive', 'negative', 'neutral'
    probabilities: Dict[str, float]  # Raw model probabilities
    emotions: Optional[Dict[str, float]] = None  # Fear, greed, etc.


class FinBERTSentimentModel:
    """
    FinBERT-based sentiment analysis model optimized for financial text.
    
    Uses pre-trained FinBERT model for financial sentiment classification.
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert", device: str = "cpu"):
        """
        Initialize the FinBERT sentiment model.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self.logger = get_logger(__name__)
        
        self.tokenizer = None
        self.model = None
        self._initialized = False
        
        # Label mappings for FinBERT
        self.label_mapping = {
            0: 'positive',
            1: 'negative', 
            2: 'neutral'
        }
        
        # Score mapping (convert to -1 to 1 scale)
        self.score_mapping = {
            'positive': 1.0,
            'negative': -1.0,
            'neutral': 0.0
        }
    
    async def initialize(self) -> None:
        """Initialize the model and tokenizer."""
        try:
            self.logger.info(f"Loading FinBERT model: {self.model_name}")

            # Check if CUDA is available and requested
            if self.device == "cuda" and not torch.cuda.is_available():
                self.logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = "cpu"

            # Load tokenizer
            self.logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir="./models/cache"
            )

            # Load model
            self.logger.info("Loading model...")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                cache_dir="./models/cache",
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16
            )

            self.model.to(self.device)
            self.model.eval()

            # Test the model with a simple input
            test_result = self.analyze_sentiment("The market is performing well today.")
            self.logger.info(f"Model test successful: {test_result.label} (confidence: {test_result.confidence:.3f})")

            self._initialized = True
            self.logger.info(f"FinBERT model loaded successfully on {self.device}")

        except Exception as e:
            self.logger.error(f"Failed to load FinBERT model: {e}")
            # Try fallback to a simpler model
            try:
                self.logger.info("Attempting fallback to distilbert-base-uncased-finetuned-sst-2-english")
                self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"

                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()

                # Update label mapping for DistilBERT
                self.label_mapping = {0: 'negative', 1: 'positive'}

                self._initialized = True
                self.logger.info("Fallback model loaded successfully")

            except Exception as fallback_error:
                self.logger.error(f"Fallback model also failed: {fallback_error}")
                raise
    
    def analyze_sentiment(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            SentimentResult with score, confidence, and probabilities
        """
        if not self._initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        if not text or not text.strip():
            return SentimentResult(
                score=0.0,
                confidence=0.0,
                label='neutral',
                probabilities={'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
            )
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
            
            # Convert to numpy for easier handling
            probs = probabilities.cpu().numpy()[0]
            
            # Get predicted label and confidence
            predicted_class = np.argmax(probs)
            predicted_label = self.label_mapping[predicted_class]
            confidence = float(probs[predicted_class])
            
            # Calculate sentiment score (-1 to 1)
            score = self._calculate_sentiment_score(probs)
            
            # Create probability dictionary
            prob_dict = {
                'positive': float(probs[0]),
                'negative': float(probs[1]),
                'neutral': float(probs[2])
            }
            
            # Calculate emotions (fear/greed indicators)
            emotions = self._calculate_emotions(probs, text)
            
            return SentimentResult(
                score=score,
                confidence=confidence,
                label=predicted_label,
                probabilities=prob_dict,
                emotions=emotions
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            # Return neutral sentiment on error
            return SentimentResult(
                score=0.0,
                confidence=0.0,
                label='neutral',
                probabilities={'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
            )
    
    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of SentimentResult objects
        """
        if not self._initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        if not texts:
            return []
        
        try:
            # Filter out empty texts
            valid_texts = [text for text in texts if text and text.strip()]
            if not valid_texts:
                return [SentimentResult(
                    score=0.0, confidence=0.0, label='neutral',
                    probabilities={'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
                ) for _ in texts]
            
            # Tokenize all texts
            inputs = self.tokenizer(
                valid_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
            
            # Convert to numpy
            probs_batch = probabilities.cpu().numpy()
            
            # Process results
            results = []
            for i, probs in enumerate(probs_batch):
                predicted_class = np.argmax(probs)
                predicted_label = self.label_mapping[predicted_class]
                confidence = float(probs[predicted_class])
                score = self._calculate_sentiment_score(probs)
                
                prob_dict = {
                    'positive': float(probs[0]),
                    'negative': float(probs[1]),
                    'neutral': float(probs[2])
                }
                
                emotions = self._calculate_emotions(probs, valid_texts[i])
                
                results.append(SentimentResult(
                    score=score,
                    confidence=confidence,
                    label=predicted_label,
                    probabilities=prob_dict,
                    emotions=emotions
                ))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing batch sentiment: {e}")
            # Return neutral sentiments on error
            return [SentimentResult(
                score=0.0, confidence=0.0, label='neutral',
                probabilities={'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
            ) for _ in texts]
    
    def _calculate_sentiment_score(self, probabilities: np.ndarray) -> float:
        """
        Calculate sentiment score from probabilities.
        
        Args:
            probabilities: Array of [positive, negative, neutral] probabilities
            
        Returns:
            Sentiment score from -1 to 1
        """
        positive_prob = probabilities[0]
        negative_prob = probabilities[1]
        neutral_prob = probabilities[2]
        
        # Calculate weighted score
        score = positive_prob * 1.0 + negative_prob * (-1.0) + neutral_prob * 0.0
        
        # Ensure score is in [-1, 1] range
        return max(-1.0, min(1.0, score))
    
    def _calculate_emotions(self, probabilities: np.ndarray, text: str) -> Dict[str, float]:
        """
        Calculate emotion indicators from sentiment probabilities.
        
        Args:
            probabilities: Array of sentiment probabilities
            text: Original text for additional context
            
        Returns:
            Dictionary of emotion scores
        """
        positive_prob = probabilities[0]
        negative_prob = probabilities[1]
        neutral_prob = probabilities[2]
        
        # Simple emotion mapping
        # In a more sophisticated implementation, you might use additional models
        emotions = {
            'fear': negative_prob * 0.8,  # Fear correlates with negative sentiment
            'greed': positive_prob * 0.7,  # Greed correlates with positive sentiment
            'neutral': neutral_prob,
            'uncertainty': neutral_prob * 0.6 + (1 - max(positive_prob, negative_prob)) * 0.4
        }
        
        # Adjust based on text content (simple keyword matching)
        text_lower = text.lower()
        if any(word in text_lower for word in ['crash', 'dump', 'bear', 'sell', 'panic']):
            emotions['fear'] = min(1.0, emotions['fear'] * 1.2)
        
        if any(word in text_lower for word in ['moon', 'bull', 'buy', 'pump', 'rocket']):
            emotions['greed'] = min(1.0, emotions['greed'] * 1.2)
        
        return emotions
