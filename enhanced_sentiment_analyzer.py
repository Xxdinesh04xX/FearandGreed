"""
Enhanced Sentiment Analyzer with Fear & Greed Index and Advanced Confidence Levels.
"""

import re
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import sqlite3


class FearGreedIndex:
    """Calculate Fear & Greed Index for market sentiment."""
    
    def __init__(self):
        # Fear indicators (negative sentiment)
        self.fear_keywords = [
            'crash', 'dump', 'panic', 'fear', 'sell', 'drop', 'fall', 'decline', 
            'bear', 'bearish', 'recession', 'correction', 'bubble', 'overvalued',
            'risk', 'danger', 'warning', 'alert', 'concern', 'worry', 'scared',
            'volatile', 'uncertainty', 'unstable', 'weak', 'loss', 'losing'
        ]
        
        # Greed indicators (positive sentiment)
        self.greed_keywords = [
            'moon', 'rocket', 'bull', 'bullish', 'pump', 'surge', 'rally', 
            'breakout', 'explosion', 'massive', 'huge', 'incredible', 'amazing',
            'buy', 'accumulate', 'hodl', 'diamond', 'hands', 'strong', 'momentum',
            'growth', 'profit', 'gains', 'winning', 'success', 'opportunity'
        ]
        
        # Extreme fear keywords (panic selling)
        self.extreme_fear_keywords = [
            'panic', 'catastrophe', 'disaster', 'collapse', 'crash', 'bloodbath',
            'massacre', 'apocalypse', 'dead', 'dying', 'worthless', 'scam'
        ]
        
        # Extreme greed keywords (FOMO)
        self.extreme_greed_keywords = [
            'fomo', 'yolo', 'lambo', 'millionaire', 'rich', 'wealthy', 'fortune',
            'jackpot', 'lottery', 'easy money', 'get rich', 'to the moon'
        ]
    
    def calculate_fear_greed_score(self, texts: List[str], volumes: List[int] = None) -> Dict[str, Any]:
        """
        Calculate Fear & Greed Index (0-100 scale).
        0-25: Extreme Fear
        25-45: Fear  
        45-55: Neutral
        55-75: Greed
        75-100: Extreme Greed
        """
        if not texts:
            return {
                'score': 50,
                'label': 'Neutral',
                'confidence': 0.0,
                'fear_signals': 0,
                'greed_signals': 0,
                'total_analyzed': 0
            }
        
        if volumes is None:
            volumes = [1] * len(texts)
        
        fear_score = 0
        greed_score = 0
        total_weight = 0
        fear_signals = 0
        greed_signals = 0
        
        for text, volume in zip(texts, volumes):
            text_lower = text.lower()
            weight = volume  # Weight by engagement/volume
            
            # Count fear indicators
            fear_count = 0
            for keyword in self.fear_keywords:
                fear_count += text_lower.count(keyword)
            
            # Count extreme fear (double weight)
            extreme_fear_count = 0
            for keyword in self.extreme_fear_keywords:
                extreme_fear_count += text_lower.count(keyword)
            
            # Count greed indicators
            greed_count = 0
            for keyword in self.greed_keywords:
                greed_count += text_lower.count(keyword)
            
            # Count extreme greed (double weight)
            extreme_greed_count = 0
            for keyword in self.extreme_greed_keywords:
                extreme_greed_count += text_lower.count(keyword)
            
            # Calculate weighted scores
            text_fear = (fear_count + extreme_fear_count * 2) * weight
            text_greed = (greed_count + extreme_greed_count * 2) * weight
            
            fear_score += text_fear
            greed_score += text_greed
            total_weight += weight
            
            if text_fear > 0:
                fear_signals += 1
            if text_greed > 0:
                greed_signals += 1
        
        # Normalize scores
        if total_weight > 0:
            fear_ratio = fear_score / total_weight
            greed_ratio = greed_score / total_weight
        else:
            fear_ratio = greed_ratio = 0
        
        # Calculate final Fear & Greed score (0-100)
        if fear_ratio + greed_ratio == 0:
            fg_score = 50  # Neutral
        else:
            # Convert to 0-100 scale where 0=extreme fear, 100=extreme greed
            greed_dominance = greed_ratio / (fear_ratio + greed_ratio)
            fg_score = greed_dominance * 100
        
        # Determine label
        if fg_score <= 25:
            label = "Extreme Fear"
        elif fg_score <= 45:
            label = "Fear"
        elif fg_score <= 55:
            label = "Neutral"
        elif fg_score <= 75:
            label = "Greed"
        else:
            label = "Extreme Greed"
        
        # Calculate confidence based on signal strength
        total_signals = fear_signals + greed_signals
        confidence = min(1.0, total_signals / max(len(texts) * 0.3, 1))
        
        return {
            'score': round(fg_score, 1),
            'label': label,
            'confidence': round(confidence, 3),
            'fear_signals': fear_signals,
            'greed_signals': greed_signals,
            'total_analyzed': len(texts),
            'fear_ratio': round(fear_ratio, 3),
            'greed_ratio': round(greed_ratio, 3)
        }


class AdvancedSentimentAnalyzer:
    """Advanced sentiment analyzer with confidence levels."""
    
    def __init__(self):
        self.fear_greed = FearGreedIndex()
        
        # Enhanced keyword sets
        self.positive_words = [
            'great', 'good', 'excellent', 'amazing', 'awesome', 'fantastic',
            'bull', 'bullish', 'up', 'rise', 'rising', 'moon', 'rocket',
            'buy', 'strong', 'momentum', 'growth', 'profit', 'gain', 'surge',
            'breakthrough', 'success', 'winning', 'opportunity', 'optimistic'
        ]
        
        self.negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'crash', 'fall',
            'bear', 'bearish', 'down', 'drop', 'sell', 'weak',
            'loss', 'decline', 'dump', 'fear', 'panic', 'concern',
            'risk', 'danger', 'warning', 'problem', 'issue', 'trouble'
        ]
        
        # Confidence modifiers
        self.high_confidence_indicators = [
            'definitely', 'certainly', 'absolutely', 'confirmed', 'official',
            'breaking', 'announced', 'reported', 'data shows', 'analysis',
            'research', 'study', 'expert', 'professional', 'institutional'
        ]
        
        self.low_confidence_indicators = [
            'maybe', 'perhaps', 'possibly', 'might', 'could', 'rumor',
            'speculation', 'unconfirmed', 'allegedly', 'supposedly',
            'opinion', 'think', 'believe', 'guess', 'hope'
        ]
    
    def analyze_text_advanced(self, text: str, metadata: Dict = None) -> Dict[str, Any]:
        """Advanced sentiment analysis with confidence scoring."""
        text_lower = text.lower()
        
        # Basic sentiment calculation
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return self._empty_result()
        
        # Calculate base sentiment score
        positive_ratio = positive_count / total_words
        negative_ratio = negative_count / total_words
        sentiment_score = (positive_ratio - negative_ratio) * 2
        sentiment_score = max(-1.0, min(1.0, sentiment_score))
        
        # Calculate confidence factors
        confidence_factors = self._calculate_confidence_factors(text, metadata)
        
        # Adjust sentiment based on confidence
        final_confidence = confidence_factors['overall_confidence']
        adjusted_sentiment = sentiment_score * final_confidence
        
        # Determine label
        if adjusted_sentiment > 0.2:
            label = 'positive'
        elif adjusted_sentiment < -0.2:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'score': round(adjusted_sentiment, 4),
            'confidence': round(final_confidence, 4),
            'label': label,
            'raw_sentiment': round(sentiment_score, 4),
            'positive_signals': positive_count,
            'negative_signals': negative_count,
            'confidence_factors': confidence_factors,
            'text_length': total_words
        }
    
    def _calculate_confidence_factors(self, text: str, metadata: Dict = None) -> Dict[str, float]:
        """Calculate various confidence factors."""
        text_lower = text.lower()
        
        # Factor 1: Text length (longer texts generally more reliable)
        text_length = len(text.split())
        length_factor = min(1.0, text_length / 20)  # Normalize to 20 words
        
        # Factor 2: High confidence indicators
        high_conf_count = sum(1 for indicator in self.high_confidence_indicators 
                             if indicator in text_lower)
        high_conf_factor = min(1.0, high_conf_count * 0.3)
        
        # Factor 3: Low confidence indicators (reduce confidence)
        low_conf_count = sum(1 for indicator in self.low_confidence_indicators 
                            if indicator in text_lower)
        low_conf_penalty = min(0.5, low_conf_count * 0.2)
        
        # Factor 4: Source credibility (if metadata available)
        source_factor = 1.0
        if metadata:
            if metadata.get('source') == 'news':
                source_factor = 1.2  # News sources more credible
            elif metadata.get('source') == 'twitter':
                # Check follower count, retweets, etc.
                retweets = metadata.get('retweet_count', 0)
                likes = metadata.get('like_count', 0)
                engagement = retweets + likes
                source_factor = min(1.5, 0.8 + engagement / 1000)
            elif metadata.get('source') == 'reddit':
                # Check upvotes, comments
                score = metadata.get('score', 0)
                comments = metadata.get('num_comments', 0)
                source_factor = min(1.3, 0.9 + (score + comments) / 100)
        
        # Factor 5: Financial symbols presence (more relevant)
        symbols = self._extract_symbols(text)
        symbol_factor = min(1.2, 1.0 + len(symbols) * 0.1)
        
        # Combine factors
        base_confidence = length_factor * symbol_factor * source_factor
        confidence_adjustment = high_conf_factor - low_conf_penalty
        overall_confidence = min(1.0, max(0.1, base_confidence + confidence_adjustment))
        
        return {
            'length_factor': round(length_factor, 3),
            'high_confidence_factor': round(high_conf_factor, 3),
            'low_confidence_penalty': round(low_conf_penalty, 3),
            'source_factor': round(source_factor, 3),
            'symbol_factor': round(symbol_factor, 3),
            'overall_confidence': round(overall_confidence, 3)
        }
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract financial symbols from text."""
        symbols = []
        
        # Find $SYMBOL patterns
        dollar_symbols = re.findall(r'\$([A-Z]{1,5})', text)
        symbols.extend(dollar_symbols)
        
        # Common symbols
        common_symbols = ['BTC', 'ETH', 'AAPL', 'TSLA', 'SPY', 'NVDA', 'MSFT', 'GOOGL']
        for symbol in common_symbols:
            if symbol.lower() in text.lower() and symbol not in symbols:
                symbols.append(symbol)
        
        return list(set(symbols))
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'score': 0.0,
            'confidence': 0.0,
            'label': 'neutral',
            'raw_sentiment': 0.0,
            'positive_signals': 0,
            'negative_signals': 0,
            'confidence_factors': {},
            'text_length': 0
        }


class EnhancedSignalGenerator:
    """Generate trading signals with Fear & Greed Index and confidence levels."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.sentiment_analyzer = AdvancedSentimentAnalyzer()
        self.fear_greed = FearGreedIndex()
    
    def generate_enhanced_signals(self, symbol: str, hours_back: int = 24) -> Dict[str, Any]:
        """Generate enhanced trading signals with Fear & Greed Index."""
        try:
            # Get recent data for symbol
            conn = sqlite3.connect(self.db_path)
            
            # Get sentiment data
            sentiment_query = """
                SELECT rd.content, sd.sentiment_score, sd.confidence, rd.collected_at, rd.source
                FROM sentiment_data sd
                JOIN raw_data rd ON sd.raw_data_id = rd.id
                WHERE sd.symbol = ? AND sd.processed_at > datetime('now', '-{} hours')
                ORDER BY sd.processed_at DESC
            """.format(hours_back)
            
            cursor = conn.cursor()
            cursor.execute(sentiment_query, (symbol,))
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return self._empty_signal(symbol)
            
            # Extract data
            texts = [row[0] for row in results]
            sentiments = [row[1] for row in results]
            confidences = [row[2] for row in results]
            sources = [row[4] for row in results]
            
            # Calculate Fear & Greed Index
            fg_index = self.fear_greed.calculate_fear_greed_score(texts)
            
            # Calculate weighted sentiment
            weighted_sentiment = sum(s * c for s, c in zip(sentiments, confidences))
            total_confidence = sum(confidences)
            
            if total_confidence > 0:
                avg_sentiment = weighted_sentiment / total_confidence
                avg_confidence = total_confidence / len(confidences)
            else:
                avg_sentiment = 0
                avg_confidence = 0
            
            # Generate signal based on multiple factors
            signal_result = self._generate_signal_logic(
                avg_sentiment, avg_confidence, fg_index, len(results)
            )
            
            # Add symbol and metadata
            signal_result.update({
                'symbol': symbol,
                'fear_greed_index': fg_index,
                'data_points': len(results),
                'source_breakdown': self._count_sources(sources),
                'analysis_period_hours': hours_back,
                'generated_at': datetime.now().isoformat()
            })
            
            return signal_result
            
        except Exception as e:
            print(f"Error generating enhanced signal for {symbol}: {e}")
            return self._empty_signal(symbol)
    
    def _generate_signal_logic(self, sentiment: float, confidence: float, 
                              fg_index: Dict, data_points: int) -> Dict[str, Any]:
        """Advanced signal generation logic."""
        
        # Base signal from sentiment
        if sentiment > 0.3 and confidence > 0.6:
            base_signal = "BUY"
            base_strength = "MODERATE"
        elif sentiment < -0.3 and confidence > 0.6:
            base_signal = "SELL"
            base_strength = "MODERATE"
        else:
            base_signal = "HOLD"
            base_strength = "WEAK"
        
        # Adjust based on Fear & Greed Index
        fg_score = fg_index['score']
        
        # Fear & Greed adjustments
        if fg_score <= 25:  # Extreme Fear - potential buying opportunity
            if base_signal == "SELL":
                base_signal = "HOLD"  # Don't sell in extreme fear
            elif base_signal == "HOLD" and sentiment > -0.1:
                base_signal = "BUY"  # Contrarian buy signal
                base_strength = "MODERATE"
        
        elif fg_score >= 75:  # Extreme Greed - potential selling opportunity
            if base_signal == "BUY":
                base_signal = "HOLD"  # Don't buy in extreme greed
            elif base_signal == "HOLD" and sentiment < 0.1:
                base_signal = "SELL"  # Contrarian sell signal
                base_strength = "MODERATE"
        
        # Strengthen signals based on data quality
        if data_points >= 20 and confidence > 0.8:
            if base_strength == "MODERATE":
                base_strength = "STRONG"
            elif base_strength == "WEAK":
                base_strength = "MODERATE"
        
        # Calculate overall confidence
        fg_confidence = fg_index['confidence']
        data_confidence = min(1.0, data_points / 10)  # More data = more confidence
        
        overall_confidence = (confidence + fg_confidence + data_confidence) / 3
        
        # Risk assessment
        risk_level = self._assess_risk(sentiment, fg_score, confidence, data_points)
        
        return {
            'signal_type': base_signal,
            'strength': base_strength,
            'confidence': round(overall_confidence, 4),
            'sentiment_score': round(sentiment, 4),
            'risk_level': risk_level,
            'reasoning': self._generate_reasoning(base_signal, sentiment, fg_index, confidence)
        }
    
    def _assess_risk(self, sentiment: float, fg_score: float, 
                    confidence: float, data_points: int) -> str:
        """Assess risk level of the signal."""
        risk_factors = 0
        
        # High volatility indicators
        if abs(sentiment) > 0.7:
            risk_factors += 1
        
        # Extreme Fear & Greed
        if fg_score <= 20 or fg_score >= 80:
            risk_factors += 1
        
        # Low confidence
        if confidence < 0.5:
            risk_factors += 1
        
        # Limited data
        if data_points < 10:
            risk_factors += 1
        
        if risk_factors >= 3:
            return "HIGH"
        elif risk_factors >= 2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_reasoning(self, signal: str, sentiment: float, 
                           fg_index: Dict, confidence: float) -> str:
        """Generate human-readable reasoning for the signal."""
        fg_label = fg_index['label']
        fg_score = fg_index['score']
        
        reasoning = f"Signal: {signal} | "
        reasoning += f"Sentiment: {sentiment:.2f} | "
        reasoning += f"Fear & Greed: {fg_label} ({fg_score:.1f}) | "
        reasoning += f"Confidence: {confidence:.2f}"
        
        return reasoning
    
    def _count_sources(self, sources: List[str]) -> Dict[str, int]:
        """Count data points by source."""
        source_count = {}
        for source in sources:
            source_count[source] = source_count.get(source, 0) + 1
        return source_count
    
    def _empty_signal(self, symbol: str) -> Dict[str, Any]:
        """Return empty signal structure."""
        return {
            'symbol': symbol,
            'signal_type': 'HOLD',
            'strength': 'WEAK',
            'confidence': 0.0,
            'sentiment_score': 0.0,
            'risk_level': 'HIGH',
            'reasoning': 'Insufficient data',
            'fear_greed_index': {
                'score': 50,
                'label': 'Neutral',
                'confidence': 0.0
            },
            'data_points': 0,
            'source_breakdown': {},
            'analysis_period_hours': 24,
            'generated_at': datetime.now().isoformat()
        }


# Test the enhanced system
if __name__ == "__main__":
    # Test Fear & Greed Index
    fg = FearGreedIndex()
    
    test_texts = [
        "Bitcoin is crashing! Panic selling everywhere! This is a disaster!",
        "HODL! Diamond hands! Bitcoin to the moon! Easy money!",
        "Market looks stable today, normal trading volume."
    ]
    
    result = fg.calculate_fear_greed_score(test_texts)
    print("Fear & Greed Index Test:")
    print(f"Score: {result['score']} ({result['label']})")
    print(f"Confidence: {result['confidence']}")
    print()
    
    # Test Advanced Sentiment
    analyzer = AdvancedSentimentAnalyzer()
    
    test_text = "Breaking: Apple announces record earnings! Definitely bullish for $AAPL stock."
    metadata = {'source': 'news', 'retweet_count': 150, 'like_count': 300}
    
    sentiment_result = analyzer.analyze_text_advanced(test_text, metadata)
    print("Advanced Sentiment Test:")
    print(f"Score: {sentiment_result['score']}")
    print(f"Confidence: {sentiment_result['confidence']}")
    print(f"Label: {sentiment_result['label']}")
    print(f"Factors: {sentiment_result['confidence_factors']}")
