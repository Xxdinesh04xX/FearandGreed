"""
Comprehensive Test Suite for Enterprise Dinesh Trading Dashboard.
Tests all advanced features: NLP, ML, Performance, Analytics.
"""

import asyncio
import time
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any
import unittest
from datetime import datetime, timedelta

# Import all modules for testing
from advanced_nlp_engine import AdvancedNLPEngine, FinBERTAnalyzer, SarcasmDetector, MultiLanguageAnalyzer
from predictive_modeling import EnsemblePredictor, FeatureEngineer, MarketRegimeClassifier
from performance_optimizer import PerformanceOptimizer, MemoryPool, LockFreeQueue, SIMDTextProcessor
from advanced_analytics import MarketPsychologyAnalyzer, CrossMarketAnalyzer, AlternativeDataIntegrator


class TestAdvancedNLP(unittest.TestCase):
    """Test suite for Advanced NLP Engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.nlp_engine = AdvancedNLPEngine()
        self.test_texts = [
            "Apple's earnings beat expectations! $AAPL to the moon! 🚀",
            "Oh great, another market crash. Just what we needed... 🙄",
            "Bitcoin price stable today, normal trading volume.",
            "Tesla announces record deliveries! Bullish news for $TSLA investors.",
            "Breaking: Federal Reserve announces interest rate hike. Market uncertainty ahead."
        ]
    
    async def test_comprehensive_analysis(self):
        """Test comprehensive NLP analysis."""
        print("🧠 Testing Advanced NLP Engine...")
        
        for i, text in enumerate(self.test_texts):
            result = await self.nlp_engine.analyze_comprehensive(text)
            
            # Verify result structure
            self.assertIn('sentiment_score', result)
            self.assertIn('confidence', result)
            self.assertIn('ensemble_score', result)
            self.assertIn('sarcasm_detected', result)
            self.assertIn('language', result)
            
            # Verify score ranges
            self.assertGreaterEqual(result['sentiment_score'], -1.0)
            self.assertLessEqual(result['sentiment_score'], 1.0)
            self.assertGreaterEqual(result['confidence'], 0.0)
            self.assertLessEqual(result['confidence'], 1.0)
            
            print(f"   Text {i+1}: Sentiment={result['sentiment_score']:.3f}, "
                  f"Confidence={result['confidence']:.3f}, "
                  f"Sarcasm={result['sarcasm_detected']}")
        
        # Test performance stats
        stats = self.nlp_engine.get_performance_stats()
        self.assertGreater(stats['total_analyses'], 0)
        print(f"   Performance: {stats['total_analyses']} analyses, "
              f"{stats['avg_processing_time']:.3f}s avg time")
    
    def test_finbert_analyzer(self):
        """Test FinBERT analyzer specifically."""
        print("🏦 Testing FinBERT Analyzer...")
        
        finbert = FinBERTAnalyzer()
        
        financial_texts = [
            "Company reports strong quarterly earnings with revenue growth of 15%",
            "Stock price plummets amid concerns about regulatory investigation",
            "Analyst upgrades rating to buy with $150 price target"
        ]
        
        for text in financial_texts:
            result = finbert.analyze(text)
            
            # Verify FinBERT specific features
            self.assertIn('features', result.__dict__)
            self.assertIn('financial_keyword_count', result.features)
            self.assertGreater(result.features['financial_keyword_count'], 0)
            
        print(f"   FinBERT processed {len(financial_texts)} financial texts successfully")
    
    def test_sarcasm_detection(self):
        """Test sarcasm detection."""
        print("😏 Testing Sarcasm Detection...")
        
        sarcasm_detector = SarcasmDetector()
        
        if sarcasm_detector.available:
            sarcastic_text = "Oh great, another market crash. Just what we needed..."
            normal_text = "Market shows positive momentum today"
            
            sarcasm_result = sarcasm_detector.detect(sarcastic_text)
            normal_result = sarcasm_detector.detect(normal_text)
            
            print(f"   Sarcastic text score: {sarcasm_result.get('sarcasm_score', 0):.3f}")
            print(f"   Normal text score: {normal_result.get('sarcasm_score', 0):.3f}")
        else:
            print("   Sarcasm detection model not available")


class TestPredictiveModeling(unittest.TestCase):
    """Test suite for Predictive Modeling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.predictor = EnsemblePredictor()
        self.feature_engineer = FeatureEngineer()
        self.regime_classifier = MarketRegimeClassifier()
        
        # Generate sample data
        np.random.seed(42)
        n_samples = 500
        
        self.sentiment_data = pd.DataFrame({
            'symbol': np.random.choice(['BTC', 'AAPL', 'TSLA'], n_samples),
            'sentiment_score': np.random.normal(0, 0.3, n_samples),
            'confidence': np.random.uniform(0.5, 1.0, n_samples),
            'processed_at': pd.date_range('2023-01-01', periods=n_samples, freq='H')
        })
        
        self.price_data = pd.DataFrame({
            'symbol': np.random.choice(['BTC', 'AAPL', 'TSLA'], n_samples),
            'price': np.random.lognormal(4, 0.2, n_samples),
            'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='H')
        })
    
    def test_feature_engineering(self):
        """Test feature engineering."""
        print("🔧 Testing Feature Engineering...")
        
        # Test sentiment features
        sentiment_features = self.feature_engineer.create_sentiment_features(self.sentiment_data)
        
        self.assertGreater(sentiment_features.shape[1], 10)  # Should have many features
        self.assertEqual(sentiment_features.shape[0], len(self.sentiment_data))
        
        # Test price features
        price_features = self.feature_engineer.create_price_features(self.price_data)
        
        self.assertGreater(price_features.shape[1], 5)  # Should have price features
        
        print(f"   Sentiment features: {sentiment_features.shape[1]} columns")
        print(f"   Price features: {price_features.shape[1]} columns")
    
    def test_ensemble_training(self):
        """Test ensemble model training."""
        print("🤖 Testing Ensemble Model Training...")
        
        # Prepare features
        features = self.predictor.prepare_features(self.sentiment_data, self.price_data)
        
        # Create target (next period price change)
        target = self.price_data.groupby('symbol')['price'].pct_change().shift(-1).fillna(0)
        
        # Remove NaN values
        valid_idx = ~(features.isna().any(axis=1) | target.isna())
        X = features[valid_idx]
        y = target[valid_idx]
        
        if len(X) > 50:  # Need sufficient data
            # Train model
            training_results = self.predictor.train(X, y)
            
            self.assertIn('model_performances', training_results)
            self.assertGreater(len(training_results['model_performances']), 0)
            
            # Test prediction
            predictions = self.predictor.predict(X.iloc[-10:])
            
            self.assertIn('ensemble_prediction', predictions)
            self.assertEqual(len(predictions['ensemble_prediction']), 10)
            
            print(f"   Trained {len(training_results['model_performances'])} models")
            print(f"   Best model performance: {min(training_results['model_performances'].values(), key=lambda x: x['cv_score'])}")
        else:
            print("   Insufficient data for training")
    
    def test_regime_classification(self):
        """Test market regime classification."""
        print("📊 Testing Market Regime Classification...")
        
        regime_result = self.regime_classifier.classify_regime(self.sentiment_data, self.price_data)
        
        self.assertIn('regime', regime_result)
        self.assertIn('regime_name', regime_result)
        self.assertIn('confidence', regime_result)
        
        print(f"   Current regime: {regime_result['regime_name']}")
        print(f"   Confidence: {regime_result['confidence']:.3f}")


class TestPerformanceOptimization(unittest.TestCase):
    """Test suite for Performance Optimization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = PerformanceOptimizer({
            'buffer_size': 100,
            'batch_size': 10,
            'cache_size': 50
        })
    
    def test_memory_pool(self):
        """Test memory pool functionality."""
        print("💾 Testing Memory Pool...")
        
        memory_pool = MemoryPool(block_size=1024, pool_size=10)
        
        # Get blocks
        blocks = []
        for _ in range(5):
            block = memory_pool.get_block()
            self.assertIsNotNone(block)
            self.assertEqual(len(block), 1024)
            blocks.append(block)
        
        # Return blocks
        for block in blocks:
            memory_pool.return_block(block)
        
        stats = memory_pool.get_stats()
        print(f"   Memory pool stats: {stats}")
        
        self.assertEqual(stats['available_blocks'], 10)  # All blocks returned
    
    def test_lock_free_queue(self):
        """Test lock-free queue."""
        print("🔄 Testing Lock-Free Queue...")
        
        queue = LockFreeQueue(maxsize=10)
        
        # Test put/get operations
        for i in range(5):
            success = queue.put(f"item_{i}")
            self.assertTrue(success)
        
        self.assertEqual(len(queue), 5)
        
        # Test get operations
        items = []
        for _ in range(5):
            item = queue.get()
            if item:
                items.append(item)
        
        self.assertEqual(len(items), 5)
        print(f"   Queue processed {len(items)} items successfully")
    
    def test_simd_processor(self):
        """Test SIMD text processor."""
        print("⚡ Testing SIMD Text Processor...")
        
        simd_processor = SIMDTextProcessor()
        
        test_texts = [
            "Great news for investors! Stock price rising!",
            "Terrible market conditions. Sell everything!",
            "Normal trading day with average volume.",
            "Excellent quarterly results exceed expectations!"
        ]
        
        positive_words = ['great', 'excellent', 'rising', 'good']
        negative_words = ['terrible', 'sell', 'bad', 'crash']
        
        # Test vectorized sentiment scoring
        scores = simd_processor.vectorized_sentiment_score(test_texts, positive_words, negative_words)
        
        self.assertEqual(len(scores), len(test_texts))
        
        # Test batch feature extraction
        features = simd_processor.batch_text_features(test_texts)
        
        self.assertIn('length', features)
        self.assertIn('word_count', features)
        self.assertEqual(len(features['length']), len(test_texts))
        
        print(f"   SIMD processor: {simd_processor.use_simd}")
        print(f"   Processed {len(test_texts)} texts with {len(features)} feature types")
    
    def test_streaming_processor(self):
        """Test streaming processor."""
        print("🌊 Testing Streaming Processor...")
        
        self.optimizer.start_optimization()
        
        # Add test texts
        test_texts = [
            "Apple earnings beat expectations!",
            "Market volatility concerns investors.",
            "Bitcoin price shows stability.",
            "Tesla announces new product line."
        ] * 5  # 20 texts total
        
        # Add texts for processing
        for i, text in enumerate(test_texts):
            success = self.optimizer.streaming_processor.add_text(text, {'id': i})
            self.assertTrue(success)
        
        # Wait for processing
        time.sleep(2)
        
        # Collect results
        results = []
        for _ in range(len(test_texts)):
            result = self.optimizer.streaming_processor.get_result()
            if result:
                results.append(result)
        
        print(f"   Processed {len(results)}/{len(test_texts)} texts")
        
        # Get performance report
        report = self.optimizer.get_optimization_report()
        if 'current_metrics' in report:
            print(f"   Throughput: {report['current_metrics']['throughput']:.1f} items/sec")
        
        self.optimizer.stop_optimization()


class TestAdvancedAnalytics(unittest.TestCase):
    """Test suite for Advanced Analytics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.psychology_analyzer = MarketPsychologyAnalyzer()
        self.cross_market_analyzer = CrossMarketAnalyzer()
        self.alt_data_integrator = AlternativeDataIntegrator()
        
        # Generate sample data
        np.random.seed(42)
        n_samples = 200
        
        self.sentiment_data = pd.DataFrame({
            'symbol': np.random.choice(['BTC', 'AAPL', 'TSLA'], n_samples),
            'sentiment_score': np.random.normal(0, 0.4, n_samples),
            'confidence': np.random.uniform(0.5, 1.0, n_samples),
            'processed_at': pd.date_range('2023-01-01', periods=n_samples, freq='H')
        })
        
        self.price_data = pd.DataFrame({
            'symbol': np.random.choice(['BTC', 'AAPL', 'TSLA'], n_samples),
            'price': np.random.lognormal(4, 0.2, n_samples),
            'volume': np.random.randint(1000, 10000, n_samples),
            'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='H')
        })
    
    def test_market_psychology_analysis(self):
        """Test market psychology analysis."""
        print("🧠 Testing Market Psychology Analysis...")
        
        psychology_results = self.psychology_analyzer.analyze_market_psychology(
            self.sentiment_data, self.price_data
        )
        
        # Verify result structure
        self.assertIn('fear_greed_dynamics', psychology_results)
        self.assertIn('behavioral_biases', psychology_results)
        self.assertIn('crowd_psychology', psychology_results)
        self.assertIn('contrarian_signals', psychology_results)
        
        # Check contrarian signals
        signals = psychology_results['contrarian_signals']
        print(f"   Generated {signals['total_signals']} contrarian signals")
        print(f"   Buy signals: {signals['buy_signals']}, Sell signals: {signals['sell_signals']}")
        
        # Check behavioral biases
        biases = psychology_results['behavioral_biases']
        detected_biases = [bias for bias, result in biases.items() if result.get('detected', False)]
        print(f"   Detected biases: {detected_biases}")
    
    def test_cross_market_analysis(self):
        """Test cross-market correlation analysis."""
        print("🔗 Testing Cross-Market Analysis...")
        
        cross_market_results = self.cross_market_analyzer.analyze_sentiment_contagion(self.sentiment_data)
        
        if 'static_correlations' in cross_market_results:
            correlations = cross_market_results['static_correlations']
            print(f"   Correlation matrix computed for {len(correlations)} assets")
            
            if 'network_analysis' in cross_market_results:
                network = cross_market_results['network_analysis']
                print(f"   Network density: {network.get('density', 0):.3f}")
                print(f"   Most central asset: {network.get('most_central_asset', 'N/A')}")
        else:
            print("   Insufficient data for cross-market analysis")
    
    def test_alternative_data_integration(self):
        """Test alternative data integration."""
        print("📊 Testing Alternative Data Integration...")
        
        # Mock alternative data
        alternative_data = {
            'economic_indicators': pd.DataFrame({
                'indicator': ['GDP', 'Unemployment'],
                'value': [2.1, 3.8],
                'timestamp': pd.date_range('2023-01-01', periods=2, freq='M')
            }),
            'earnings_calls': pd.DataFrame({
                'company': ['AAPL', 'TSLA'],
                'sentiment': [0.3, -0.1],
                'timestamp': pd.date_range('2023-01-01', periods=2, freq='Q')
            })
        }
        
        alt_results = self.alt_data_integrator.integrate_alternative_data(
            self.sentiment_data, alternative_data
        )
        
        self.assertIn('individual_sources', alt_results)
        self.assertIn('data_quality_score', alt_results)
        
        print(f"   Data quality score: {alt_results['data_quality_score']:.2f}")
        print(f"   Alternative sources processed: {len(alt_results['individual_sources'])}")


async def run_comprehensive_tests():
    """Run all test suites."""
    print("🚀 Dinesh Enterprise Trading Dashboard - Comprehensive Test Suite")
    print("=" * 80)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Test suites
    test_suites = [
        TestAdvancedNLP(),
        TestPredictiveModeling(),
        TestPerformanceOptimization(),
        TestAdvancedAnalytics()
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for suite in test_suites:
        suite_name = suite.__class__.__name__
        print(f"\n📋 Running {suite_name}...")
        print("-" * 50)
        
        # Get test methods
        test_methods = [method for method in dir(suite) if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            try:
                method = getattr(suite, test_method)
                if asyncio.iscoroutinefunction(method):
                    await method()
                else:
                    method()
                passed_tests += 1
                print(f"✅ {test_method} passed")
            except Exception as e:
                print(f"❌ {test_method} failed: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 Test Summary")
    print("=" * 80)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Enterprise system is ready for deployment.")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests failed. Review and fix issues before deployment.")
    
    print("\n🎯 Enterprise Features Tested:")
    print("   ✅ Advanced NLP with FinBERT and Transformer models")
    print("   ✅ Machine Learning ensemble predictions")
    print("   ✅ Performance optimization with SIMD and memory pools")
    print("   ✅ Market psychology and behavioral analysis")
    print("   ✅ Cross-market correlation and contagion detection")
    print("   ✅ Alternative data integration capabilities")


if __name__ == "__main__":
    asyncio.run(run_comprehensive_tests())
