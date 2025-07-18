"""
Advanced Predictive Modeling for Sentiment-Based Trading.
Implements ensemble methods, price prediction, and market regime classification.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Any, Tuple, Optional
import sqlite3
from datetime import datetime, timedelta
import logging
import joblib
import warnings
warnings.filterwarnings("ignore")


class FeatureEngineer:
    """Advanced feature engineering for sentiment-based prediction."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.scalers = {}
        self.feature_names = []
        
    def create_sentiment_features(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced sentiment features."""
        features = pd.DataFrame()
        
        # Basic sentiment statistics
        features['sentiment_mean'] = sentiment_df.groupby('symbol')['sentiment_score'].transform('mean')
        features['sentiment_std'] = sentiment_df.groupby('symbol')['sentiment_score'].transform('std')
        features['sentiment_skew'] = sentiment_df.groupby('symbol')['sentiment_score'].transform('skew')
        features['sentiment_kurt'] = sentiment_df.groupby('symbol')['sentiment_score'].transform('kurt')
        
        # Rolling statistics (different windows)
        for window in [5, 10, 20, 50]:
            features[f'sentiment_ma_{window}'] = sentiment_df.groupby('symbol')['sentiment_score'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            features[f'sentiment_std_{window}'] = sentiment_df.groupby('symbol')['sentiment_score'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
            features[f'sentiment_momentum_{window}'] = sentiment_df.groupby('symbol')['sentiment_score'].transform(
                lambda x: x.diff(window)
            )
        
        # Sentiment velocity and acceleration
        features['sentiment_velocity'] = sentiment_df.groupby('symbol')['sentiment_score'].transform(
            lambda x: x.diff()
        )
        features['sentiment_acceleration'] = sentiment_df.groupby('symbol')['sentiment_score'].transform(
            lambda x: x.diff().diff()
        )
        
        # Confidence-weighted sentiment
        if 'confidence' in sentiment_df.columns:
            features['weighted_sentiment'] = sentiment_df['sentiment_score'] * sentiment_df['confidence']
            features['confidence_mean'] = sentiment_df.groupby('symbol')['confidence'].transform('mean')
            features['confidence_std'] = sentiment_df.groupby('symbol')['confidence'].transform('std')
        
        # Volume-based features (if available)
        if 'volume' in sentiment_df.columns:
            features['sentiment_volume_product'] = sentiment_df['sentiment_score'] * sentiment_df['volume']
            features['volume_weighted_sentiment'] = sentiment_df.groupby('symbol').apply(
                lambda x: (x['sentiment_score'] * x['volume']).sum() / x['volume'].sum()
            ).reindex(sentiment_df.index, level=1)
        
        # Time-based features
        sentiment_df['hour'] = pd.to_datetime(sentiment_df['processed_at']).dt.hour
        sentiment_df['day_of_week'] = pd.to_datetime(sentiment_df['processed_at']).dt.dayofweek
        
        features['hour_sin'] = np.sin(2 * np.pi * sentiment_df['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * sentiment_df['hour'] / 24)
        features['dow_sin'] = np.sin(2 * np.pi * sentiment_df['day_of_week'] / 7)
        features['dow_cos'] = np.cos(2 * np.pi * sentiment_df['day_of_week'] / 7)
        
        # Sentiment regime features
        features['sentiment_regime'] = self._classify_sentiment_regime(sentiment_df['sentiment_score'])
        features['regime_duration'] = features.groupby(['symbol', 'sentiment_regime']).cumcount()
        
        # Cross-asset sentiment correlation
        if len(sentiment_df['symbol'].unique()) > 1:
            features['cross_asset_sentiment_corr'] = self._calculate_cross_asset_correlation(sentiment_df)
        
        return features.fillna(0)
    
    def create_price_features(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based technical features."""
        features = pd.DataFrame()
        
        # Basic price features
        features['price'] = price_df['price']
        features['log_price'] = np.log(price_df['price'])
        features['price_change'] = price_df.groupby('symbol')['price'].pct_change()
        features['log_return'] = price_df.groupby('symbol')['log_price'].diff()
        
        # Volatility features
        for window in [5, 10, 20, 50]:
            features[f'volatility_{window}'] = price_df.groupby('symbol')['log_return'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
            features[f'price_ma_{window}'] = price_df.groupby('symbol')['price'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            features[f'price_std_{window}'] = price_df.groupby('symbol')['price'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(price_df)
        features['macd'], features['macd_signal'] = self._calculate_macd(price_df)
        features['bollinger_upper'], features['bollinger_lower'] = self._calculate_bollinger_bands(price_df)
        
        # Price momentum
        for period in [1, 3, 5, 10, 20]:
            features[f'momentum_{period}'] = price_df.groupby('symbol')['price'].transform(
                lambda x: x.pct_change(period)
            )
        
        return features.fillna(0)
    
    def _classify_sentiment_regime(self, sentiment: pd.Series) -> pd.Series:
        """Classify sentiment into regimes."""
        conditions = [
            sentiment > 0.3,
            (sentiment > 0.1) & (sentiment <= 0.3),
            (sentiment >= -0.1) & (sentiment <= 0.1),
            (sentiment >= -0.3) & (sentiment < -0.1),
            sentiment < -0.3
        ]
        choices = [4, 3, 2, 1, 0]  # Very Positive, Positive, Neutral, Negative, Very Negative
        return pd.Series(np.select(conditions, choices, default=2), index=sentiment.index)
    
    def _calculate_cross_asset_correlation(self, sentiment_df: pd.DataFrame) -> pd.Series:
        """Calculate cross-asset sentiment correlation."""
        # Pivot sentiment by symbol
        sentiment_pivot = sentiment_df.pivot_table(
            index='processed_at', columns='symbol', values='sentiment_score'
        )
        
        # Calculate rolling correlation with market average
        market_sentiment = sentiment_pivot.mean(axis=1)
        correlations = []
        
        for symbol in sentiment_pivot.columns:
            corr = sentiment_pivot[symbol].rolling(20, min_periods=5).corr(market_sentiment)
            correlations.append(corr)
        
        # Combine back to original format
        result = pd.concat(correlations, keys=sentiment_pivot.columns)
        return result.reindex(sentiment_df.set_index(['symbol', 'processed_at']).index, level=[0, 1])
    
    def _calculate_rsi(self, price_df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = price_df.groupby('symbol')['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, price_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema12 = price_df.groupby('symbol')['price'].transform(lambda x: x.ewm(span=12).mean())
        ema26 = price_df.groupby('symbol')['price'].transform(lambda x: x.ewm(span=26).mean())
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return macd, signal
    
    def _calculate_bollinger_bands(self, price_df: pd.DataFrame, period: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        ma = price_df.groupby('symbol')['price'].transform(lambda x: x.rolling(period, min_periods=1).mean())
        std = price_df.groupby('symbol')['price'].transform(lambda x: x.rolling(period, min_periods=1).std())
        upper = ma + (std * 2)
        lower = ma - (std * 2)
        return upper, lower


class EnsemblePredictor:
    """Ensemble model for price movement prediction."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize ensemble predictor."""
        self.config = config or {}
        self.models = {}
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineer()
        self.is_trained = False
        
        # Model performance tracking
        self.model_scores = {}
        self.feature_importance = {}
        
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize individual models for ensemble."""
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'neural_network': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42,
                early_stopping=True
            ),
            'svr': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1)
        }
        
        return models
    
    def prepare_features(self, sentiment_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training/prediction."""
        # Create sentiment features
        sentiment_features = self.feature_engineer.create_sentiment_features(sentiment_df)
        
        # Create price features
        price_features = self.feature_engineer.create_price_features(price_df)
        
        # Combine features
        features = pd.concat([sentiment_features, price_features], axis=1)
        
        # Remove highly correlated features
        features = self._remove_correlated_features(features)
        
        return features
    
    def _remove_correlated_features(self, features: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """Remove highly correlated features."""
        corr_matrix = features.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        return features.drop(columns=to_drop)
    
    def train(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2) -> Dict[str, Any]:
        """Train ensemble model."""
        logging.info("Training ensemble predictor...")
        
        # Initialize models
        self.models = self._initialize_models()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Train individual models and evaluate
        model_performances = {}
        
        for name, model in self.models.items():
            try:
                # Cross-validation
                cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='neg_mean_squared_error')
                model_performances[name] = {
                    'cv_score': -cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                # Train on full dataset
                model.fit(X_scaled, y)
                
                # Feature importance (if available)
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(zip(X.columns, model.feature_importances_))
                elif hasattr(model, 'coef_'):
                    self.feature_importance[name] = dict(zip(X.columns, np.abs(model.coef_)))
                
                logging.info(f"Trained {name}: CV Score = {model_performances[name]['cv_score']:.4f}")
                
            except Exception as e:
                logging.error(f"Failed to train {name}: {e}")
                del self.models[name]
        
        # Create ensemble using voting regressor
        if len(self.models) >= 2:
            # Select best models for ensemble (top 5)
            best_models = sorted(model_performances.items(), key=lambda x: x[1]['cv_score'])[:5]
            ensemble_models = [(name, self.models[name]) for name, _ in best_models]
            
            self.ensemble_model = VotingRegressor(
                estimators=ensemble_models,
                n_jobs=-1
            )
            
            # Train ensemble
            self.ensemble_model.fit(X_scaled, y)
            
            # Evaluate ensemble
            ensemble_cv_scores = cross_val_score(
                self.ensemble_model, X_scaled, y, cv=tscv, scoring='neg_mean_squared_error'
            )
            
            ensemble_performance = {
                'cv_score': -ensemble_cv_scores.mean(),
                'cv_std': ensemble_cv_scores.std()
            }
            
            model_performances['ensemble'] = ensemble_performance
            
            logging.info(f"Ensemble CV Score: {ensemble_performance['cv_score']:.4f}")
        
        self.model_scores = model_performances
        self.is_trained = True
        
        return {
            'model_performances': model_performances,
            'feature_importance': self.feature_importance,
            'n_features': X.shape[1],
            'n_samples': X.shape[0]
        }
    
    def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions using ensemble model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Individual model predictions
        predictions = {}
        for name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)
                predictions[name] = pred
            except Exception as e:
                logging.error(f"Prediction failed for {name}: {e}")
        
        # Ensemble prediction
        if self.ensemble_model is not None:
            ensemble_pred = self.ensemble_model.predict(X_scaled)
            predictions['ensemble'] = ensemble_pred
        
        # Calculate prediction statistics
        pred_array = np.array(list(predictions.values()))
        prediction_stats = {
            'mean': np.mean(pred_array, axis=0),
            'std': np.std(pred_array, axis=0),
            'min': np.min(pred_array, axis=0),
            'max': np.max(pred_array, axis=0),
            'median': np.median(pred_array, axis=0)
        }
        
        return {
            'predictions': predictions,
            'ensemble_prediction': predictions.get('ensemble', prediction_stats['mean']),
            'prediction_stats': prediction_stats,
            'confidence': 1.0 / (1.0 + prediction_stats['std'])  # Inverse of uncertainty
        }
    
    def save_model(self, filepath: str):
        """Save trained model."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'models': self.models,
            'ensemble_model': self.ensemble_model,
            'scaler': self.scaler,
            'feature_engineer': self.feature_engineer,
            'model_scores': self.model_scores,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(model_data, filepath)
        logging.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model."""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.ensemble_model = model_data['ensemble_model']
        self.scaler = model_data['scaler']
        self.feature_engineer = model_data['feature_engineer']
        self.model_scores = model_data['model_scores']
        self.feature_importance = model_data['feature_importance']
        self.is_trained = True
        
        logging.info(f"Model loaded from {filepath}")


class MarketRegimeClassifier:
    """Classify market regimes based on sentiment patterns."""
    
    def __init__(self):
        """Initialize market regime classifier."""
        self.regimes = {
            0: "Bear Market",
            1: "Correction",
            2: "Neutral",
            3: "Bull Market",
            4: "Euphoria"
        }
        
    def classify_regime(self, sentiment_data: pd.DataFrame, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Classify current market regime."""
        
        # Calculate regime indicators
        avg_sentiment = sentiment_data['sentiment_score'].mean()
        sentiment_volatility = sentiment_data['sentiment_score'].std()
        price_momentum = price_data['price'].pct_change(20).iloc[-1] if len(price_data) >= 20 else 0
        
        # Regime classification logic
        if avg_sentiment > 0.5 and price_momentum > 0.2:
            regime = 4  # Euphoria
        elif avg_sentiment > 0.2 and price_momentum > 0.1:
            regime = 3  # Bull Market
        elif avg_sentiment < -0.5 and price_momentum < -0.2:
            regime = 0  # Bear Market
        elif avg_sentiment < -0.2 and price_momentum < -0.1:
            regime = 1  # Correction
        else:
            regime = 2  # Neutral
        
        return {
            'regime': regime,
            'regime_name': self.regimes[regime],
            'confidence': min(abs(avg_sentiment) + abs(price_momentum), 1.0),
            'indicators': {
                'avg_sentiment': avg_sentiment,
                'sentiment_volatility': sentiment_volatility,
                'price_momentum': price_momentum
            }
        }


# Test the predictive modeling system
if __name__ == "__main__":
    def test_predictive_modeling():
        """Test the predictive modeling system."""
        print("🤖 Testing Predictive Modeling System")
        print("=" * 50)
        
        # Generate sample data
        np.random.seed(42)
        n_samples = 1000
        
        # Sample sentiment data
        sentiment_data = pd.DataFrame({
            'symbol': np.random.choice(['BTC', 'AAPL', 'TSLA'], n_samples),
            'sentiment_score': np.random.normal(0, 0.3, n_samples),
            'confidence': np.random.uniform(0.5, 1.0, n_samples),
            'processed_at': pd.date_range('2023-01-01', periods=n_samples, freq='H')
        })
        
        # Sample price data
        price_data = pd.DataFrame({
            'symbol': np.random.choice(['BTC', 'AAPL', 'TSLA'], n_samples),
            'price': np.random.lognormal(4, 0.2, n_samples),
            'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='H')
        })
        
        # Initialize predictor
        predictor = EnsemblePredictor()
        
        # Prepare features
        print("📊 Preparing features...")
        features = predictor.prepare_features(sentiment_data, price_data)
        print(f"   Features shape: {features.shape}")
        
        # Create target (next period price change)
        target = price_data.groupby('symbol')['price'].pct_change().shift(-1).fillna(0)
        
        # Remove NaN values
        valid_idx = ~(features.isna().any(axis=1) | target.isna())
        X = features[valid_idx]
        y = target[valid_idx]
        
        print(f"   Training samples: {len(X)}")
        
        # Train model
        print("🤖 Training ensemble model...")
        training_results = predictor.train(X, y)
        
        print("📈 Model Performance:")
        for model_name, performance in training_results['model_performances'].items():
            print(f"   {model_name}: {performance['cv_score']:.4f} ± {performance['cv_std']:.4f}")
        
        # Test prediction
        print("\n🔮 Testing predictions...")
        test_features = X.iloc[-10:]
        predictions = predictor.predict(test_features)
        
        print(f"   Ensemble prediction shape: {predictions['ensemble_prediction'].shape}")
        print(f"   Prediction confidence: {predictions['confidence'].mean():.3f}")
        
        # Test market regime classifier
        print("\n📊 Testing market regime classification...")
        regime_classifier = MarketRegimeClassifier()
        regime_result = regime_classifier.classify_regime(sentiment_data, price_data)
        
        print(f"   Current regime: {regime_result['regime_name']}")
        print(f"   Confidence: {regime_result['confidence']:.3f}")
        print(f"   Avg sentiment: {regime_result['indicators']['avg_sentiment']:.3f}")
        
        print("\n✅ Predictive modeling test completed!")
    
    test_predictive_modeling()
