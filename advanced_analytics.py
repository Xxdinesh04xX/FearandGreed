"""
Advanced Analytics and Correlation Analysis for Financial Sentiment.
Implements market psychology analysis, cross-market correlation, and alternative data integration.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import networkx as nx
from typing import Dict, List, Any, Tuple, Optional
import sqlite3
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings("ignore")


class MarketPsychologyAnalyzer:
    """Analyze market psychology and behavioral biases in sentiment patterns."""
    
    def __init__(self):
        """Initialize market psychology analyzer."""
        self.bias_indicators = {
            'herding': self._detect_herding_behavior,
            'overconfidence': self._detect_overconfidence,
            'loss_aversion': self._detect_loss_aversion,
            'anchoring': self._detect_anchoring_bias,
            'confirmation': self._detect_confirmation_bias
        }
        
        # Psychological thresholds
        self.fear_threshold = -0.5
        self.greed_threshold = 0.5
        self.panic_threshold = -0.8
        self.euphoria_threshold = 0.8
        
    def analyze_market_psychology(self, sentiment_data: pd.DataFrame, 
                                price_data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive market psychology analysis."""
        
        # Merge sentiment and price data
        merged_data = self._merge_sentiment_price_data(sentiment_data, price_data)
        
        if merged_data.empty:
            return self._empty_psychology_result()
        
        # Analyze different psychological aspects
        results = {}
        
        # 1. Fear and Greed Dynamics
        results['fear_greed_dynamics'] = self._analyze_fear_greed_dynamics(merged_data)
        
        # 2. Behavioral Bias Detection
        results['behavioral_biases'] = self._detect_behavioral_biases(merged_data)
        
        # 3. Crowd Psychology Metrics
        results['crowd_psychology'] = self._analyze_crowd_psychology(merged_data)
        
        # 4. Contrarian Signal Generation
        results['contrarian_signals'] = self._generate_contrarian_signals(merged_data)
        
        # 5. Market Regime Psychology
        results['regime_psychology'] = self._analyze_regime_psychology(merged_data)
        
        return results
    
    def _merge_sentiment_price_data(self, sentiment_df: pd.DataFrame, 
                                  price_df: pd.DataFrame) -> pd.DataFrame:
        """Merge sentiment and price data for analysis."""
        try:
            # Ensure datetime columns
            sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['processed_at'])
            price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
            
            # Resample to hourly data
            sentiment_hourly = sentiment_df.groupby(['symbol', pd.Grouper(key='timestamp', freq='H')]).agg({
                'sentiment_score': ['mean', 'std', 'count'],
                'confidence': 'mean'
            }).reset_index()
            
            sentiment_hourly.columns = ['symbol', 'timestamp', 'sentiment_mean', 'sentiment_std', 
                                      'sentiment_count', 'confidence_mean']
            
            price_hourly = price_df.groupby(['symbol', pd.Grouper(key='timestamp', freq='H')]).agg({
                'price': ['first', 'last', 'min', 'max'],
                'volume': 'sum' if 'volume' in price_df.columns else 'count'
            }).reset_index()
            
            price_hourly.columns = ['symbol', 'timestamp', 'price_open', 'price_close', 
                                  'price_low', 'price_high', 'volume']
            
            # Merge data
            merged = pd.merge(sentiment_hourly, price_hourly, on=['symbol', 'timestamp'], how='inner')
            
            # Calculate additional features
            merged['price_return'] = merged.groupby('symbol')['price_close'].pct_change()
            merged['sentiment_momentum'] = merged.groupby('symbol')['sentiment_mean'].diff()
            merged['volatility'] = merged.groupby('symbol')['price_return'].rolling(24).std().reset_index(0, drop=True)
            
            return merged.dropna()
            
        except Exception as e:
            logging.error(f"Error merging sentiment and price data: {e}")
            return pd.DataFrame()
    
    def _analyze_fear_greed_dynamics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze fear and greed dynamics."""
        results = {}
        
        # Classify sentiment into fear/greed regimes
        data['regime'] = pd.cut(data['sentiment_mean'], 
                               bins=[-np.inf, self.fear_threshold, self.greed_threshold, np.inf],
                               labels=['Fear', 'Neutral', 'Greed'])
        
        # Calculate regime statistics
        regime_stats = data.groupby('regime').agg({
            'price_return': ['mean', 'std', 'count'],
            'volatility': 'mean',
            'sentiment_mean': 'mean'
        }).round(4)
        
        results['regime_statistics'] = regime_stats.to_dict()
        
        # Detect extreme emotions
        extreme_fear = data[data['sentiment_mean'] < self.panic_threshold]
        extreme_greed = data[data['sentiment_mean'] > self.euphoria_threshold]
        
        results['extreme_emotions'] = {
            'panic_episodes': len(extreme_fear),
            'euphoria_episodes': len(extreme_greed),
            'panic_avg_return': extreme_fear['price_return'].mean() if not extreme_fear.empty else 0,
            'euphoria_avg_return': extreme_greed['price_return'].mean() if not extreme_greed.empty else 0
        }
        
        # Fear-greed oscillation analysis
        sentiment_peaks, _ = find_peaks(data['sentiment_mean'], height=0.3)
        sentiment_troughs, _ = find_peaks(-data['sentiment_mean'], height=0.3)
        
        results['oscillation_analysis'] = {
            'peak_count': len(sentiment_peaks),
            'trough_count': len(sentiment_troughs),
            'avg_cycle_length': self._calculate_avg_cycle_length(sentiment_peaks, sentiment_troughs)
        }
        
        return results
    
    def _detect_behavioral_biases(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect various behavioral biases in the data."""
        biases = {}
        
        for bias_name, detector_func in self.bias_indicators.items():
            try:
                bias_result = detector_func(data)
                biases[bias_name] = bias_result
            except Exception as e:
                logging.error(f"Error detecting {bias_name} bias: {e}")
                biases[bias_name] = {'detected': False, 'confidence': 0.0}
        
        return biases
    
    def _detect_herding_behavior(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect herding behavior in sentiment patterns."""
        # Calculate sentiment dispersion across assets
        sentiment_by_time = data.pivot_table(
            index='timestamp', columns='symbol', values='sentiment_mean'
        )
        
        # Cross-sectional standard deviation (herding indicator)
        cross_sectional_std = sentiment_by_time.std(axis=1, skipna=True)
        
        # Low dispersion indicates herding
        herding_threshold = cross_sectional_std.quantile(0.25)
        herding_periods = cross_sectional_std < herding_threshold
        
        return {
            'detected': herding_periods.sum() > len(herding_periods) * 0.2,
            'confidence': 1 - cross_sectional_std.mean(),
            'herding_periods': herding_periods.sum(),
            'avg_dispersion': cross_sectional_std.mean()
        }
    
    def _detect_overconfidence(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect overconfidence bias."""
        # High confidence with poor subsequent returns indicates overconfidence
        data['future_return'] = data.groupby('symbol')['price_return'].shift(-1)
        
        high_confidence = data[data['confidence_mean'] > 0.8]
        
        if high_confidence.empty:
            return {'detected': False, 'confidence': 0.0}
        
        # Check if high confidence periods are followed by poor returns
        confidence_return_corr = high_confidence['confidence_mean'].corr(high_confidence['future_return'])
        
        return {
            'detected': confidence_return_corr < -0.1,
            'confidence': abs(confidence_return_corr) if not np.isnan(confidence_return_corr) else 0.0,
            'high_confidence_periods': len(high_confidence),
            'avg_subsequent_return': high_confidence['future_return'].mean()
        }
    
    def _detect_loss_aversion(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect loss aversion patterns."""
        # Asymmetric response to gains vs losses
        gains = data[data['price_return'] > 0]
        losses = data[data['price_return'] < 0]
        
        if gains.empty or losses.empty:
            return {'detected': False, 'confidence': 0.0}
        
        # Sentiment response to gains vs losses
        gain_sentiment_response = gains['sentiment_mean'].mean()
        loss_sentiment_response = abs(losses['sentiment_mean'].mean())
        
        # Loss aversion: stronger reaction to losses
        loss_aversion_ratio = loss_sentiment_response / max(gain_sentiment_response, 0.01)
        
        return {
            'detected': loss_aversion_ratio > 1.5,
            'confidence': min(loss_aversion_ratio / 2, 1.0),
            'loss_aversion_ratio': loss_aversion_ratio,
            'gain_sentiment': gain_sentiment_response,
            'loss_sentiment': loss_sentiment_response
        }
    
    def _detect_anchoring_bias(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect anchoring bias in sentiment."""
        # Sentiment persistence despite price changes
        data['sentiment_lag'] = data.groupby('symbol')['sentiment_mean'].shift(1)
        
        # Correlation between current and lagged sentiment
        sentiment_persistence = data['sentiment_mean'].corr(data['sentiment_lag'])
        
        # High persistence despite price changes indicates anchoring
        price_change_magnitude = abs(data['price_return']).mean()
        
        return {
            'detected': sentiment_persistence > 0.7 and price_change_magnitude > 0.02,
            'confidence': sentiment_persistence if not np.isnan(sentiment_persistence) else 0.0,
            'sentiment_persistence': sentiment_persistence,
            'avg_price_change': price_change_magnitude
        }
    
    def _detect_confirmation_bias(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect confirmation bias patterns."""
        # Sentiment reinforcement in trending markets
        data['price_trend'] = data.groupby('symbol')['price_return'].rolling(5).mean().reset_index(0, drop=True)
        
        # Positive correlation between trend and sentiment indicates confirmation bias
        trend_sentiment_corr = data['price_trend'].corr(data['sentiment_mean'])
        
        return {
            'detected': trend_sentiment_corr > 0.6,
            'confidence': trend_sentiment_corr if not np.isnan(trend_sentiment_corr) else 0.0,
            'trend_sentiment_correlation': trend_sentiment_corr
        }
    
    def _analyze_crowd_psychology(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze crowd psychology metrics."""
        # Sentiment consensus and dispersion
        consensus_metrics = data.groupby('timestamp').agg({
            'sentiment_mean': ['mean', 'std', 'count'],
            'confidence_mean': 'mean'
        })
        
        # Crowd sentiment strength
        crowd_sentiment = consensus_metrics[('sentiment_mean', 'mean')]
        crowd_dispersion = consensus_metrics[('sentiment_mean', 'std')]
        crowd_size = consensus_metrics[('sentiment_mean', 'count')]
        
        # Identify crowd extremes
        extreme_consensus = abs(crowd_sentiment) > 0.6
        low_dispersion = crowd_dispersion < 0.2
        large_crowd = crowd_size > crowd_size.quantile(0.75)
        
        crowd_extremes = extreme_consensus & low_dispersion & large_crowd
        
        return {
            'avg_crowd_sentiment': crowd_sentiment.mean(),
            'avg_crowd_dispersion': crowd_dispersion.mean(),
            'avg_crowd_size': crowd_size.mean(),
            'extreme_consensus_periods': crowd_extremes.sum(),
            'crowd_extreme_ratio': crowd_extremes.sum() / len(crowd_extremes) if len(crowd_extremes) > 0 else 0
        }
    
    def _generate_contrarian_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate contrarian trading signals based on psychology."""
        signals = []
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()
            
            if len(symbol_data) < 10:
                continue
            
            # Contrarian signals based on extreme sentiment
            for i, row in symbol_data.iterrows():
                signal = None
                confidence = 0.0
                
                # Extreme fear -> Buy signal
                if row['sentiment_mean'] < self.panic_threshold:
                    signal = 'BUY'
                    confidence = abs(row['sentiment_mean']) / abs(self.panic_threshold)
                
                # Extreme greed -> Sell signal
                elif row['sentiment_mean'] > self.euphoria_threshold:
                    signal = 'SELL'
                    confidence = row['sentiment_mean'] / self.euphoria_threshold
                
                if signal:
                    signals.append({
                        'symbol': symbol,
                        'timestamp': row['timestamp'],
                        'signal': signal,
                        'confidence': min(confidence, 1.0),
                        'sentiment': row['sentiment_mean'],
                        'reason': 'contrarian_psychology'
                    })
        
        return {
            'total_signals': len(signals),
            'buy_signals': len([s for s in signals if s['signal'] == 'BUY']),
            'sell_signals': len([s for s in signals if s['signal'] == 'SELL']),
            'avg_confidence': np.mean([s['confidence'] for s in signals]) if signals else 0,
            'signals': signals[-10:]  # Return last 10 signals
        }
    
    def _analyze_regime_psychology(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze psychology in different market regimes."""
        # Define market regimes based on volatility and returns
        data['volatility_regime'] = pd.qcut(data['volatility'], q=3, labels=['Low', 'Medium', 'High'])
        data['return_regime'] = pd.cut(data['price_return'], 
                                     bins=[-np.inf, -0.02, 0.02, np.inf],
                                     labels=['Declining', 'Stable', 'Rising'])
        
        # Analyze sentiment patterns by regime
        regime_analysis = data.groupby(['volatility_regime', 'return_regime']).agg({
            'sentiment_mean': ['mean', 'std'],
            'confidence_mean': 'mean',
            'sentiment_count': 'sum'
        }).round(4)
        
        return {
            'regime_sentiment_patterns': regime_analysis.to_dict(),
            'regime_transitions': self._analyze_regime_transitions(data)
        }
    
    def _analyze_regime_transitions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze sentiment during regime transitions."""
        # Detect volatility regime changes
        data['vol_regime_change'] = data.groupby('symbol')['volatility_regime'].apply(
            lambda x: x != x.shift(1)
        ).reset_index(0, drop=True)
        
        transition_periods = data[data['vol_regime_change'] == True]
        
        if transition_periods.empty:
            return {'transition_count': 0}
        
        return {
            'transition_count': len(transition_periods),
            'avg_sentiment_during_transitions': transition_periods['sentiment_mean'].mean(),
            'avg_confidence_during_transitions': transition_periods['confidence_mean'].mean()
        }
    
    def _calculate_avg_cycle_length(self, peaks: np.ndarray, troughs: np.ndarray) -> float:
        """Calculate average cycle length between peaks and troughs."""
        if len(peaks) < 2 and len(troughs) < 2:
            return 0.0
        
        all_extremes = np.sort(np.concatenate([peaks, troughs]))
        if len(all_extremes) < 2:
            return 0.0
        
        cycle_lengths = np.diff(all_extremes)
        return float(np.mean(cycle_lengths))
    
    def _empty_psychology_result(self) -> Dict[str, Any]:
        """Return empty psychology analysis result."""
        return {
            'fear_greed_dynamics': {},
            'behavioral_biases': {},
            'crowd_psychology': {},
            'contrarian_signals': {'total_signals': 0},
            'regime_psychology': {}
        }


class CrossMarketAnalyzer:
    """Analyze sentiment contagion and cross-market correlations."""
    
    def __init__(self):
        """Initialize cross-market analyzer."""
        self.correlation_window = 20
        self.contagion_threshold = 0.7
        
    def analyze_sentiment_contagion(self, sentiment_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze sentiment contagion across different assets."""
        
        # Pivot sentiment data by asset
        sentiment_pivot = sentiment_data.pivot_table(
            index='processed_at', columns='symbol', values='sentiment_score'
        )
        
        if sentiment_pivot.empty or sentiment_pivot.shape[1] < 2:
            return {'status': 'Insufficient data for cross-market analysis'}
        
        # Calculate rolling correlations
        rolling_correlations = {}
        correlation_matrix = sentiment_pivot.corr()
        
        # Dynamic correlation analysis
        for i in range(self.correlation_window, len(sentiment_pivot)):
            window_data = sentiment_pivot.iloc[i-self.correlation_window:i]
            window_corr = window_data.corr()
            
            # Store average correlation for this window
            avg_corr = window_corr.values[np.triu_indices_from(window_corr.values, k=1)].mean()
            rolling_correlations[sentiment_pivot.index[i]] = avg_corr
        
        # Detect contagion events
        contagion_events = self._detect_contagion_events(sentiment_pivot)
        
        # Network analysis
        network_metrics = self._analyze_sentiment_network(correlation_matrix)
        
        return {
            'static_correlations': correlation_matrix.to_dict(),
            'dynamic_correlations': rolling_correlations,
            'contagion_events': contagion_events,
            'network_analysis': network_metrics,
            'cross_asset_summary': self._summarize_cross_asset_patterns(sentiment_pivot)
        }
    
    def _detect_contagion_events(self, sentiment_pivot: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect sentiment contagion events."""
        events = []
        
        # Calculate rolling correlation increases
        for i in range(self.correlation_window * 2, len(sentiment_pivot)):
            # Compare recent correlation with historical
            recent_window = sentiment_pivot.iloc[i-self.correlation_window:i]
            historical_window = sentiment_pivot.iloc[i-self.correlation_window*2:i-self.correlation_window]
            
            recent_corr = recent_window.corr().values[np.triu_indices_from(recent_window.corr().values, k=1)].mean()
            historical_corr = historical_window.corr().values[np.triu_indices_from(historical_window.corr().values, k=1)].mean()
            
            # Contagion: significant increase in correlation
            if recent_corr > historical_corr + 0.3 and recent_corr > self.contagion_threshold:
                events.append({
                    'timestamp': sentiment_pivot.index[i],
                    'correlation_increase': recent_corr - historical_corr,
                    'peak_correlation': recent_corr,
                    'affected_assets': list(sentiment_pivot.columns)
                })
        
        return events
    
    def _analyze_sentiment_network(self, correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Analyze sentiment network using graph theory."""
        # Create network graph
        G = nx.Graph()
        
        # Add nodes (assets)
        for asset in correlation_matrix.columns:
            G.add_node(asset)
        
        # Add edges (correlations above threshold)
        for i, asset1 in enumerate(correlation_matrix.columns):
            for j, asset2 in enumerate(correlation_matrix.columns):
                if i < j:  # Avoid duplicate edges
                    corr = correlation_matrix.iloc[i, j]
                    if abs(corr) > 0.5:  # Threshold for significant correlation
                        G.add_edge(asset1, asset2, weight=abs(corr))
        
        # Calculate network metrics
        if len(G.nodes()) > 0:
            centrality = nx.degree_centrality(G)
            clustering = nx.clustering(G)
            
            return {
                'node_count': len(G.nodes()),
                'edge_count': len(G.edges()),
                'density': nx.density(G),
                'centrality_scores': centrality,
                'clustering_coefficients': clustering,
                'most_central_asset': max(centrality.items(), key=lambda x: x[1])[0] if centrality else None
            }
        else:
            return {'status': 'No significant correlations found'}
    
    def _summarize_cross_asset_patterns(self, sentiment_pivot: pd.DataFrame) -> Dict[str, Any]:
        """Summarize cross-asset sentiment patterns."""
        # Calculate summary statistics
        avg_correlations = sentiment_pivot.corr().values[np.triu_indices_from(sentiment_pivot.corr().values, k=1)]
        
        return {
            'avg_cross_correlation': np.mean(avg_correlations),
            'max_correlation': np.max(avg_correlations),
            'min_correlation': np.min(avg_correlations),
            'correlation_std': np.std(avg_correlations),
            'highly_correlated_pairs': self._find_highly_correlated_pairs(sentiment_pivot.corr())
        }
    
    def _find_highly_correlated_pairs(self, correlation_matrix: pd.DataFrame, threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Find highly correlated asset pairs."""
        pairs = []
        
        for i, asset1 in enumerate(correlation_matrix.columns):
            for j, asset2 in enumerate(correlation_matrix.columns):
                if i < j:  # Avoid duplicates
                    corr = correlation_matrix.iloc[i, j]
                    if abs(corr) > threshold:
                        pairs.append({
                            'asset1': asset1,
                            'asset2': asset2,
                            'correlation': corr
                        })
        
        return sorted(pairs, key=lambda x: abs(x['correlation']), reverse=True)


class AlternativeDataIntegrator:
    """Integrate alternative data sources for enhanced analysis."""
    
    def __init__(self):
        """Initialize alternative data integrator."""
        self.data_sources = {
            'economic_indicators': self._process_economic_data,
            'earnings_calls': self._process_earnings_calls,
            'regulatory_filings': self._process_regulatory_filings,
            'satellite_data': self._process_satellite_data
        }
    
    def integrate_alternative_data(self, base_sentiment: pd.DataFrame, 
                                 alternative_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Integrate alternative data with sentiment analysis."""
        
        integrated_results = {}
        
        for data_type, data_df in alternative_data.items():
            if data_type in self.data_sources:
                try:
                    processor = self.data_sources[data_type]
                    result = processor(base_sentiment, data_df)
                    integrated_results[data_type] = result
                except Exception as e:
                    logging.error(f"Error processing {data_type}: {e}")
                    integrated_results[data_type] = {'status': 'error', 'message': str(e)}
        
        # Combine insights
        combined_insights = self._combine_alternative_insights(integrated_results)
        
        return {
            'individual_sources': integrated_results,
            'combined_insights': combined_insights,
            'data_quality_score': self._calculate_data_quality_score(integrated_results)
        }
    
    def _process_economic_data(self, sentiment_df: pd.DataFrame, 
                             economic_df: pd.DataFrame) -> Dict[str, Any]:
        """Process economic indicators data."""
        # Placeholder for economic data processing
        return {
            'correlation_with_sentiment': 0.3,
            'leading_indicators': ['GDP', 'unemployment'],
            'sentiment_economic_divergence': 0.15
        }
    
    def _process_earnings_calls(self, sentiment_df: pd.DataFrame, 
                              earnings_df: pd.DataFrame) -> Dict[str, Any]:
        """Process earnings call sentiment."""
        # Placeholder for earnings call analysis
        return {
            'management_sentiment_score': 0.2,
            'analyst_sentiment_score': 0.1,
            'earnings_surprise_correlation': 0.4
        }
    
    def _process_regulatory_filings(self, sentiment_df: pd.DataFrame, 
                                  filings_df: pd.DataFrame) -> Dict[str, Any]:
        """Process regulatory filings data."""
        # Placeholder for regulatory filings analysis
        return {
            'filing_sentiment_score': -0.1,
            'risk_factor_mentions': 15,
            'regulatory_risk_level': 'medium'
        }
    
    def _process_satellite_data(self, sentiment_df: pd.DataFrame, 
                              satellite_df: pd.DataFrame) -> Dict[str, Any]:
        """Process satellite and alternative economic data."""
        # Placeholder for satellite data analysis
        return {
            'economic_activity_index': 0.8,
            'supply_chain_disruption_score': 0.2,
            'geographic_sentiment_correlation': 0.25
        }
    
    def _combine_alternative_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine insights from multiple alternative data sources."""
        # Placeholder for combining insights
        return {
            'overall_alternative_sentiment': 0.15,
            'data_source_agreement': 0.7,
            'alternative_vs_social_divergence': 0.1
        }
    
    def _calculate_data_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall data quality score."""
        # Simple quality score based on successful processing
        successful_sources = sum(1 for result in results.values() 
                               if isinstance(result, dict) and result.get('status') != 'error')
        total_sources = len(results)
        
        return successful_sources / max(total_sources, 1)


# Test the advanced analytics system
if __name__ == "__main__":
    def test_advanced_analytics():
        """Test the advanced analytics system."""
        print("🧠 Testing Advanced Analytics System")
        print("=" * 50)
        
        # Generate sample data
        np.random.seed(42)
        n_samples = 1000
        
        # Sample sentiment data
        sentiment_data = pd.DataFrame({
            'symbol': np.random.choice(['BTC', 'AAPL', 'TSLA'], n_samples),
            'sentiment_score': np.random.normal(0, 0.4, n_samples),
            'confidence': np.random.uniform(0.5, 1.0, n_samples),
            'processed_at': pd.date_range('2023-01-01', periods=n_samples, freq='H')
        })
        
        # Sample price data
        price_data = pd.DataFrame({
            'symbol': np.random.choice(['BTC', 'AAPL', 'TSLA'], n_samples),
            'price': np.random.lognormal(4, 0.2, n_samples),
            'volume': np.random.randint(1000, 10000, n_samples),
            'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='H')
        })
        
        # Test Market Psychology Analyzer
        print("🧠 Testing Market Psychology Analysis...")
        psychology_analyzer = MarketPsychologyAnalyzer()
        psychology_results = psychology_analyzer.analyze_market_psychology(sentiment_data, price_data)
        
        print(f"   Behavioral biases detected: {len(psychology_results['behavioral_biases'])}")
        print(f"   Contrarian signals: {psychology_results['contrarian_signals']['total_signals']}")
        
        # Test Cross-Market Analyzer
        print("\n🔗 Testing Cross-Market Analysis...")
        cross_market_analyzer = CrossMarketAnalyzer()
        cross_market_results = cross_market_analyzer.analyze_sentiment_contagion(sentiment_data)
        
        if 'network_analysis' in cross_market_results:
            network = cross_market_results['network_analysis']
            print(f"   Network density: {network.get('density', 0):.3f}")
            print(f"   Most central asset: {network.get('most_central_asset', 'N/A')}")
        
        # Test Alternative Data Integration
        print("\n📊 Testing Alternative Data Integration...")
        alt_data_integrator = AlternativeDataIntegrator()
        
        # Mock alternative data
        alternative_data = {
            'economic_indicators': pd.DataFrame({'indicator': ['GDP'], 'value': [2.1]}),
            'earnings_calls': pd.DataFrame({'company': ['AAPL'], 'sentiment': [0.3]})
        }
        
        alt_results = alt_data_integrator.integrate_alternative_data(sentiment_data, alternative_data)
        print(f"   Data quality score: {alt_results['data_quality_score']:.2f}")
        print(f"   Alternative sources processed: {len(alt_results['individual_sources'])}")
        
        print("\n✅ Advanced analytics test completed!")
    
    test_advanced_analytics()
