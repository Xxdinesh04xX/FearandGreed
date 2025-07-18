"""
Backtesting engine for GoQuant Sentiment Trader.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..config import Config
from ..utils.logger import get_logger
from ..database.manager import DatabaseManager
from ..sentiment.analyzer import SentimentAnalyzer
from ..signals.generator import SignalGenerator
from ..signals.models import TradingSignalData, SignalType


@dataclass
class BacktestResult:
    """Results of a backtesting run."""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_trade_return: float
    trades: List[Dict[str, Any]]
    equity_curve: pd.DataFrame
    performance_metrics: Dict[str, Any]


class BacktestingEngine:
    """
    Backtesting engine for sentiment-based trading strategies.
    
    Simulates historical trading based on sentiment analysis and signal generation.
    """
    
    def __init__(self, config: Config):
        """Initialize the backtesting engine."""
        self.config = config
        self.logger = get_logger(__name__)
        
        # Components
        self.db_manager: Optional[DatabaseManager] = None
        self.sentiment_analyzer: Optional[SentimentAnalyzer] = None
        self.signal_generator: Optional[SignalGenerator] = None
        
        # Backtesting parameters
        self.initial_capital = config.backtest_initial_capital
        self.commission_rate = 0.001  # 0.1% commission
        self.slippage_rate = 0.0005   # 0.05% slippage
        
        # State tracking
        self.current_capital = self.initial_capital
        self.positions = {}  # symbol -> position_size
        self.trades = []
        self.equity_curve = []
    
    def set_components(self, db_manager: DatabaseManager, 
                      sentiment_analyzer: SentimentAnalyzer,
                      signal_generator: SignalGenerator) -> None:
        """Set the required components for backtesting."""
        self.db_manager = db_manager
        self.sentiment_analyzer = sentiment_analyzer
        self.signal_generator = signal_generator
    
    async def run_backtest(self, 
                          start_date: datetime,
                          end_date: datetime,
                          symbols: List[str]) -> BacktestResult:
        """
        Run a complete backtesting simulation.
        
        Args:
            start_date: Start date for backtesting
            end_date: End date for backtesting
            symbols: List of symbols to trade
            
        Returns:
            BacktestResult with performance metrics
        """
        self.logger.info(f"Starting backtest from {start_date} to {end_date}")
        self.logger.info(f"Symbols: {symbols}")
        self.logger.info(f"Initial capital: ${self.initial_capital:,.2f}")
        
        # Reset state
        self._reset_state()
        
        try:
            # Get historical data
            historical_data = await self._get_historical_data(start_date, end_date, symbols)
            
            if not historical_data:
                raise ValueError("No historical data available for backtesting")
            
            # Run simulation
            await self._run_simulation(historical_data, symbols)
            
            # Calculate results
            result = self._calculate_results(start_date, end_date)
            
            self.logger.info(f"Backtest completed. Final capital: ${result.final_capital:,.2f}")
            self.logger.info(f"Total return: {result.total_return:.2%}")
            self.logger.info(f"Sharpe ratio: {result.sharpe_ratio:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Backtesting failed: {e}")
            raise
    
    def _reset_state(self) -> None:
        """Reset backtesting state."""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
    
    async def _get_historical_data(self, 
                                  start_date: datetime,
                                  end_date: datetime,
                                  symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Get historical price and sentiment data."""
        historical_data = {}
        
        for symbol in symbols:
            try:
                # In a real implementation, this would fetch actual historical data
                # For now, generate synthetic data for demonstration
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                
                # Generate synthetic price data
                np.random.seed(42)  # For reproducible results
                returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
                prices = [100.0]  # Starting price
                
                for ret in returns[1:]:
                    prices.append(prices[-1] * (1 + ret))
                
                # Generate synthetic sentiment data
                sentiment_scores = np.random.normal(0, 0.3, len(dates))
                sentiment_scores = np.clip(sentiment_scores, -1, 1)
                
                df = pd.DataFrame({
                    'date': dates,
                    'price': prices,
                    'sentiment_score': sentiment_scores,
                    'volume': np.random.randint(1000000, 10000000, len(dates))
                })
                
                historical_data[symbol] = df
                
            except Exception as e:
                self.logger.error(f"Failed to get historical data for {symbol}: {e}")
        
        return historical_data
    
    async def _run_simulation(self, 
                             historical_data: Dict[str, pd.DataFrame],
                             symbols: List[str]) -> None:
        """Run the backtesting simulation."""
        # Get all unique dates
        all_dates = set()
        for df in historical_data.values():
            all_dates.update(df['date'])
        
        sorted_dates = sorted(all_dates)
        
        for current_date in sorted_dates:
            # Update equity curve
            portfolio_value = self._calculate_portfolio_value(historical_data, current_date)
            self.equity_curve.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'cash': self.current_capital,
                'positions_value': portfolio_value - self.current_capital
            })
            
            # Generate signals for this date
            signals = await self._generate_signals_for_date(historical_data, current_date, symbols)
            
            # Execute trades based on signals
            for signal in signals:
                await self._execute_trade(signal, historical_data, current_date)
    
    async def _generate_signals_for_date(self,
                                        historical_data: Dict[str, pd.DataFrame],
                                        current_date: datetime,
                                        symbols: List[str]) -> List[TradingSignalData]:
        """Generate trading signals for a specific date."""
        signals = []
        
        for symbol in symbols:
            if symbol not in historical_data:
                continue
            
            df = historical_data[symbol]
            current_data = df[df['date'] <= current_date]
            
            if len(current_data) < 5:  # Need minimum data points
                continue
            
            # Get recent sentiment data (last 5 days)
            recent_sentiment = current_data.tail(5)['sentiment_score'].values
            
            # Simple signal generation based on sentiment
            avg_sentiment = np.mean(recent_sentiment)
            sentiment_trend = recent_sentiment[-1] - recent_sentiment[0]
            
            # Generate signal based on sentiment
            if avg_sentiment > 0.2 and sentiment_trend > 0.1:
                signal_type = SignalType.BUY
                confidence = min(0.9, abs(avg_sentiment) + abs(sentiment_trend))
            elif avg_sentiment < -0.2 and sentiment_trend < -0.1:
                signal_type = SignalType.SELL
                confidence = min(0.9, abs(avg_sentiment) + abs(sentiment_trend))
            else:
                continue  # No signal
            
            # Create signal
            from ..signals.models import SignalStrength
            
            strength = SignalStrength.STRONG if confidence > 0.7 else \
                      SignalStrength.MODERATE if confidence > 0.5 else \
                      SignalStrength.WEAK
            
            signal = TradingSignalData(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                sentiment_score=avg_sentiment,
                generated_at=current_date
            )
            
            signals.append(signal)
        
        return signals
    
    async def _execute_trade(self,
                           signal: TradingSignalData,
                           historical_data: Dict[str, pd.DataFrame],
                           current_date: datetime) -> None:
        """Execute a trade based on a signal."""
        symbol = signal.symbol
        
        if symbol not in historical_data:
            return
        
        df = historical_data[symbol]
        current_data = df[df['date'] == current_date]
        
        if current_data.empty:
            return
        
        current_price = current_data.iloc[0]['price']
        
        # Calculate position size (risk management)
        position_size = self._calculate_position_size(signal, current_price)
        
        if position_size == 0:
            return
        
        # Apply slippage and commission
        if signal.signal_type == SignalType.BUY:
            execution_price = current_price * (1 + self.slippage_rate)
            trade_value = position_size * execution_price
            commission = trade_value * self.commission_rate
            total_cost = trade_value + commission
            
            if total_cost <= self.current_capital:
                # Execute buy
                self.current_capital -= total_cost
                self.positions[symbol] = self.positions.get(symbol, 0) + position_size
                
                trade = {
                    'date': current_date,
                    'symbol': symbol,
                    'action': 'BUY',
                    'quantity': position_size,
                    'price': execution_price,
                    'value': trade_value,
                    'commission': commission,
                    'signal_confidence': signal.confidence
                }
                self.trades.append(trade)
        
        elif signal.signal_type == SignalType.SELL:
            current_position = self.positions.get(symbol, 0)
            if current_position > 0:
                # Execute sell
                sell_quantity = min(position_size, current_position)
                execution_price = current_price * (1 - self.slippage_rate)
                trade_value = sell_quantity * execution_price
                commission = trade_value * self.commission_rate
                net_proceeds = trade_value - commission
                
                self.current_capital += net_proceeds
                self.positions[symbol] -= sell_quantity
                
                trade = {
                    'date': current_date,
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': sell_quantity,
                    'price': execution_price,
                    'value': trade_value,
                    'commission': commission,
                    'signal_confidence': signal.confidence
                }
                self.trades.append(trade)
    
    def _calculate_position_size(self, signal: TradingSignalData, price: float) -> float:
        """Calculate position size based on signal strength and risk management."""
        # Base position size as percentage of capital
        base_percentage = {
            'WEAK': 0.02,      # 2%
            'MODERATE': 0.05,   # 5%
            'STRONG': 0.10      # 10%
        }.get(signal.strength.value, 0.02)
        
        # Adjust by confidence
        adjusted_percentage = base_percentage * signal.confidence
        
        # Calculate position size
        position_value = self.current_capital * adjusted_percentage
        position_size = position_value / price
        
        return position_size
    
    def _calculate_portfolio_value(self, 
                                  historical_data: Dict[str, pd.DataFrame],
                                  current_date: datetime) -> float:
        """Calculate total portfolio value."""
        total_value = self.current_capital
        
        for symbol, quantity in self.positions.items():
            if quantity > 0 and symbol in historical_data:
                df = historical_data[symbol]
                current_data = df[df['date'] == current_date]
                
                if not current_data.empty:
                    current_price = current_data.iloc[0]['price']
                    total_value += quantity * current_price
        
        return total_value
    
    def _calculate_results(self, start_date: datetime, end_date: datetime) -> BacktestResult:
        """Calculate backtesting results and performance metrics."""
        if not self.equity_curve:
            raise ValueError("No equity curve data available")
        
        equity_df = pd.DataFrame(self.equity_curve)
        
        # Basic metrics
        final_capital = equity_df['portfolio_value'].iloc[-1]
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        
        # Annualized return
        days = (end_date - start_date).days
        years = days / 365.25
        annualized_return = (final_capital / self.initial_capital) ** (1 / years) - 1 if years > 0 else 0
        
        # Sharpe ratio (simplified)
        returns = equity_df['portfolio_value'].pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Maximum drawdown
        rolling_max = equity_df['portfolio_value'].expanding().max()
        drawdown = (equity_df['portfolio_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        total_trades = len(self.trades)
        if total_trades > 0:
            # Calculate trade returns (simplified)
            winning_trades = sum(1 for trade in self.trades if trade['action'] == 'SELL')  # Simplified
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            average_trade_return = total_return / total_trades if total_trades > 0 else 0
        else:
            winning_trades = 0
            win_rate = 0
            average_trade_return = 0
        
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=total_trades - winning_trades,
            average_trade_return=average_trade_return,
            trades=self.trades,
            equity_curve=equity_df,
            performance_metrics={
                'volatility': returns.std() * np.sqrt(252) if len(returns) > 0 else 0,
                'best_day': returns.max() if len(returns) > 0 else 0,
                'worst_day': returns.min() if len(returns) > 0 else 0,
                'positive_days': (returns > 0).sum() if len(returns) > 0 else 0,
                'negative_days': (returns < 0).sum() if len(returns) > 0 else 0
            }
        )
