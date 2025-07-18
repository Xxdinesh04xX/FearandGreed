"""
Simple backtesting for GoQuant Sentiment Trader.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any


class SimpleBacktester:
    """Simple backtesting engine."""
    
    def __init__(self, db_path='simple_goquant.db', initial_capital=100000):
        self.db_path = db_path
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # symbol -> shares
        self.trades = []
        self.portfolio_values = []
    
    def get_historical_data(self, days_back=30):
        """Get historical sentiment and price data."""
        conn = sqlite3.connect(self.db_path)
        
        # Get sentiment data
        sentiment_query = """
            SELECT symbol, sentiment_score, confidence, processed_at
            FROM sentiment_data
            WHERE processed_at > datetime('now', '-{} days')
            ORDER BY processed_at
        """.format(days_back)
        
        sentiment_df = pd.read_sql_query(sentiment_query, conn)
        
        # Get trading signals
        signals_query = """
            SELECT symbol, signal_type, strength, confidence, sentiment_score, generated_at
            FROM trading_signals
            WHERE generated_at > datetime('now', '-{} days')
            ORDER BY generated_at
        """.format(days_back)
        
        signals_df = pd.read_sql_query(signals_query, conn)
        
        conn.close()
        return sentiment_df, signals_df
    
    def simulate_price_from_sentiment(self, symbol, sentiment_score, base_price=100):
        """Simulate price movement based on sentiment (for demo)."""
        # Simple simulation: positive sentiment -> price increase
        price_change = sentiment_score * 0.05  # 5% max change per sentiment point
        noise = np.random.normal(0, 0.01)  # 1% random noise
        
        new_price = base_price * (1 + price_change + noise)
        return max(new_price, 1.0)  # Minimum price of $1
    
    def execute_trade(self, symbol, signal_type, price, confidence):
        """Execute a trade based on signal."""
        position_size = self.calculate_position_size(confidence)
        
        if signal_type == 'BUY':
            shares_to_buy = int((self.current_capital * position_size) / price)
            cost = shares_to_buy * price
            
            if cost <= self.current_capital:
                self.current_capital -= cost
                self.positions[symbol] = self.positions.get(symbol, 0) + shares_to_buy
                
                trade = {
                    'symbol': symbol,
                    'action': 'BUY',
                    'shares': shares_to_buy,
                    'price': price,
                    'cost': cost,
                    'timestamp': datetime.now()
                }
                self.trades.append(trade)
                return True
        
        elif signal_type == 'SELL':
            shares_owned = self.positions.get(symbol, 0)
            if shares_owned > 0:
                shares_to_sell = min(shares_owned, int(shares_owned * position_size))
                proceeds = shares_to_sell * price
                
                self.current_capital += proceeds
                self.positions[symbol] -= shares_to_sell
                
                trade = {
                    'symbol': symbol,
                    'action': 'SELL',
                    'shares': shares_to_sell,
                    'price': price,
                    'proceeds': proceeds,
                    'timestamp': datetime.now()
                }
                self.trades.append(trade)
                return True
        
        return False
    
    def calculate_position_size(self, confidence):
        """Calculate position size based on confidence."""
        # Higher confidence = larger position (max 20% of capital)
        return min(0.2, confidence * 0.25)
    
    def calculate_portfolio_value(self, current_prices):
        """Calculate total portfolio value."""
        portfolio_value = self.current_capital
        
        for symbol, shares in self.positions.items():
            if shares > 0 and symbol in current_prices:
                portfolio_value += shares * current_prices[symbol]
        
        return portfolio_value
    
    def run_backtest(self):
        """Run the backtest simulation."""
        print("🔄 Running Backtest Simulation")
        print("=" * 35)
        
        sentiment_df, signals_df = self.get_historical_data()
        
        if signals_df.empty:
            print("❌ No historical signals found. Run the trader first to generate data.")
            return None
        
        print(f"📊 Found {len(signals_df)} historical signals")
        print(f"💰 Starting capital: ${self.initial_capital:,.2f}")
        
        # Simulate prices for each symbol
        symbols = signals_df['symbol'].unique()
        current_prices = {symbol: 100.0 for symbol in symbols}  # Start at $100
        
        # Process signals chronologically
        signals_df['generated_at'] = pd.to_datetime(signals_df['generated_at'])
        signals_df = signals_df.sort_values('generated_at')
        
        for _, signal in signals_df.iterrows():
            symbol = signal['symbol']
            signal_type = signal['signal_type']
            confidence = signal['confidence']
            sentiment_score = signal['sentiment_score']
            
            # Update simulated price based on sentiment
            current_prices[symbol] = self.simulate_price_from_sentiment(
                symbol, sentiment_score, current_prices[symbol]
            )
            
            # Execute trade
            executed = self.execute_trade(symbol, signal_type, current_prices[symbol], confidence)
            
            if executed:
                portfolio_value = self.calculate_portfolio_value(current_prices)
                self.portfolio_values.append({
                    'timestamp': signal['generated_at'],
                    'portfolio_value': portfolio_value,
                    'cash': self.current_capital
                })
        
        # Calculate final results
        final_portfolio_value = self.calculate_portfolio_value(current_prices)
        total_return = (final_portfolio_value - self.initial_capital) / self.initial_capital
        
        results = {
            'initial_capital': self.initial_capital,
            'final_portfolio_value': final_portfolio_value,
            'total_return': total_return,
            'total_trades': len(self.trades),
            'final_positions': self.positions,
            'trades': self.trades,
            'portfolio_values': self.portfolio_values
        }
        
        self.print_results(results)
        return results
    
    def print_results(self, results):
        """Print backtest results."""
        print("\n" + "=" * 35)
        print("📈 BACKTEST RESULTS")
        print("=" * 35)
        
        print(f"Initial Capital:     ${results['initial_capital']:,.2f}")
        print(f"Final Portfolio:     ${results['final_portfolio_value']:,.2f}")
        print(f"Total Return:        {results['total_return']:.2%}")
        print(f"Total Trades:        {results['total_trades']}")
        
        if results['total_return'] > 0:
            print("🎉 Profitable strategy!")
        else:
            print("📉 Strategy needs improvement")
        
        print(f"\nFinal Positions:")
        for symbol, shares in results['final_positions'].items():
            if shares > 0:
                print(f"  {symbol}: {shares} shares")
        
        print(f"\nRecent Trades:")
        for trade in results['trades'][-5:]:  # Show last 5 trades
            action_emoji = "🟢" if trade['action'] == 'BUY' else "🔴"
            print(f"  {action_emoji} {trade['action']} {trade['shares']} {trade['symbol']} @ ${trade['price']:.2f}")


def main():
    """Run backtest."""
    backtester = SimpleBacktester()
    results = backtester.run_backtest()
    
    if results:
        print("\n💡 Tips for improvement:")
        print("1. Collect more data over longer periods")
        print("2. Refine sentiment analysis accuracy")
        print("3. Adjust position sizing strategy")
        print("4. Add stop-loss and take-profit rules")


if __name__ == "__main__":
    main()
