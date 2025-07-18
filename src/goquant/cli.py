"""
Command-line interface for GoQuant Sentiment Trader.
"""

import asyncio
import argparse
import sys
from typing import Optional

from .main import SentimentTrader
from .config import get_config
from .utils.logger import setup_logger


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="GoQuant Sentiment Trader - Real-time sentiment analysis and trading signals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  goquant-sentiment run                    # Start the full system
  goquant-sentiment collect                # Run data collection only
  goquant-sentiment analyze                # Run sentiment analysis only
  goquant-sentiment signals                # Generate trading signals only
  goquant-sentiment dashboard              # Start web dashboard only
  goquant-sentiment backtest --start 2023-01-01 --end 2023-12-31
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        help="Path to configuration file",
        default=".env"
    )
    
    parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run command (default - start full system)
    run_parser = subparsers.add_parser("run", help="Start the full sentiment trading system")
    run_parser.add_argument(
        "--daemon", "-d",
        action="store_true",
        help="Run as daemon process"
    )
    
    # Collect command
    collect_parser = subparsers.add_parser("collect", help="Run data collection only")
    collect_parser.add_argument(
        "--symbols",
        nargs="+",
        help="Specific symbols to collect data for"
    )
    collect_parser.add_argument(
        "--once",
        action="store_true",
        help="Run collection once and exit"
    )
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run sentiment analysis only")
    analyze_parser.add_argument(
        "--text",
        help="Analyze sentiment of specific text"
    )
    
    # Signals command
    signals_parser = subparsers.add_parser("signals", help="Generate trading signals")
    signals_parser.add_argument(
        "--symbol",
        help="Generate signal for specific symbol"
    )
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Start web dashboard")
    dashboard_parser.add_argument(
        "--host",
        default="localhost",
        help="Dashboard host"
    )
    dashboard_parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Dashboard port"
    )
    
    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtesting")
    backtest_parser.add_argument(
        "--start",
        required=True,
        help="Start date (YYYY-MM-DD)"
    )
    backtest_parser.add_argument(
        "--end",
        required=True,
        help="End date (YYYY-MM-DD)"
    )
    backtest_parser.add_argument(
        "--symbols",
        nargs="+",
        help="Symbols to backtest"
    )
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show system status")
    
    return parser


async def run_full_system(args) -> None:
    """Run the full sentiment trading system."""
    trader = SentimentTrader()
    
    try:
        await trader.start()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"System error: {e}")
    finally:
        await trader.stop()


async def run_data_collection(args) -> None:
    """Run data collection only."""
    from .data.collector import DataCollector
    
    config = get_config()
    collector = DataCollector(config)
    
    try:
        await collector.initialize()
        
        if args.once:
            # Run collection once
            if args.symbols:
                results = await collector.collect_for_symbols(args.symbols)
                print(f"Collection results: {results}")
            else:
                results = await collector.collect_all()
                print(f"Collection results: {results}")
        else:
            # Run continuous collection
            print("Starting continuous data collection...")
            while True:
                try:
                    if args.symbols:
                        results = await collector.collect_for_symbols(args.symbols)
                    else:
                        results = await collector.collect_all()
                    
                    print(f"Collected: {results}")
                    await asyncio.sleep(config.data_collection_interval)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Collection error: {e}")
                    await asyncio.sleep(10)
    
    except Exception as e:
        print(f"Failed to initialize data collector: {e}")
        sys.exit(1)


async def run_sentiment_analysis(args) -> None:
    """Run sentiment analysis."""
    from .sentiment.analyzer import SentimentAnalyzer
    
    config = get_config()
    analyzer = SentimentAnalyzer(config)
    
    try:
        await analyzer.initialize()
        
        if args.text:
            # Analyze specific text
            result = await analyzer.analyze_text(args.text)
            print(f"Sentiment Analysis Result:")
            print(f"  Text: {args.text}")
            print(f"  Score: {result.score:.3f}")
            print(f"  Label: {result.label}")
            print(f"  Confidence: {result.confidence:.3f}")
            if result.emotions:
                print(f"  Emotions: {result.emotions}")
        else:
            # Process pending data
            processed = await analyzer.process_pending_data()
            print(f"Processed {processed} records")
    
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        sys.exit(1)


async def run_signal_generation(args) -> None:
    """Run trading signal generation."""
    from .signals.generator import SignalGenerator
    from .database.manager import DatabaseManager
    
    config = get_config()
    generator = SignalGenerator(config)
    db_manager = DatabaseManager(config.database_url)
    
    try:
        await db_manager.initialize()
        generator.set_database_manager(db_manager)
        
        if args.symbol:
            # Generate signal for specific symbol
            signal = await generator._generate_signal_for_symbol(args.symbol)
            if signal:
                print(f"Generated signal for {args.symbol}:")
                print(f"  Type: {signal.signal_type.value}")
                print(f"  Strength: {signal.strength.value}")
                print(f"  Confidence: {signal.confidence:.3f}")
                print(f"  Sentiment Score: {signal.sentiment_score:.3f}")
                if signal.fear_greed_index:
                    print(f"  Fear/Greed Index: {signal.fear_greed_index:.1f}")
            else:
                print(f"No signal generated for {args.symbol}")
        else:
            # Generate signals for all symbols
            signals = await generator.generate_signals()
            print(f"Generated {len(signals)} signals:")
            for signal in signals:
                print(f"  {signal.symbol}: {signal.signal_type.value} ({signal.strength.value})")
    
    except Exception as e:
        print(f"Signal generation error: {e}")
        sys.exit(1)
    finally:
        await db_manager.close()


def run_dashboard(args) -> None:
    """Start the web dashboard."""
    trader = SentimentTrader()
    
    # Override dashboard settings if provided
    if args.host:
        trader.config.dashboard_host = args.host
    if args.port:
        trader.config.dashboard_port = args.port
    
    print(f"Starting dashboard at http://{trader.config.dashboard_host}:{trader.config.dashboard_port}")
    trader.start_dashboard()


async def run_backtest(args) -> None:
    """Run backtesting."""
    from datetime import datetime
    from .backtesting.engine import BacktestingEngine
    from .sentiment.analyzer import SentimentAnalyzer
    from .signals.generator import SignalGenerator
    from .database.manager import DatabaseManager

    config = get_config()

    try:
        # Parse dates
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
        end_date = datetime.strptime(args.end, '%Y-%m-%d')

        # Use provided symbols or default
        symbols = args.symbols if args.symbols else config.default_assets

        print(f"Running backtest from {start_date.date()} to {end_date.date()}")
        print(f"Symbols: {symbols}")
        print(f"Initial capital: ${config.backtest_initial_capital:,.2f}")

        # Initialize components
        db_manager = DatabaseManager(config.database_url)
        await db_manager.initialize()

        sentiment_analyzer = SentimentAnalyzer(config)
        await sentiment_analyzer.initialize()
        sentiment_analyzer.set_database_manager(db_manager)

        signal_generator = SignalGenerator(config)
        signal_generator.set_database_manager(db_manager)

        # Initialize backtesting engine
        backtest_engine = BacktestingEngine(config)
        backtest_engine.set_components(db_manager, sentiment_analyzer, signal_generator)

        # Run backtest
        result = await backtest_engine.run_backtest(start_date, end_date, symbols)

        # Display results
        print("\n" + "="*50)
        print("BACKTESTING RESULTS")
        print("="*50)
        print(f"Period: {result.start_date.date()} to {result.end_date.date()}")
        print(f"Initial Capital: ${result.initial_capital:,.2f}")
        print(f"Final Capital: ${result.final_capital:,.2f}")
        print(f"Total Return: {result.total_return:.2%}")
        print(f"Annualized Return: {result.annualized_return:.2%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Maximum Drawdown: {result.max_drawdown:.2%}")
        print(f"Win Rate: {result.win_rate:.2%}")
        print(f"Total Trades: {result.total_trades}")
        print(f"Winning Trades: {result.winning_trades}")
        print(f"Losing Trades: {result.losing_trades}")
        print(f"Average Trade Return: {result.average_trade_return:.2%}")

        # Performance metrics
        print("\nPerformance Metrics:")
        for metric, value in result.performance_metrics.items():
            if isinstance(value, float):
                print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
            else:
                print(f"  {metric.replace('_', ' ').title()}: {value}")

        # Save results to file
        import json
        results_file = f"backtest_results_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"

        # Convert result to JSON-serializable format
        result_dict = {
            'start_date': result.start_date.isoformat(),
            'end_date': result.end_date.isoformat(),
            'initial_capital': result.initial_capital,
            'final_capital': result.final_capital,
            'total_return': result.total_return,
            'annualized_return': result.annualized_return,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'win_rate': result.win_rate,
            'total_trades': result.total_trades,
            'winning_trades': result.winning_trades,
            'losing_trades': result.losing_trades,
            'average_trade_return': result.average_trade_return,
            'trades': result.trades,
            'performance_metrics': result.performance_metrics
        }

        with open(results_file, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)

        print(f"\nResults saved to: {results_file}")

    except Exception as e:
        print(f"Backtesting failed: {e}")
        sys.exit(1)
    finally:
        if 'db_manager' in locals():
            await db_manager.close()


async def show_status(args) -> None:
    """Show system status."""
    print("System Status:")
    print("  Status: Not implemented yet")
    # TODO: Implement status checking
    pass


async def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logger("goquant", args.log_level)
    
    # Default command is 'run'
    if not args.command:
        args.command = "run"
    
    # Route to appropriate handler
    try:
        if args.command == "run":
            await run_full_system(args)
        elif args.command == "collect":
            await run_data_collection(args)
        elif args.command == "analyze":
            await run_sentiment_analysis(args)
        elif args.command == "signals":
            await run_signal_generation(args)
        elif args.command == "dashboard":
            run_dashboard(args)
        elif args.command == "backtest":
            await run_backtest(args)
        elif args.command == "status":
            await show_status(args)
        else:
            parser.print_help()
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
