"""
Dash web application for real-time sentiment monitoring and trading signals.
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd

from ..utils.logger import get_logger


def create_dashboard_app(sentiment_trader=None):
    """
    Create and configure the Dash web application.
    
    Args:
        sentiment_trader: SentimentTrader instance for data access
        
    Returns:
        Dash application instance
    """
    logger = get_logger(__name__)
    
    # Initialize Dash app
    app = dash.Dash(__name__, title="GoQuant Sentiment Trader")
    
    # Define the layout
    app.layout = html.Div([
        # Header
        html.Div([
            html.H1("GoQuant Sentiment Trader", className="header-title"),
            html.P("Real-time sentiment analysis and trading signals", className="header-subtitle")
        ], className="header"),
        
        # Main content
        html.Div([
            # Top row - Key metrics
            html.Div([
                html.Div([
                    html.H3("Overall Market Sentiment"),
                    html.Div(id="overall-sentiment", className="metric-value"),
                    html.P("Weighted average across all sources", className="metric-description")
                ], className="metric-card"),
                
                html.Div([
                    html.H3("Fear & Greed Index"),
                    html.Div(id="fear-greed-index", className="metric-value"),
                    html.P("Market emotion indicator", className="metric-description")
                ], className="metric-card"),
                
                html.Div([
                    html.H3("Active Signals"),
                    html.Div(id="active-signals-count", className="metric-value"),
                    html.P("Current trading recommendations", className="metric-description")
                ], className="metric-card"),
                
                html.Div([
                    html.H3("Data Points (24h)"),
                    html.Div(id="data-points-count", className="metric-value"),
                    html.P("Processed sentiment data", className="metric-description")
                ], className="metric-card")
            ], className="metrics-row"),
            
            # Second row - Charts
            html.Div([
                html.Div([
                    html.H3("Sentiment Timeline"),
                    dcc.Graph(id="sentiment-timeline")
                ], className="chart-container"),
                
                html.Div([
                    html.H3("Symbol Sentiment Breakdown"),
                    dcc.Graph(id="symbol-sentiment")
                ], className="chart-container")
            ], className="charts-row"),
            
            # Third row - Trading signals and data sources
            html.Div([
                html.Div([
                    html.H3("Recent Trading Signals"),
                    html.Div(id="trading-signals-table")
                ], className="table-container"),
                
                html.Div([
                    html.H3("Data Source Status"),
                    html.Div(id="data-sources-status")
                ], className="status-container")
            ], className="bottom-row")
        ], className="main-content"),
        
        # Auto-refresh component
        dcc.Interval(
            id='interval-component',
            interval=30*1000,  # Update every 30 seconds
            n_intervals=0
        )
    ])
    
    # Callbacks for updating data
    @app.callback(
        [Output('overall-sentiment', 'children'),
         Output('fear-greed-index', 'children'),
         Output('active-signals-count', 'children'),
         Output('data-points-count', 'children')],
        [Input('interval-component', 'n_intervals')]
    )
    def update_metrics(n):
        """Update key metrics display."""
        try:
            if sentiment_trader and sentiment_trader.db_manager:
                # Get real data from the system
                import asyncio

                # Run async operations in sync context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    # Get portfolio signals
                    portfolio_signals = loop.run_until_complete(
                        sentiment_trader.signal_generator.calculate_portfolio_signals()
                    )

                    # Get active signals count
                    active_signals = loop.run_until_complete(
                        sentiment_trader.db_manager.get_active_signals()
                    )

                    # Format metrics
                    overall_sentiment = f"{portfolio_signals.get('overall_sentiment', 0):.3f}"
                    if portfolio_signals.get('overall_sentiment', 0) > 0:
                        overall_sentiment += " (Positive)"
                    elif portfolio_signals.get('overall_sentiment', 0) < 0:
                        overall_sentiment += " (Negative)"
                    else:
                        overall_sentiment += " (Neutral)"

                    fear_greed_score = portfolio_signals.get('fear_greed_index', 50)
                    if fear_greed_score > 60:
                        fear_greed_label = "Greed"
                    elif fear_greed_score < 40:
                        fear_greed_label = "Fear"
                    else:
                        fear_greed_label = "Neutral"
                    fear_greed = f"{fear_greed_score:.0f} ({fear_greed_label})"

                    active_signals_count = str(len(active_signals))
                    data_points = "1,247"  # This would come from actual data count

                    return overall_sentiment, fear_greed, active_signals_count, data_points

                finally:
                    loop.close()
            else:
                # Fallback to mock data
                overall_sentiment = "0.15 (Positive)"
                fear_greed = "65 (Greed)"
                active_signals = "3"
                data_points = "1,247"

                return overall_sentiment, fear_greed, active_signals, data_points

        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
            return "Error", "Error", "Error", "Error"
    
    @app.callback(
        Output('sentiment-timeline', 'figure'),
        [Input('interval-component', 'n_intervals')]
    )
    def update_sentiment_timeline(n):
        """Update sentiment timeline chart."""
        try:
            # Mock data - replace with actual data
            dates = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                                end=datetime.now(), freq='H')
            sentiment_scores = [0.1 + 0.3 * (i % 5 - 2) / 2 for i in range(len(dates))]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=sentiment_scores,
                mode='lines+markers',
                name='Sentiment Score',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.update_layout(
                title="24-Hour Sentiment Timeline",
                xaxis_title="Time",
                yaxis_title="Sentiment Score",
                yaxis=dict(range=[-1, 1]),
                height=400,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error updating sentiment timeline: {e}")
            return go.Figure()
    
    @app.callback(
        Output('symbol-sentiment', 'figure'),
        [Input('interval-component', 'n_intervals')]
    )
    def update_symbol_sentiment(n):
        """Update symbol sentiment breakdown chart."""
        try:
            # Mock data - replace with actual data
            symbols = ['BTC', 'ETH', 'AAPL', 'TSLA', 'SPY']
            sentiment_scores = [0.2, 0.1, -0.1, 0.3, 0.05]
            
            colors = ['green' if score > 0 else 'red' for score in sentiment_scores]
            
            fig = go.Figure(data=[
                go.Bar(x=symbols, y=sentiment_scores, marker_color=colors)
            ])
            
            fig.update_layout(
                title="Current Sentiment by Symbol",
                xaxis_title="Symbol",
                yaxis_title="Sentiment Score",
                yaxis=dict(range=[-1, 1]),
                height=400,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error updating symbol sentiment: {e}")
            return go.Figure()
    
    @app.callback(
        Output('trading-signals-table', 'children'),
        [Input('interval-component', 'n_intervals')]
    )
    def update_trading_signals(n):
        """Update trading signals table."""
        try:
            # Mock data - replace with actual data
            signals_data = [
                {"Symbol": "BTC", "Signal": "BUY", "Strength": "STRONG", "Confidence": "85%", "Time": "10:30 AM"},
                {"Symbol": "AAPL", "Signal": "SELL", "Strength": "MODERATE", "Confidence": "72%", "Time": "09:45 AM"},
                {"Symbol": "ETH", "Signal": "BUY", "Strength": "WEAK", "Confidence": "65%", "Time": "08:15 AM"}
            ]
            
            table_rows = []
            for signal in signals_data:
                row = html.Tr([
                    html.Td(signal["Symbol"]),
                    html.Td(signal["Signal"], className=f"signal-{signal['Signal'].lower()}"),
                    html.Td(signal["Strength"]),
                    html.Td(signal["Confidence"]),
                    html.Td(signal["Time"])
                ])
                table_rows.append(row)
            
            table = html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Symbol"),
                        html.Th("Signal"),
                        html.Th("Strength"),
                        html.Th("Confidence"),
                        html.Th("Time")
                    ])
                ]),
                html.Tbody(table_rows)
            ], className="signals-table")
            
            return table
            
        except Exception as e:
            logger.error(f"Error updating trading signals: {e}")
            return html.Div("Error loading signals")
    
    @app.callback(
        Output('data-sources-status', 'children'),
        [Input('interval-component', 'n_intervals')]
    )
    def update_data_sources_status(n):
        """Update data sources status."""
        try:
            # Mock data - replace with actual data
            sources_status = [
                {"Source": "Twitter", "Status": "Active", "Last Update": "30s ago", "Records": "156"},
                {"Source": "Reddit", "Status": "Active", "Last Update": "45s ago", "Records": "89"},
                {"Source": "News", "Status": "Active", "Last Update": "2m ago", "Records": "23"},
                {"Source": "Financial Data", "Status": "Active", "Last Update": "1m ago", "Records": "12"}
            ]
            
            status_items = []
            for source in sources_status:
                status_item = html.Div([
                    html.Div([
                        html.Span(source["Source"], className="source-name"),
                        html.Span(source["Status"], className=f"status-{source['Status'].lower()}")
                    ], className="source-header"),
                    html.Div([
                        html.Span(f"Last: {source['Last Update']}", className="source-detail"),
                        html.Span(f"Records: {source['Records']}", className="source-detail")
                    ], className="source-details")
                ], className="source-status-item")
                status_items.append(status_item)
            
            return html.Div(status_items, className="sources-status-container")
            
        except Exception as e:
            logger.error(f"Error updating data sources status: {e}")
            return html.Div("Error loading status")
    
    # Add basic CSS styling
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f5f5f5; }
                .header { background-color: #2c3e50; color: white; padding: 20px; text-align: center; }
                .header-title { margin: 0; font-size: 2.5em; }
                .header-subtitle { margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.8; }
                .main-content { padding: 20px; max-width: 1400px; margin: 0 auto; }
                .metrics-row { display: flex; gap: 20px; margin-bottom: 30px; }
                .metric-card { flex: 1; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .metric-value { font-size: 2em; font-weight: bold; color: #2c3e50; margin: 10px 0; }
                .metric-description { color: #7f8c8d; margin: 0; }
                .charts-row { display: flex; gap: 20px; margin-bottom: 30px; }
                .chart-container { flex: 1; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .bottom-row { display: flex; gap: 20px; }
                .table-container, .status-container { flex: 1; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .signals-table { width: 100%; border-collapse: collapse; }
                .signals-table th, .signals-table td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
                .signals-table th { background-color: #f8f9fa; font-weight: bold; }
                .signal-buy { color: #27ae60; font-weight: bold; }
                .signal-sell { color: #e74c3c; font-weight: bold; }
                .source-status-item { margin-bottom: 15px; padding: 10px; border-left: 4px solid #3498db; background-color: #f8f9fa; }
                .source-header { display: flex; justify-content: space-between; margin-bottom: 5px; }
                .source-name { font-weight: bold; }
                .status-active { color: #27ae60; font-weight: bold; }
                .source-details { display: flex; gap: 20px; font-size: 0.9em; color: #7f8c8d; }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    # Add health check endpoint
    @app.server.route('/health')
    def health_check():
        """Health check endpoint for monitoring."""
        try:
            # Basic health check
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '0.1.0'
            }

            if sentiment_trader:
                # Check component health
                if hasattr(sentiment_trader, 'sentiment_analyzer') and sentiment_trader.sentiment_analyzer._initialized:
                    health_status['sentiment_analyzer'] = 'healthy'
                else:
                    health_status['sentiment_analyzer'] = 'not_initialized'

                if hasattr(sentiment_trader, 'db_manager') and sentiment_trader.db_manager:
                    health_status['database'] = 'healthy'
                else:
                    health_status['database'] = 'not_connected'

            return health_status, 200

        except Exception as e:
            return {'status': 'error', 'error': str(e)}, 500

    return app
