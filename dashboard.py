"""
Real-time web dashboard for GoQuant Sentiment Trader.
"""

import dash
from dash import dcc, html, Input, Output, callback_context
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import threading
import asyncio
from enhanced_trader import EnhancedTrader


class GoQuantDashboard:
    """Real-time dashboard for sentiment trading."""
    
    def __init__(self, db_path='simple_goquant.db'):
        self.db_path = db_path
        self.app = dash.Dash(__name__, title="GoQuant Sentiment Trader")
        self.setup_layout()
        self.setup_callbacks()
        
        # Background trader
        self.trader = None
        self.trader_running = False
    
    def get_data_from_db(self):
        """Get data from database."""
        conn = sqlite3.connect(self.db_path)
        
        # Get recent sentiment data
        sentiment_df = pd.read_sql_query("""
            SELECT symbol, sentiment_score, confidence, processed_at
            FROM sentiment_data
            WHERE processed_at > datetime('now', '-24 hours')
            ORDER BY processed_at DESC
        """, conn)
        
        # Get recent signals
        signals_df = pd.read_sql_query("""
            SELECT symbol, signal_type, strength, confidence, sentiment_score, generated_at
            FROM trading_signals
            WHERE generated_at > datetime('now', '-24 hours')
            ORDER BY generated_at DESC
            LIMIT 20
        """, conn)
        
        # Get raw data stats
        stats_df = pd.read_sql_query("""
            SELECT source, COUNT(*) as count
            FROM raw_data
            WHERE collected_at > datetime('now', '-24 hours')
            GROUP BY source
        """, conn)
        
        conn.close()
        return sentiment_df, signals_df, stats_df
    
    def setup_layout(self):
        """Setup dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("🚀 GoQuant Sentiment Trader", className="header-title"),
                html.P("Real-time sentiment analysis and trading signals", className="header-subtitle"),
                html.Div([
                    html.Button("▶️ Start Trader", id="start-btn", n_clicks=0, className="btn btn-success"),
                    html.Button("⏸️ Stop Trader", id="stop-btn", n_clicks=0, className="btn btn-danger"),
                    html.Div(id="trader-status", className="status-indicator")
                ], className="control-panel")
            ], className="header"),
            
            # Main content
            html.Div([
                # Top row - Key metrics
                html.Div([
                    html.Div([
                        html.H3("📊 Data Points (24h)"),
                        html.Div(id="data-points-metric", className="metric-value"),
                        html.P("Collected from all sources", className="metric-description")
                    ], className="metric-card"),
                    
                    html.Div([
                        html.H3("🎯 Active Signals"),
                        html.Div(id="signals-metric", className="metric-value"),
                        html.P("Current trading recommendations", className="metric-description")
                    ], className="metric-card"),
                    
                    html.Div([
                        html.H3("🧠 Avg Sentiment"),
                        html.Div(id="sentiment-metric", className="metric-value"),
                        html.P("Overall market sentiment", className="metric-description")
                    ], className="metric-card"),
                    
                    html.Div([
                        html.H3("🔄 Last Update"),
                        html.Div(id="update-time-metric", className="metric-value"),
                        html.P("System last refresh", className="metric-description")
                    ], className="metric-card")
                ], className="metrics-row"),
                
                # Second row - Charts
                html.Div([
                    html.Div([
                        html.H3("📈 Sentiment Timeline"),
                        dcc.Graph(id="sentiment-timeline")
                    ], className="chart-container"),
                    
                    html.Div([
                        html.H3("📊 Symbol Sentiment"),
                        dcc.Graph(id="symbol-sentiment")
                    ], className="chart-container")
                ], className="charts-row"),
                
                # Third row - Tables
                html.Div([
                    html.Div([
                        html.H3("🎯 Recent Trading Signals"),
                        html.Div(id="signals-table")
                    ], className="table-container"),
                    
                    html.Div([
                        html.H3("📡 Data Sources"),
                        html.Div(id="sources-table")
                    ], className="table-container")
                ], className="tables-row")
            ], className="main-content"),
            
            # Auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=30*1000,  # Update every 30 seconds
                n_intervals=0
            )
        ])
    
    def setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            [Output('data-points-metric', 'children'),
             Output('signals-metric', 'children'),
             Output('sentiment-metric', 'children'),
             Output('update-time-metric', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_metrics(n):
            """Update key metrics."""
            try:
                sentiment_df, signals_df, stats_df = self.get_data_from_db()
                
                # Data points
                total_data_points = stats_df['count'].sum() if not stats_df.empty else 0
                
                # Active signals
                active_signals = len(signals_df)
                
                # Average sentiment
                avg_sentiment = sentiment_df['sentiment_score'].mean() if not sentiment_df.empty else 0
                sentiment_text = f"{avg_sentiment:.3f}"
                if avg_sentiment > 0.1:
                    sentiment_text += " 📈"
                elif avg_sentiment < -0.1:
                    sentiment_text += " 📉"
                else:
                    sentiment_text += " ➡️"
                
                # Last update
                update_time = datetime.now().strftime("%H:%M:%S")
                
                return str(total_data_points), str(active_signals), sentiment_text, update_time
                
            except Exception as e:
                return "Error", "Error", "Error", "Error"
        
        @self.app.callback(
            Output('sentiment-timeline', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_sentiment_timeline(n):
            """Update sentiment timeline chart."""
            try:
                sentiment_df, _, _ = self.get_data_from_db()
                
                if sentiment_df.empty:
                    return go.Figure().add_annotation(text="No data available", showarrow=False)
                
                # Convert timestamp and group by hour
                sentiment_df['processed_at'] = pd.to_datetime(sentiment_df['processed_at'])
                sentiment_df['hour'] = sentiment_df['processed_at'].dt.floor('H')
                
                # Calculate hourly average sentiment
                hourly_sentiment = sentiment_df.groupby('hour')['sentiment_score'].mean().reset_index()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hourly_sentiment['hour'],
                    y=hourly_sentiment['sentiment_score'],
                    mode='lines+markers',
                    name='Sentiment Score',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title="24-Hour Sentiment Timeline",
                    xaxis_title="Time",
                    yaxis_title="Sentiment Score",
                    yaxis=dict(range=[-1, 1]),
                    height=400,
                    hovermode='x unified'
                )
                
                return fig
                
            except Exception as e:
                return go.Figure().add_annotation(text=f"Error: {str(e)}", showarrow=False)
        
        @self.app.callback(
            Output('symbol-sentiment', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_symbol_sentiment(n):
            """Update symbol sentiment chart."""
            try:
                sentiment_df, _, _ = self.get_data_from_db()
                
                if sentiment_df.empty:
                    return go.Figure().add_annotation(text="No data available", showarrow=False)
                
                # Calculate average sentiment by symbol
                symbol_sentiment = sentiment_df.groupby('symbol')['sentiment_score'].mean().reset_index()
                symbol_sentiment = symbol_sentiment.sort_values('sentiment_score', ascending=True)
                
                # Color based on sentiment
                colors = ['red' if score < 0 else 'green' for score in symbol_sentiment['sentiment_score']]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=symbol_sentiment['sentiment_score'],
                        y=symbol_sentiment['symbol'],
                        orientation='h',
                        marker_color=colors,
                        text=[f"{score:.3f}" for score in symbol_sentiment['sentiment_score']],
                        textposition='auto'
                    )
                ])
                
                fig.update_layout(
                    title="Current Sentiment by Symbol",
                    xaxis_title="Sentiment Score",
                    yaxis_title="Symbol",
                    xaxis=dict(range=[-1, 1]),
                    height=400
                )
                
                return fig
                
            except Exception as e:
                return go.Figure().add_annotation(text=f"Error: {str(e)}", showarrow=False)
        
        @self.app.callback(
            Output('signals-table', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_signals_table(n):
            """Update trading signals table."""
            try:
                _, signals_df, _ = self.get_data_from_db()
                
                if signals_df.empty:
                    return html.P("No recent signals")
                
                # Create table rows
                table_rows = []
                for _, signal in signals_df.iterrows():
                    signal_emoji = "🟢" if signal['signal_type'] == 'BUY' else "🔴" if signal['signal_type'] == 'SELL' else "🟡"
                    
                    row = html.Tr([
                        html.Td(signal['symbol']),
                        html.Td([signal_emoji, f" {signal['signal_type']}"]),
                        html.Td(signal['strength']),
                        html.Td(f"{signal['confidence']:.2f}"),
                        html.Td(f"{signal['sentiment_score']:+.3f}"),
                        html.Td(pd.to_datetime(signal['generated_at']).strftime('%H:%M'))
                    ])
                    table_rows.append(row)
                
                table = html.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("Symbol"),
                            html.Th("Signal"),
                            html.Th("Strength"),
                            html.Th("Confidence"),
                            html.Th("Sentiment"),
                            html.Th("Time")
                        ])
                    ]),
                    html.Tbody(table_rows)
                ], className="signals-table")
                
                return table
                
            except Exception as e:
                return html.P(f"Error loading signals: {e}")
        
        @self.app.callback(
            Output('sources-table', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_sources_table(n):
            """Update data sources table."""
            try:
                _, _, stats_df = self.get_data_from_db()
                
                if stats_df.empty:
                    return html.P("No data collected yet")
                
                # Create table rows
                table_rows = []
                for _, stat in stats_df.iterrows():
                    source_emoji = {
                        'twitter': '🐦',
                        'reddit': '🔴',
                        'news': '📰',
                        'yahoo_finance': '📈',
                        'finnhub': '📊'
                    }.get(stat['source'], '📡')
                    
                    row = html.Tr([
                        html.Td([source_emoji, f" {stat['source'].title()}"]),
                        html.Td(f"{stat['count']:,}"),
                        html.Td("🟢 Active")
                    ])
                    table_rows.append(row)
                
                table = html.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("Source"),
                            html.Th("Data Points (24h)"),
                            html.Th("Status")
                        ])
                    ]),
                    html.Tbody(table_rows)
                ], className="sources-table")
                
                return table
                
            except Exception as e:
                return html.P(f"Error loading sources: {e}")
    
    def run(self, host='localhost', port=8050, debug=False):
        """Run the dashboard."""
        # Add CSS styling
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <style>
                    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; background: #f5f7fa; }
                    .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; }
                    .header-title { margin: 0; font-size: 2.5em; font-weight: bold; }
                    .header-subtitle { margin: 10px 0; font-size: 1.2em; opacity: 0.9; }
                    .control-panel { margin-top: 15px; }
                    .btn { padding: 10px 20px; margin: 0 10px; border: none; border-radius: 5px; cursor: pointer; font-weight: bold; }
                    .btn-success { background: #28a745; color: white; }
                    .btn-danger { background: #dc3545; color: white; }
                    .main-content { padding: 20px; max-width: 1400px; margin: 0 auto; }
                    .metrics-row, .charts-row, .tables-row { display: flex; gap: 20px; margin-bottom: 30px; }
                    .metric-card, .chart-container, .table-container { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                    .metric-card { flex: 1; text-align: center; }
                    .metric-value { font-size: 2.5em; font-weight: bold; color: #667eea; margin: 10px 0; }
                    .metric-description { color: #6c757d; margin: 0; }
                    .chart-container { flex: 1; }
                    .table-container { flex: 1; }
                    .signals-table, .sources-table { width: 100%; border-collapse: collapse; }
                    .signals-table th, .signals-table td, .sources-table th, .sources-table td { padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }
                    .signals-table th, .sources-table th { background: #f8f9fa; font-weight: bold; }
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
        
        print(f"🌐 Starting GoQuant Dashboard at http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    dashboard = GoQuantDashboard()
    dashboard.run()
