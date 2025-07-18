"""
Unified GoQuant Dashboard with integrated trader controls.
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import threading
import asyncio
import time
from twitter_friendly_trader import TwitterFriendlyTrader


class UnifiedGoQuantDashboard:
    """Unified dashboard with integrated trader."""
    
    def __init__(self, db_path='simple_goquant.db'):
        self.db_path = db_path
        self.app = dash.Dash(__name__, title="GoQuant Unified Dashboard")
        
        # Trader control
        self.trader = None
        self.trader_thread = None
        self.trader_running = False
        self.trader_status = "Stopped"
        self.last_update = "Never"
        
        self.setup_layout()
        self.setup_callbacks()
    
    def get_data_from_db(self):
        """Get data from database."""
        try:
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
            
        except Exception as e:
            print(f"Database error: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    def setup_layout(self):
        """Setup unified dashboard layout."""
        self.app.layout = html.Div([
            # Header with controls
            html.Div([
                html.H1("🚀 GoQuant Unified Trading Dashboard", className="header-title"),
                html.P("Real-time sentiment analysis, trading signals, and system control", className="header-subtitle"),
                
                # Control Panel
                html.Div([
                    html.Div([
                        html.Button("▶️ Start Trader", id="start-btn", n_clicks=0, className="btn btn-success"),
                        html.Button("⏸️ Stop Trader", id="stop-btn", n_clicks=0, className="btn btn-danger"),
                        html.Button("🔄 Run Once", id="once-btn", n_clicks=0, className="btn btn-primary"),
                    ], className="button-group"),
                    
                    html.Div([
                        html.Div(id="trader-status", className="status-indicator"),
                        html.Div(id="last-update", className="update-time"),
                    ], className="status-group")
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
                        html.H3("🔄 System Status"),
                        html.Div(id="system-status-metric", className="metric-value"),
                        html.P("Trader operational status", className="metric-description")
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
                
                # Third row - Tables and logs
                html.Div([
                    html.Div([
                        html.H3("🎯 Recent Trading Signals"),
                        html.Div(id="signals-table")
                    ], className="table-container"),
                    
                    html.Div([
                        html.H3("📡 Data Sources & System Log"),
                        html.Div(id="sources-table"),
                        html.Hr(),
                        html.Div(id="system-log", className="system-log")
                    ], className="table-container")
                ], className="tables-row")
            ], className="main-content"),
            
            # Auto-refresh (longer interval to reduce scroll jumping)
            dcc.Interval(
                id='interval-component',
                interval=30*1000,  # Update every 30 seconds
                n_intervals=0
            ),
            
            # Hidden div to store trader status
            html.Div(id='trader-state', style={'display': 'none'})
        ])
    
    def setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            [Output('trader-state', 'children'),
             Output('trader-status', 'children'),
             Output('last-update', 'children')],
            [Input('start-btn', 'n_clicks'),
             Input('stop-btn', 'n_clicks'),
             Input('once-btn', 'n_clicks'),
             Input('interval-component', 'n_intervals')],
            [State('trader-state', 'children')]
        )
        def control_trader(start_clicks, stop_clicks, once_clicks, n_intervals, current_state):
            """Control trader start/stop."""
            ctx = callback_context
            
            if ctx.triggered:
                button_id = ctx.triggered[0]['prop_id'].split('.')[0]
                
                if button_id == 'start-btn' and start_clicks > 0:
                    if not self.trader_running:
                        self.start_trader_continuous()
                        status = "🟢 Running Continuously"
                        update_time = f"Started at {datetime.now().strftime('%H:%M:%S')}"
                    else:
                        status = "🟢 Already Running"
                        update_time = self.last_update
                
                elif button_id == 'stop-btn' and stop_clicks > 0:
                    if self.trader_running:
                        self.stop_trader()
                        status = "🔴 Stopped"
                        update_time = f"Stopped at {datetime.now().strftime('%H:%M:%S')}"
                    else:
                        status = "🔴 Already Stopped"
                        update_time = self.last_update
                
                elif button_id == 'once-btn' and once_clicks > 0:
                    self.run_trader_once()
                    status = "🟡 Running Once..."
                    update_time = f"Single run at {datetime.now().strftime('%H:%M:%S')}"
                
                else:
                    status = "🟢 Running" if self.trader_running else "🔴 Stopped"
                    update_time = self.last_update
            else:
                status = "🟢 Running" if self.trader_running else "🔴 Stopped"
                update_time = self.last_update
            
            return "", status, update_time
        
        @self.app.callback(
            [Output('data-points-metric', 'children'),
             Output('signals-metric', 'children'),
             Output('sentiment-metric', 'children'),
             Output('system-status-metric', 'children')],
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
                
                # System status
                if self.trader_running:
                    system_status = "🟢 Active"
                else:
                    system_status = "🔴 Idle"
                
                return str(total_data_points), str(active_signals), sentiment_text, system_status
                
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
            [Output('sources-table', 'children'),
             Output('system-log', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_sources_and_log(n):
            """Update data sources table and system log."""
            try:
                _, _, stats_df = self.get_data_from_db()
                
                # Sources table
                if stats_df.empty:
                    sources_table = html.P("No data collected yet")
                else:
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
                    
                    sources_table = html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Source"),
                                html.Th("Data Points (24h)"),
                                html.Th("Status")
                            ])
                        ]),
                        html.Tbody(table_rows)
                    ], className="sources-table")
                
                # System log
                log_entries = [
                    f"🕐 {datetime.now().strftime('%H:%M:%S')} - Dashboard refreshed",
                    f"📊 Trader Status: {'Running' if self.trader_running else 'Stopped'}",
                    f"🔄 Last Update: {self.last_update}",
                    f"💾 Database: {self.db_path}",
                ]
                
                log_div = html.Div([
                    html.P(entry, className="log-entry") for entry in log_entries
                ])
                
                return sources_table, log_div
                
            except Exception as e:
                return html.P(f"Error: {e}"), html.P(f"Log error: {e}")
    
    def start_trader_continuous(self):
        """Start trader in continuous mode."""
        if not self.trader_running:
            self.trader_running = True
            self.trader_thread = threading.Thread(target=self._run_trader_continuous, daemon=True)
            self.trader_thread.start()
            print("🚀 Trader started in continuous mode")
    
    def stop_trader(self):
        """Stop the trader."""
        self.trader_running = False
        print("🛑 Trader stopped")
    
    def run_trader_once(self):
        """Run trader once."""
        thread = threading.Thread(target=self._run_trader_once, daemon=True)
        thread.start()
        print("🔄 Running trader once")
    
    def _run_trader_continuous(self):
        """Run trader continuously in background thread."""
        async def continuous_loop():
            trader = TwitterFriendlyTrader()
            await trader.data_collector.initialize()
            
            try:
                while self.trader_running:
                    print("🔄 Running trader cycle...")
                    await trader.run_cycle()
                    self.last_update = datetime.now().strftime('%H:%M:%S')
                    
                    # Wait 20 minutes between cycles
                    for _ in range(1200):  # 20 minutes = 1200 seconds
                        if not self.trader_running:
                            break
                        await asyncio.sleep(1)
            finally:
                await trader.data_collector.close()
        
        asyncio.run(continuous_loop())
    
    def _run_trader_once(self):
        """Run trader once in background thread."""
        async def single_run():
            trader = TwitterFriendlyTrader()
            await trader.data_collector.initialize()
            
            try:
                print("🔄 Running single trader cycle...")
                await trader.run_cycle()
                self.last_update = datetime.now().strftime('%H:%M:%S')
            finally:
                await trader.data_collector.close()
        
        asyncio.run(single_run())
    
    def run(self, host='localhost', port=8050, debug=False):
        """Run the unified dashboard."""
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
                    body {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        margin: 0;
                        background: #f5f7fa;
                        scroll-behavior: smooth;
                        overflow-x: hidden;
                    }
                    html { scroll-behavior: smooth; }
                    .header {
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 20px;
                        text-align: center;
                        position: sticky;
                        top: 0;
                        z-index: 1000;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                    .header-title { margin: 0; font-size: 2.5em; font-weight: bold; }
                    .header-subtitle { margin: 10px 0; font-size: 1.2em; opacity: 0.9; }
                    .control-panel { margin-top: 15px; display: flex; justify-content: center; align-items: center; gap: 30px; }
                    .button-group { display: flex; gap: 10px; }
                    .status-group { display: flex; flex-direction: column; align-items: center; }
                    .btn { padding: 12px 24px; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; font-size: 14px; }
                    .btn-success { background: #28a745; color: white; }
                    .btn-danger { background: #dc3545; color: white; }
                    .btn-primary { background: #007bff; color: white; }
                    .status-indicator { font-size: 1.2em; font-weight: bold; margin-bottom: 5px; }
                    .update-time { font-size: 0.9em; opacity: 0.8; }
                    .main-content {
                        padding: 20px;
                        max-width: 1400px;
                        margin: 0 auto;
                        min-height: calc(100vh - 200px);
                    }
                    .metrics-row, .charts-row, .tables-row {
                        display: flex;
                        gap: 20px;
                        margin-bottom: 30px;
                        min-height: fit-content;
                    }
                    .metric-card, .chart-container, .table-container {
                        background: white;
                        padding: 20px;
                        border-radius: 10px;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                        min-height: fit-content;
                        position: relative;
                    }
                    .metric-card { flex: 1; text-align: center; }
                    .metric-value { font-size: 2.5em; font-weight: bold; color: #667eea; margin: 10px 0; }
                    .metric-description { color: #6c757d; margin: 0; }
                    .chart-container { flex: 1; }
                    .table-container { flex: 1; }
                    .signals-table, .sources-table { width: 100%; border-collapse: collapse; }
                    .signals-table th, .signals-table td, .sources-table th, .sources-table td { padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }
                    .signals-table th, .sources-table th { background: #f8f9fa; font-weight: bold; }
                    .system-log { background: #f8f9fa; padding: 15px; border-radius: 5px; max-height: 200px; overflow-y: auto; }
                    .log-entry { margin: 5px 0; font-family: monospace; font-size: 12px; }

                    /* Prevent scroll jumping during updates */
                    .dash-table-container { position: relative; }
                    .plotly-graph-div { position: relative; }

                    /* Smooth transitions */
                    * { transition: none !important; }
                </style>
                <script>
                    // Preserve scroll position during updates
                    let lastScrollPosition = 0;

                    function preserveScrollPosition() {
                        lastScrollPosition = window.pageYOffset || document.documentElement.scrollTop;
                    }

                    function restoreScrollPosition() {
                        if (lastScrollPosition > 0) {
                            window.scrollTo(0, lastScrollPosition);
                        }
                    }

                    // Save scroll position before updates
                    window.addEventListener('beforeunload', preserveScrollPosition);

                    // Restore scroll position after updates
                    document.addEventListener('DOMContentLoaded', function() {
                        setTimeout(restoreScrollPosition, 100);
                    });

                    // Monitor for Dash updates and preserve scroll
                    if (window.dash_clientside) {
                        window.dash_clientside.no_update = window.dash_clientside.no_update || {};
                    }
                </script>
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
        
        print(f"🌐 Starting GoQuant Unified Dashboard at http://{host}:{port}")
        print("🎯 Features:")
        print("   ▶️ Start/Stop trader controls")
        print("   🔄 Run single cycles")
        print("   📊 Real-time monitoring")
        print("   🐦 Twitter integration with rate limiting")
        print("   📈 Live charts and signals")
        
        self.app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    dashboard = UnifiedGoQuantDashboard()
    dashboard.run()
