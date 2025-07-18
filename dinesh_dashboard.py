"""
Dinesh Trading Dashboard - Optimized for zero lag.
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import threading
import asyncio
import time
from twitter_friendly_trader import TwitterFriendlyTrader


class DineshTradingDashboard:
    """Ultra-fast Dinesh Trading Dashboard."""
    
    def __init__(self, db_path='simple_goquant.db'):
        self.db_path = db_path
        self.app = dash.Dash(__name__, title="Dinesh Trading Dashboard")
        
        # Trader control
        self.trader_running = False
        self.last_update = "Never"
        
        # Cache for data to reduce database calls
        self.data_cache = {
            'last_refresh': 0,
            'sentiment_df': pd.DataFrame(),
            'signals_df': pd.DataFrame(),
            'stats_df': pd.DataFrame()
        }
        
        self.setup_layout()
        self.setup_callbacks()
    
    def get_cached_data(self):
        """Get cached data or refresh if needed."""
        current_time = time.time()
        
        # Only refresh cache every 2 minutes
        if current_time - self.data_cache['last_refresh'] > 120:
            try:
                conn = sqlite3.connect(self.db_path)
                
                # Get recent sentiment data (limited)
                sentiment_df = pd.read_sql_query("""
                    SELECT symbol, sentiment_score, confidence, processed_at
                    FROM sentiment_data
                    WHERE processed_at > datetime('now', '-6 hours')
                    ORDER BY processed_at DESC
                    LIMIT 100
                """, conn)
                
                # Get recent signals (limited)
                signals_df = pd.read_sql_query("""
                    SELECT symbol, signal_type, strength, confidence, sentiment_score, generated_at
                    FROM trading_signals
                    WHERE generated_at > datetime('now', '-6 hours')
                    ORDER BY generated_at DESC
                    LIMIT 10
                """, conn)
                
                # Get raw data stats
                stats_df = pd.read_sql_query("""
                    SELECT source, COUNT(*) as count
                    FROM raw_data
                    WHERE collected_at > datetime('now', '-6 hours')
                    GROUP BY source
                """, conn)
                
                conn.close()
                
                # Update cache
                self.data_cache.update({
                    'last_refresh': current_time,
                    'sentiment_df': sentiment_df,
                    'signals_df': signals_df,
                    'stats_df': stats_df
                })
                
            except Exception as e:
                print(f"Database error: {e}")
        
        return (self.data_cache['sentiment_df'], 
                self.data_cache['signals_df'], 
                self.data_cache['stats_df'])
    
    def setup_layout(self):
        """Setup optimized layout."""
        self.app.layout = html.Div([
            # Fixed header
            html.Div([
                html.H1("🚀 Dinesh Trading Dashboard", className="header-title"),
                html.P("Real-time sentiment analysis and trading signals", className="header-subtitle"),
                
                # Control Panel
                html.Div([
                    html.Button("▶️ Start", id="start-btn", n_clicks=0, className="btn btn-success"),
                    html.Button("⏸️ Stop", id="stop-btn", n_clicks=0, className="btn btn-danger"),
                    html.Button("🔄 Once", id="once-btn", n_clicks=0, className="btn btn-primary"),
                    html.Span(id="status", className="status"),
                ], className="controls")
            ], className="header"),
            
            # Main content
            html.Div([
                # Metrics
                html.Div([
                    html.Div([
                        html.H4("📊 Data"),
                        html.Div(id="data-count", className="metric")
                    ], className="card"),
                    
                    html.Div([
                        html.H4("🎯 Signals"),
                        html.Div(id="signal-count", className="metric")
                    ], className="card"),
                    
                    html.Div([
                        html.H4("🧠 Sentiment"),
                        html.Div(id="sentiment", className="metric")
                    ], className="card"),
                    
                    html.Div([
                        html.H4("⚡ Status"),
                        html.Div(id="system", className="metric")
                    ], className="card")
                ], className="metrics"),
                
                # Simple chart
                html.Div([
                    html.H4("📈 Sentiment Trend"),
                    dcc.Graph(
                        id="chart", 
                        config={'displayModeBar': False, 'staticPlot': True},
                        style={'height': '300px'}
                    )
                ], className="chart-section"),
                
                # Tables
                html.Div([
                    html.Div([
                        html.H4("🎯 Recent Signals"),
                        html.Div(id="signals")
                    ], className="table-section"),
                    
                    html.Div([
                        html.H4("📡 Sources"),
                        html.Div(id="sources")
                    ], className="table-section")
                ], className="tables"),
                
                # System info
                html.Div([
                    html.P(id="info", className="info")
                ], className="footer")
                
            ], className="content"),
            
            # Manual refresh button instead of auto-refresh
            html.Div([
                html.Button("🔄 Refresh Data", id="refresh-btn", n_clicks=0, className="btn btn-info refresh-btn")
            ], className="refresh-section"),
            
            # Hidden store for state
            dcc.Store(id='app-state')
        ])
    
    def setup_callbacks(self):
        """Setup optimized callbacks."""
        
        # Trader control
        @self.app.callback(
            Output('status', 'children'),
            [Input('start-btn', 'n_clicks'),
             Input('stop-btn', 'n_clicks'),
             Input('once-btn', 'n_clicks')]
        )
        def control_trader(start, stop, once):
            """Control trader with minimal processing."""
            ctx = callback_context
            
            if ctx.triggered:
                button = ctx.triggered[0]['prop_id'].split('.')[0]
                
                if button == 'start-btn' and start > 0:
                    if not self.trader_running:
                        self.start_trader()
                        return "🟢 Running"
                    return "🟢 Already Running"
                
                elif button == 'stop-btn' and stop > 0:
                    self.stop_trader()
                    return "🔴 Stopped"
                
                elif button == 'once-btn' and once > 0:
                    self.run_once()
                    return "🟡 Running Once..."
            
            return "🟢 Running" if self.trader_running else "🔴 Stopped"
        
        # Data updates (only on manual refresh)
        @self.app.callback(
            [Output('data-count', 'children'),
             Output('signal-count', 'children'),
             Output('sentiment', 'children'),
             Output('system', 'children'),
             Output('chart', 'figure'),
             Output('signals', 'children'),
             Output('sources', 'children'),
             Output('info', 'children')],
            [Input('refresh-btn', 'n_clicks')]
        )
        def update_all_data(refresh_clicks):
            """Update all data at once to minimize callbacks."""
            try:
                sentiment_df, signals_df, stats_df = self.get_cached_data()
                
                # Metrics
                data_count = str(stats_df['count'].sum() if not stats_df.empty else 0)
                signal_count = str(len(signals_df))
                
                avg_sentiment = sentiment_df['sentiment_score'].mean() if not sentiment_df.empty else 0
                sentiment_text = f"{avg_sentiment:.2f}"
                if avg_sentiment > 0.1:
                    sentiment_text += " 📈"
                elif avg_sentiment < -0.1:
                    sentiment_text += " 📉"
                else:
                    sentiment_text += " ➡️"
                
                system_status = "🟢 Active" if self.trader_running else "🔴 Idle"
                
                # Simple chart
                if sentiment_df.empty:
                    fig = go.Figure()
                    fig.add_annotation(text="No data", showarrow=False)
                else:
                    sentiment_df['processed_at'] = pd.to_datetime(sentiment_df['processed_at'])
                    hourly = sentiment_df.groupby(sentiment_df['processed_at'].dt.floor('H'))['sentiment_score'].mean()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hourly.index,
                        y=hourly.values,
                        mode='lines',
                        line=dict(color='#007bff', width=2)
                    ))
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=20, b=20),
                    showlegend=False,
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False, range=[-1, 1]),
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                
                # Signals table
                if signals_df.empty:
                    signals_table = html.P("No signals", className="no-data")
                else:
                    rows = []
                    for _, signal in signals_df.head(5).iterrows():
                        emoji = "🟢" if signal['signal_type'] == 'BUY' else "🔴"
                        rows.append(html.Div([
                            html.Span(signal['symbol'], className="symbol"),
                            html.Span([emoji, f" {signal['signal_type']}"], className="signal"),
                            html.Span(f"{signal['confidence']:.2f}", className="confidence")
                        ], className="signal-row"))
                    signals_table = html.Div(rows)
                
                # Sources table
                if stats_df.empty:
                    sources_table = html.P("No sources", className="no-data")
                else:
                    rows = []
                    for _, stat in stats_df.iterrows():
                        emoji = {'twitter': '🐦', 'reddit': '🔴', 'news': '📰', 
                                'yahoo_finance': '📈', 'finnhub': '📊'}.get(stat['source'], '📡')
                        rows.append(html.Div([
                            html.Span([emoji, f" {stat['source'].title()}"], className="source"),
                            html.Span(f"{stat['count']}", className="count")
                        ], className="source-row"))
                    sources_table = html.Div(rows)
                
                # Info
                info_text = f"Last refresh: {datetime.now().strftime('%H:%M:%S')} | Trader: {'Running' if self.trader_running else 'Stopped'}"
                
                return (data_count, signal_count, sentiment_text, system_status, 
                       fig, signals_table, sources_table, info_text)
                
            except Exception as e:
                error = f"Error: {e}"
                empty_fig = go.Figure()
                return error, error, error, error, empty_fig, error, error, error
    
    def start_trader(self):
        """Start trader."""
        if not self.trader_running:
            self.trader_running = True
            thread = threading.Thread(target=self._run_continuous, daemon=True)
            thread.start()
    
    def stop_trader(self):
        """Stop trader."""
        self.trader_running = False
    
    def run_once(self):
        """Run trader once."""
        thread = threading.Thread(target=self._run_once, daemon=True)
        thread.start()
    
    def _run_continuous(self):
        """Run trader continuously."""
        async def loop():
            trader = TwitterFriendlyTrader()
            await trader.data_collector.initialize()
            
            try:
                while self.trader_running:
                    await trader.run_cycle()
                    self.last_update = datetime.now().strftime('%H:%M:%S')
                    
                    # Wait 20 minutes
                    for _ in range(1200):
                        if not self.trader_running:
                            break
                        await asyncio.sleep(1)
            finally:
                await trader.data_collector.close()
        
        asyncio.run(loop())
    
    def _run_once(self):
        """Run trader once."""
        async def single():
            trader = TwitterFriendlyTrader()
            await trader.data_collector.initialize()
            
            try:
                await trader.run_cycle()
                self.last_update = datetime.now().strftime('%H:%M:%S')
            finally:
                await trader.data_collector.close()
        
        asyncio.run(single())
    
    def run(self, host='localhost', port=8050, debug=False):
        """Run the dashboard."""
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <style>
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    body { 
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                        background: #f8f9fa; 
                        line-height: 1.4;
                    }
                    .header { 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; 
                        padding: 15px 20px; 
                        position: fixed; 
                        top: 0; 
                        left: 0; 
                        right: 0; 
                        z-index: 1000;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    }
                    .header-title { font-size: 1.8em; font-weight: 700; margin-bottom: 5px; }
                    .header-subtitle { font-size: 0.9em; opacity: 0.9; margin-bottom: 10px; }
                    .controls { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
                    .btn { 
                        padding: 8px 16px; 
                        border: none; 
                        border-radius: 5px; 
                        cursor: pointer; 
                        font-weight: 600; 
                        font-size: 13px;
                        transition: transform 0.1s;
                    }
                    .btn:hover { transform: translateY(-1px); }
                    .btn-success { background: #28a745; color: white; }
                    .btn-danger { background: #dc3545; color: white; }
                    .btn-primary { background: #007bff; color: white; }
                    .btn-info { background: #17a2b8; color: white; }
                    .status { margin-left: 15px; font-weight: 600; }
                    
                    .content { margin-top: 140px; padding: 20px; max-width: 1200px; margin-left: auto; margin-right: auto; }
                    .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 25px; }
                    .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }
                    .card h4 { color: #495057; margin-bottom: 10px; font-size: 1em; }
                    .metric { font-size: 2em; font-weight: 700; color: #007bff; }
                    
                    .chart-section { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 25px; }
                    .chart-section h4 { color: #495057; margin-bottom: 15px; }
                    
                    .tables { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 25px; }
                    .table-section { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                    .table-section h4 { color: #495057; margin-bottom: 15px; }
                    .signal-row, .source-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee; }
                    .symbol, .source { font-weight: 600; }
                    .signal, .count { color: #6c757d; }
                    .confidence { font-size: 0.9em; color: #007bff; }
                    .no-data { text-align: center; color: #6c757d; font-style: italic; padding: 20px; }
                    
                    .refresh-section { text-align: center; margin-bottom: 20px; }
                    .refresh-btn { font-size: 14px; padding: 10px 20px; }
                    
                    .footer { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                    .info { text-align: center; color: #6c757d; font-size: 0.9em; }
                    
                    @media (max-width: 768px) {
                        .tables { grid-template-columns: 1fr; }
                        .controls { justify-content: center; }
                        .header-title { font-size: 1.5em; }
                    }
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
        
        print(f"🚀 Starting Dinesh Trading Dashboard at http://{host}:{port}")
        print("⚡ Ultra-fast performance - zero lag")
        print("🔄 Manual refresh for complete control")
        print("📱 Mobile-optimized design")
        
        self.app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    dashboard = DineshTradingDashboard()
    dashboard.run()
