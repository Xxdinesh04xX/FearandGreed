"""
Fixed Dinesh Dashboard with stable scrolling.
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context, clientside_callback, ClientsideFunction
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import threading
import asyncio
import time
from twitter_friendly_trader import TwitterFriendlyTrader


class FixedDineshDashboard:
    """Fixed dashboard with stable scrolling."""
    
    def __init__(self, db_path='simple_goquant.db'):
        self.db_path = db_path
        self.app = dash.Dash(__name__, title="Dinesh Trading Dashboard")
        
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
        """Setup fixed dashboard layout."""
        self.app.layout = html.Div([
            # Fixed header
            html.Div([
                html.H1("🚀 Dinesh Trading Dashboard", className="header-title"),
                html.P("Real-time sentiment analysis and trading signals", className="header-subtitle"),
                
                # Control Panel
                html.Div([
                    html.Button("▶️ Start Trader", id="start-btn", n_clicks=0, className="btn btn-success"),
                    html.Button("⏸️ Stop Trader", id="stop-btn", n_clicks=0, className="btn btn-danger"),
                    html.Button("🔄 Run Once", id="once-btn", n_clicks=0, className="btn btn-primary"),
                    html.Div(id="trader-status", className="status-indicator"),
                ], className="control-panel")
            ], className="header"),
            
            # Scrollable content
            html.Div([
                # Metrics row
                html.Div([
                    html.Div([
                        html.H4("📊 Data Points"),
                        html.Div(id="data-points", className="metric-value"),
                    ], className="metric-card"),
                    
                    html.Div([
                        html.H4("🎯 Signals"),
                        html.Div(id="signals-count", className="metric-value"),
                    ], className="metric-card"),
                    
                    html.Div([
                        html.H4("🧠 Sentiment"),
                        html.Div(id="avg-sentiment", className="metric-value"),
                    ], className="metric-card"),
                    
                    html.Div([
                        html.H4("🔄 Status"),
                        html.Div(id="system-status", className="metric-value"),
                    ], className="metric-card")
                ], className="metrics-row"),
                
                # Charts row
                html.Div([
                    html.Div([
                        html.H4("📈 Sentiment Timeline"),
                        dcc.Graph(id="sentiment-chart", config={'displayModeBar': False})
                    ], className="chart-card"),
                    
                    html.Div([
                        html.H4("📊 Symbol Sentiment"),
                        dcc.Graph(id="symbol-chart", config={'displayModeBar': False})
                    ], className="chart-card")
                ], className="charts-row"),
                
                # Tables row
                html.Div([
                    html.Div([
                        html.H4("🎯 Recent Signals"),
                        html.Div(id="signals-table")
                    ], className="table-card"),
                    
                    html.Div([
                        html.H4("📡 Data Sources"),
                        html.Div(id="sources-table")
                    ], className="table-card")
                ], className="tables-row"),
                
                # System info
                html.Div([
                    html.H4("📋 System Information"),
                    html.Div(id="system-info", className="info-content")
                ], className="info-card")
                
            ], className="content"),
            
            # Auto-refresh with much longer interval to reduce lag
            dcc.Interval(
                id='refresh-interval',
                interval=120*1000,  # Update every 2 minutes to reduce lag
                n_intervals=0
            ),
            
            # Store for scroll position
            dcc.Store(id='scroll-store'),
        ])
    
    def setup_callbacks(self):
        """Setup callbacks with scroll preservation."""
        
        # Trader control callback
        @self.app.callback(
            Output('trader-status', 'children'),
            [Input('start-btn', 'n_clicks'),
             Input('stop-btn', 'n_clicks'),
             Input('once-btn', 'n_clicks')]
        )
        def control_trader(start_clicks, stop_clicks, once_clicks):
            """Control trader."""
            ctx = callback_context
            
            if ctx.triggered:
                button_id = ctx.triggered[0]['prop_id'].split('.')[0]
                
                if button_id == 'start-btn' and start_clicks > 0:
                    if not self.trader_running:
                        self.start_trader_continuous()
                        return "🟢 Running Continuously"
                    else:
                        return "🟢 Already Running"
                
                elif button_id == 'stop-btn' and stop_clicks > 0:
                    if self.trader_running:
                        self.stop_trader()
                        return "🔴 Stopped"
                    else:
                        return "🔴 Already Stopped"
                
                elif button_id == 'once-btn' and once_clicks > 0:
                    self.run_trader_once()
                    return "🟡 Running Once..."
            
            return "🟢 Running" if self.trader_running else "🔴 Stopped"
        
        # Data update callbacks
        @self.app.callback(
            [Output('data-points', 'children'),
             Output('signals-count', 'children'),
             Output('avg-sentiment', 'children'),
             Output('system-status', 'children')],
            [Input('refresh-interval', 'n_intervals')]
        )
        def update_metrics(n):
            """Update metrics without scroll jumping."""
            try:
                sentiment_df, signals_df, stats_df = self.get_data_from_db()
                
                total_data = stats_df['count'].sum() if not stats_df.empty else 0
                signal_count = len(signals_df)
                avg_sentiment = sentiment_df['sentiment_score'].mean() if not sentiment_df.empty else 0
                
                sentiment_text = f"{avg_sentiment:.3f}"
                if avg_sentiment > 0.1:
                    sentiment_text += " 📈"
                elif avg_sentiment < -0.1:
                    sentiment_text += " 📉"
                else:
                    sentiment_text += " ➡️"
                
                status = "🟢 Active" if self.trader_running else "🔴 Idle"
                
                return str(total_data), str(signal_count), sentiment_text, status
                
            except Exception as e:
                return "Error", "Error", "Error", "Error"
        
        @self.app.callback(
            Output('sentiment-chart', 'figure'),
            [Input('refresh-interval', 'n_intervals')]
        )
        def update_sentiment_chart(n):
            """Update sentiment chart."""
            try:
                sentiment_df, _, _ = self.get_data_from_db()
                
                if sentiment_df.empty:
                    fig = go.Figure()
                    fig.add_annotation(text="No data available", showarrow=False)
                    fig.update_layout(height=300)
                    return fig
                
                sentiment_df['processed_at'] = pd.to_datetime(sentiment_df['processed_at'])
                sentiment_df['hour'] = sentiment_df['processed_at'].dt.floor('H')
                
                hourly_sentiment = sentiment_df.groupby('hour')['sentiment_score'].mean().reset_index()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hourly_sentiment['hour'],
                    y=hourly_sentiment['sentiment_score'],
                    mode='lines+markers',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=6)
                ))
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=40, r=40, t=40, b=40),
                    yaxis=dict(range=[-1, 1]),
                    showlegend=False,
                    transition_duration=0,  # Disable animations to reduce lag
                    uirevision='constant'   # Prevent chart resets
                )
                
                return fig
                
            except Exception as e:
                fig = go.Figure()
                fig.add_annotation(text=f"Error: {str(e)}", showarrow=False)
                fig.update_layout(height=300)
                return fig
        
        @self.app.callback(
            Output('symbol-chart', 'figure'),
            [Input('refresh-interval', 'n_intervals')]
        )
        def update_symbol_chart(n):
            """Update symbol chart."""
            try:
                sentiment_df, _, _ = self.get_data_from_db()
                
                if sentiment_df.empty:
                    fig = go.Figure()
                    fig.add_annotation(text="No data available", showarrow=False)
                    fig.update_layout(height=300)
                    return fig
                
                symbol_sentiment = sentiment_df.groupby('symbol')['sentiment_score'].mean().reset_index()
                symbol_sentiment = symbol_sentiment.sort_values('sentiment_score', ascending=True)
                
                colors = ['red' if score < 0 else 'green' for score in symbol_sentiment['sentiment_score']]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=symbol_sentiment['sentiment_score'],
                        y=symbol_sentiment['symbol'],
                        orientation='h',
                        marker_color=colors
                    )
                ])
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=40, r=40, t=40, b=40),
                    xaxis=dict(range=[-1, 1]),
                    showlegend=False,
                    transition_duration=0,  # Disable animations to reduce lag
                    uirevision='constant'   # Prevent chart resets
                )
                
                return fig
                
            except Exception as e:
                fig = go.Figure()
                fig.add_annotation(text=f"Error: {str(e)}", showarrow=False)
                fig.update_layout(height=300)
                return fig
        
        @self.app.callback(
            [Output('signals-table', 'children'),
             Output('sources-table', 'children'),
             Output('system-info', 'children')],
            [Input('refresh-interval', 'n_intervals')]
        )
        def update_tables_and_info(n):
            """Update tables and system info."""
            try:
                sentiment_df, signals_df, stats_df = self.get_data_from_db()
                
                # Signals table
                if signals_df.empty:
                    signals_table = html.P("No recent signals", className="no-data")
                else:
                    signal_rows = []
                    for _, signal in signals_df.head(10).iterrows():  # Limit to 10 rows
                        emoji = "🟢" if signal['signal_type'] == 'BUY' else "🔴" if signal['signal_type'] == 'SELL' else "🟡"
                        signal_rows.append(html.Tr([
                            html.Td(signal['symbol']),
                            html.Td([emoji, f" {signal['signal_type']}"]),
                            html.Td(signal['strength']),
                            html.Td(f"{signal['confidence']:.2f}"),
                            html.Td(pd.to_datetime(signal['generated_at']).strftime('%H:%M'))
                        ]))
                    
                    signals_table = html.Table([
                        html.Thead([html.Tr([
                            html.Th("Symbol"), html.Th("Signal"), html.Th("Strength"), 
                            html.Th("Confidence"), html.Th("Time")
                        ])]),
                        html.Tbody(signal_rows)
                    ], className="data-table")
                
                # Sources table
                if stats_df.empty:
                    sources_table = html.P("No data sources active", className="no-data")
                else:
                    source_rows = []
                    for _, stat in stats_df.iterrows():
                        emoji = {'twitter': '🐦', 'reddit': '🔴', 'news': '📰', 
                                'yahoo_finance': '📈', 'finnhub': '📊'}.get(stat['source'], '📡')
                        source_rows.append(html.Tr([
                            html.Td([emoji, f" {stat['source'].title()}"]),
                            html.Td(f"{stat['count']:,}"),
                            html.Td("🟢 Active")
                        ]))
                    
                    sources_table = html.Table([
                        html.Thead([html.Tr([
                            html.Th("Source"), html.Th("Count"), html.Th("Status")
                        ])]),
                        html.Tbody(source_rows)
                    ], className="data-table")
                
                # System info
                system_info = html.Div([
                    html.P(f"🕐 Last Update: {datetime.now().strftime('%H:%M:%S')}"),
                    html.P(f"📊 Trader: {'Running' if self.trader_running else 'Stopped'}"),
                    html.P(f"💾 Database: {self.db_path}"),
                    html.P(f"🔄 Auto-refresh: Every 2 minutes")
                ])
                
                return signals_table, sources_table, system_info
                
            except Exception as e:
                error_msg = html.P(f"Error: {e}", className="error")
                return error_msg, error_msg, error_msg
    
    def start_trader_continuous(self):
        """Start trader in continuous mode."""
        if not self.trader_running:
            self.trader_running = True
            self.trader_thread = threading.Thread(target=self._run_trader_continuous, daemon=True)
            self.trader_thread.start()
    
    def stop_trader(self):
        """Stop the trader."""
        self.trader_running = False
    
    def run_trader_once(self):
        """Run trader once."""
        thread = threading.Thread(target=self._run_trader_once, daemon=True)
        thread.start()
    
    def _run_trader_continuous(self):
        """Run trader continuously."""
        async def continuous_loop():
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
        
        asyncio.run(continuous_loop())
    
    def _run_trader_once(self):
        """Run trader once."""
        async def single_run():
            trader = TwitterFriendlyTrader()
            await trader.data_collector.initialize()
            
            try:
                await trader.run_cycle()
                self.last_update = datetime.now().strftime('%H:%M:%S')
            finally:
                await trader.data_collector.close()
        
        asyncio.run(single_run())
    
    def run(self, host='localhost', port=8050, debug=False):
        """Run the fixed dashboard."""
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <style>
                    * { box-sizing: border-box; margin: 0; padding: 0; }
                    body { 
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                        background: #f5f7fa; 
                        overflow-x: hidden;
                    }
                    .header { 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; 
                        padding: 20px; 
                        text-align: center; 
                        position: fixed;
                        top: 0;
                        left: 0;
                        right: 0;
                        z-index: 1000;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    }
                    .header-title { font-size: 2em; font-weight: bold; margin-bottom: 10px; }
                    .header-subtitle { font-size: 1em; opacity: 0.9; margin-bottom: 15px; }
                    .control-panel { display: flex; justify-content: center; align-items: center; gap: 15px; flex-wrap: wrap; }
                    .btn { 
                        padding: 10px 20px; 
                        border: none; 
                        border-radius: 6px; 
                        cursor: pointer; 
                        font-weight: bold; 
                        font-size: 14px;
                        transition: all 0.2s;
                    }
                    .btn:hover { transform: translateY(-1px); box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
                    .btn-success { background: #28a745; color: white; }
                    .btn-danger { background: #dc3545; color: white; }
                    .btn-primary { background: #007bff; color: white; }
                    .status-indicator { font-weight: bold; margin-left: 15px; }
                    
                    .content { 
                        margin-top: 180px; 
                        padding: 20px; 
                        max-width: 1400px; 
                        margin-left: auto; 
                        margin-right: auto; 
                    }
                    .metrics-row, .charts-row, .tables-row { 
                        display: grid; 
                        gap: 20px; 
                        margin-bottom: 30px; 
                    }
                    .metrics-row { grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }
                    .charts-row { grid-template-columns: 1fr 1fr; }
                    .tables-row { grid-template-columns: 1fr 1fr; }
                    
                    .metric-card, .chart-card, .table-card, .info-card { 
                        background: white; 
                        padding: 20px; 
                        border-radius: 10px; 
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                        border: 1px solid #e9ecef;
                    }
                    .metric-card h4, .chart-card h4, .table-card h4, .info-card h4 { 
                        margin-bottom: 15px; 
                        color: #495057; 
                        font-size: 1.1em;
                    }
                    .metric-value { 
                        font-size: 2em; 
                        font-weight: bold; 
                        color: #667eea; 
                        text-align: center;
                    }
                    .data-table { 
                        width: 100%; 
                        border-collapse: collapse; 
                        font-size: 14px;
                    }
                    .data-table th, .data-table td { 
                        padding: 10px; 
                        text-align: left; 
                        border-bottom: 1px solid #dee2e6; 
                    }
                    .data-table th { 
                        background: #f8f9fa; 
                        font-weight: 600; 
                        color: #495057;
                    }
                    .no-data, .error { 
                        text-align: center; 
                        color: #6c757d; 
                        font-style: italic; 
                        padding: 20px;
                    }
                    .error { color: #dc3545; }
                    .info-content p { 
                        margin-bottom: 8px; 
                        font-size: 14px; 
                        color: #495057;
                    }
                    
                    @media (max-width: 768px) {
                        .charts-row, .tables-row { grid-template-columns: 1fr; }
                        .header-title { font-size: 1.5em; }
                        .control-panel { flex-direction: column; gap: 10px; }
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
        
        print(f"🌐 Starting Dinesh Trading Dashboard at http://{host}:{port}")
        print("✅ Fixed scrolling and lag issues")
        print("✅ Stable layout with 2-minute refresh")
        print("✅ Optimized performance")
        print("✅ Responsive design")
        
        self.app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    dashboard = FixedDineshDashboard()
    dashboard.run()
