import sys
sys.path.insert(0, "./.python_packages/lib/python3.10/site-packages")
import dash
from dash import dcc, html, Input, Output, callback_context, State, clientside_callback
import pandas as pd
import requests
import pyodbc
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc
from diskcache import Cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from dash.exceptions import PreventUpdate
from dash import dash_table
from botbuilder.schema import Activity
import redis
import os
import logging
import io
import asyncio
import threading  # For lock
# Configure logging
logging.basicBasic(level=logging.INFO)
logger = logging.getLogger(__name__)
# Initialize app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, '/assets/style.css'])
server = app.server
# Cache setup: Azure Redis if REDIS_URL set, else local diskcache
REDIS_URL = os.getenv('REDIS_URL')
if REDIS_URL:
    cache = redis.Redis.from_url(REDIS_URL)
else:
    cache = Cache("cache") # Local fallback
# Environment variables for Azure
API_TOKEN = os.getenv('API_TOKEN', "5b67e106-173f-4281-a83f-87b2bdc3b1f1")
API_PASSWORD = os.getenv('API_PASSWORD', "Welcome1")
API_SITENAME = os.getenv('API_SITENAME', "fivestar")
API_USERID = os.getenv('API_USERID', "jeff.thompson")
SQL_SERVER = os.getenv('SQL_SERVER', "SQL-03")
SQL_DATABASE = os.getenv('SQL_DATABASE', "TECHSYS")
SQL_USERNAME = os.getenv('SQL_USERNAME')
SQL_PASSWORD = os.getenv('SQL_PASSWORD')
# Bot service env vars
MICROSOFT_APP_ID = os.getenv('MICROSOFT_APP_ID')
MICROSOFT_APP_PASSWORD = os.getenv('MICROSOFT_APP_PASSWORD')
# Lazy loading globals
lock = threading.Lock()
df = None
dates, start_date, end_date = get_date_range()  # This is fast, keep module-level
location_options = []  # Placeholder, loaded later
initial_table_data = []  # Placeholder
initial_refresh_text = "Loading data..."  # Placeholder
initial_alert_rows = [html.Tr([html.Td("Loading alerts...", colSpan=1, style={'padding': '8px', 'border': '1px solid #dee2e6', 'textAlign': 'center', 'fontFamily': 'Inter', 'fontSize': '14px'})])]

def load_data_if_needed(force_refresh=False):
    global df
    with lock:
        if df is None or force_refresh:
            df = fetch_data(force_refresh=force_refresh)
    return df

# ... (keep get_location_codes, get_employee_names, get_date_range, fetch_location_data, fetch_data as-is)

app.layout = dbc.Container(fluid=True, children=[
    html.Div([
        html.H1('Timeclock Dashboard (Five Guys USA)', style={'textAlign': 'center', 'color': '#0056b3', 'fontSize': '30px', 'fontFamily': 'Poppins', 'fontWeight': '700', 'marginBottom': '10px', 'marginTop': '20px'}),
        html.Div(f"Date Range: {start_date} to {end_date}", style={'textAlign': 'center', 'fontSize': '16px', 'color': '#6c757d', 'marginBottom': '10px', 'fontWeight': '400', 'fontFamily': 'Inter'}),
        html.Div(id='refresh-time', style={'textAlign': 'center', 'fontSize': '14px', 'color': '#6c757d', 'marginBottom': '10px', 'fontWeight': '400', 'fontFamily': 'Inter'}),
        html.Div(id='utc-refresh-time', children=initial_refresh_text, style={'display': 'none'}),
        html.Div([
            dbc.Button('Refresh Data', id='refresh-button', n_clicks=0, disabled=False, className='btn-success', style={'marginRight': '8px'}),
            dbc.Button('Export to Excel', id='export-button', n_clicks=0, className='btn-primary')
        ], style={'display': 'flex', 'justifyContent': 'center', 'marginTop': '10px', 'marginBottom': '20px'})
    ]),
    dbc.Card(
        dbc.CardBody([
            html.H3('Late ClockOut Employees', style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px', 'fontFamily': 'Poppins', 'fontWeight': '600', 'fontSize': '24px'}),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div([
                            dcc.Dropdown(
                                id='location-filter',
                                options=[],  # Loaded dynamically
                                value=None,
                                placeholder="Select Location",
                                searchable=False
                            )
                        ], className='dropdown'),
                        width=4,
                        className='mb-3'
                    ),
                    dbc.Col(
                        html.Div([
                            dcc.Input(
                                id='search-input',
                                type='text',
                                placeholder='Enter employee name or number',
                                className='filter-box'
                            )
                        ], className='search-bar'),
                        width=4,
                        className='mb-3'
                    ),
                    dbc.Col(width=4)
                ],
                style={'marginBottom': '20px'},
            ),
            dash_table.DataTable(
                id='late-clockout-table',
                columns=[
                    {'name': 'Location Name', 'id': 'location', 'sortable': True},
                    {'name': 'Employee Number', 'id': 'employeeNumber', 'sortable': True},
                    {'name': 'First Name', 'id': 'first_name', 'sortable': True},
                    {'name': 'Last Name', 'id': 'last_name', 'sortable': True},
                    {'name': 'Labor Date', 'id': 'laborDate', 'sortable': True},
                    {'name': 'Clock Out Time', 'id': 'clockOut', 'sortable': True}
                ],
                data=[],  # Loaded dynamically
                page_action='native',
                page_size=20,
                style_table={'overflowX': 'auto'},
                style_header={
                    'backgroundColor': '#007bff',
                    'color': 'white',
                    'fontWeight': '500',
                    'fontFamily': 'Inter',
                    'textAlign': 'left',
                    'cursor': 'pointer',
                    'userSelect': 'none'
                },
                style_cell={
                    'border': '1px solid #dee2e6',
                    'padding': '8px',
                    'fontFamily': 'Inter',
                    'textAlign': 'left',
                    'fontSize': '14px'
                },
                style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#f9f9f9'}],
                sort_action='custom',
                sort_mode='single',
                sort_by=[]
            ),
            html.H3('Alerts', style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px', 'marginTop': '30px', 'fontFamily': 'Poppins', 'fontWeight': '600', 'fontSize': '24px'}),
            html.Table(id='alerts-table', children=initial_alert_rows, style={'width': '100%', 'border': '1px solid #dee2e6', 'borderRadius': '5px', 'overflow': 'hidden'}),
            dcc.Download(id='download-excel'),
            html.Div(id='dummy', style={'display': 'none'}),
            html.Div(id='init-load-trigger', style={'display': 'none'})  # Hidden trigger for init load
        ]),
        style={'boxShadow': '0 4px 8px rgba(0,0,0,0.1)', 'borderRadius': '10px', 'backgroundColor': 'white', 'margin': '20px auto', 'maxWidth': '80%'}
    ),
    dcc.Interval(id='refresh-interval', interval=15*60*1000, n_intervals=0, disabled=True)
])

# New callback for initial load (triggers on app start)
@app.callback(
    [
        Output('late-clockout-table', 'data'),
        Output('location-filter', 'options'),
        Output('alerts-table', 'children'),
        Output('utc-refresh-time', 'children')
    ],
    Input('init-load-trigger', 'children')  # Dummy input to trigger once
)
def initial_load(_):
    load_data_if_needed()  # Load data here
    if 'error' in df.columns:
        return [], [], [html.Tr(html.Td("Error fetching data: " + df['error'].iloc[0], style={'padding': '8px', 'border': '1px solid #dee2e6', 'textAlign': 'center', 'fontFamily': 'Inter', 'fontSize': '14px'}))], "Error occurred"
    
    filtered_df = df.copy()
    filtered_df['laborDate_dt'] = pd.to_datetime(filtered_df['laborDate'], errors='coerce')
    filtered_df = filtered_df.sort_values(['laborDate_dt', 'clockOut_dt'], ascending=[False, False])
    filtered_df.drop(columns=['laborDate_dt'], inplace=True)
    filtered_df['clockOut'] = filtered_df['clockOut_dt'].dt.strftime('%I:%M %p')
    filtered_df['laborDate'] = pd.to_datetime(filtered_df['laborDate'], errors='coerce').dt.strftime('%m/%d/%Y')
    table_data = filtered_df[['location', 'employeeNumber', 'first_name', 'last_name', 'laborDate', 'clockOut']].to_dict('records')
 
    location_options = [{'label': loc, 'value': loc} for loc in sorted(df['location'].unique())]
    
    alerts = []
    location_counts = df.groupby('location').size()
    high_locations = location_counts[location_counts > 5].index
    for loc in high_locations:
        alerts.append(f"High number of late clockouts at {loc}!")
    employee_counts = df.groupby(['employeeNumber', 'first_name', 'last_name', 'location']).size()
    high_employees = employee_counts[employee_counts > 3]
    for (emp_num, first, last, loc) in high_employees.index:
        alerts.append(f"{first} {last} at {loc} has repeated late clock outs")
 
    if alerts:
        alert_rows = [html.Tr([html.Th('Alert Message', style={'backgroundColor': '#dc3545', 'color': 'white', 'padding': '8px', 'border': '1px solid #dee2e6', 'fontFamily': 'Inter', 'textAlign': 'left', 'fontSize': '14px'})])]
        for alert in alerts:
            alert_rows.append(html.Tr([html.Td(alert, style={'backgroundColor': '#fff3cd', 'padding': '8px', 'border': '1px solid #dee2e6', 'fontFamily': 'Inter', 'textAlign': 'left', 'fontSize': '14px'})]))
    else:
        alert_rows = [html.Tr([html.Td("No alerts", colSpan=1, style={'padding': '8px', 'border': '1px solid #dee2e6', 'textAlign': 'center', 'fontFamily': 'Inter', 'fontSize': '14px'})])]
 
    refresh_text = f"Last refreshed: {df['refresh_time'].iloc[0] if 'refresh_time' in df.columns else 'Unknown'} | {len(df)} late clockOut events across {len(dates)} days"
 
    return table_data, location_options, alert_rows, refresh_text

# ... (keep clientside_callbacks as-is)

# Update dashboard callback (use load_data_if_needed)
@app.callback(
    [
        Output('late-clockout-table', 'data'),
        Output('utc-refresh-time', 'children'),
        Output('refresh-button', 'disabled'),
        Output('refresh-interval', 'disabled'),
        Output('alerts-table', 'children'),
        Output('download-excel', 'data')
    ],
    [
        Input('refresh-button', 'n_clicks'),
        Input('location-filter', 'value'),
        Input('search-input', 'value'),
        Input('refresh-interval', 'n_intervals'),
        Input('export-button', 'n_clicks'),
        Input('late-clockout-table', 'sort_by')
    ]
)
def update_dashboard(n_clicks, selected_location, search_value, n_intervals, export_n_clicks, sort_by):
    global df
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
 
    if triggered_id in ['refresh-button', 'refresh-interval'] and (n_clicks > 0 or n_intervals > 0):
        df = load_data_if_needed(force_refresh=True)
    else:
        load_data_if_needed()  # Ensure loaded if not already
 
    # ... (rest of the callback unchanged, using df as before)

# Bot class (load data lazily in on_turn)
class ClockoutBot:
    async def on_turn(self, turn_context: TurnContext):
        if turn_context.activity.type == ActivityTypes.message:
            query = turn_context.activity.text.lower()
            if "forgot to clock out yesterday" in query:
                load_data_if_needed()  # Lazy load
                yesterday = (datetime.now() - timedelta(days=1)).strftime('%m/%d/%Y')
                filtered_df = df[df['laborDate'] == yesterday]
              
                if filtered_df.empty:
                    response = "No one forgot to clock out yesterday."
                else:
                    md_table = filtered_df[['location', 'employeeNumber', 'first_name', 'last_name', 'clockOut']].to_markdown(index=False)
                    response = f"Employees who forgot to clock out yesterday:\n\n{md_table}"
              
                await turn_context.send_activity(MessageFactory.text(response))
            else:
                await turn_context.send_activity("I can help with late clockouts—try 'who forgot to clock out yesterday'.")
BOT = ClockoutBot()
SETTINGS = BotFrameworkAdapterSettings(
    MICROSOFT_APP_ID,
    MICROSOFT_APP_PASSWORD,
    channel_auth_tenant=os.getenv('MICROSOFT_APP_TENANT_ID')
)
ADAPTER = BotFrameworkAdapter(SETTINGS)
# Bot route (/api/messages)
@app.server.route('/api/messages', methods=['POST'])
def messages():
    logger.info("Received request to /api/messages")
    content_type = request.headers.get('Content-Type', '')
    if 'application/json' in content_type.lower():
        body = request.json
    else:
        logger.error("Unsupported Media Type: " + content_type)
        return 'Unsupported Media Type', 415
  
    try:
        activity = Activity.deserialize(body)
        auth_header = request.headers.get('Authorization', '')
        asyncio.run(ADAPTER.process_activity(activity, auth_header, BOT.on_turn))
        logger.info("Processed activity")
        return '', 201
    except Exception as e:
        logger.error(f"Bot error: {str(e)}")
        return str(e), 500
if __name__ == '__main__':
    app.run_server(debug=False)
