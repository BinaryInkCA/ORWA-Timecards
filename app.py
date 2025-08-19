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
import redis
import os
import logging
import io
# Configure logging
logging.basicConfig(level=logging.INFO)
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
def get_location_codes():
    cache_key = "locations_data"
    try:
        if isinstance(cache, redis.Redis):
            cached_data = cache.get(cache_key)
            if cached_data:
                cached_data = cached_data.decode('utf-8')
        else:
            cached_data = cache.get(cache_key)
       
        if cached_data:
            logger.info("Using cached location codes")
            print(f"Retrieved {len(pd.read_json(StringIO(cached_data)))} locations from cache")
            return pd.read_json(StringIO(cached_data))
       
        conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={SQL_SERVER};"
            f"DATABASE={SQL_DATABASE};"
            f"UID={SQL_USERNAME};"
            f"PWD={SQL_PASSWORD};"
            "Connect Timeout=60;"
        )
        conn = pyodbc.connect(conn_str)
        query = "SELECT LOCATION_NAME, LOCATION_CODE FROM T_LOCATION WHERE LOCATION_ACTIVE = 'Y' AND (LOCATION_NAME LIKE 'FG - OR%' OR LOCATION_NAME LIKE 'FG - WA%')"
        df_locations = pd.read_sql(query, conn)
        conn.close()
       
        df_locations['brand'] = 'Five Guys USA'
       
        cached_json = df_locations[['LOCATION_CODE', 'LOCATION_NAME', 'brand']].to_json()
        if isinstance(cache, redis.Redis):
            cache.set(cache_key, cached_json.encode('utf-8'), ex=86400) # 24 hours
        else:
            cache.set(cache_key, cached_json, expire=86400)
       
        print(f"Retrieved {len(df_locations)} locations from SQL")
        return df_locations[['LOCATION_CODE', 'LOCATION_NAME', 'brand']]
    except Exception as e:
        print(f"Error fetching location codes: {e}")
        return pd.DataFrame(columns=['LOCATION_CODE', 'LOCATION_NAME', 'brand'])
def get_employee_names():
    cache_key = "employee_names"
    try:
        if isinstance(cache, redis.Redis):
            cached_data = cache.get(cache_key)
            if cached_data:
                cached_data = cached_data.decode('utf-8')
        else:
            cached_data = cache.get(cache_key)
       
        if cached_data:
            logger.info("Using cached employee names")
            return pd.read_json(StringIO(cached_data))
       
        conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={SQL_SERVER};"
            f"DATABASE={SQL_DATABASE};"
            f"UID={SQL_USERNAME};"
            f"PWD={SQL_PASSWORD};"
            "Connect Timeout=60;"
        )
        conn = pyodbc.connect(conn_str)
        query = "SELECT EMPLOYEE_NUMBER, FIRST_NAME, LAST_NAME FROM T_EMPLOYEE"
        df_employees = pd.read_sql(query, conn)
        conn.close()
       
        cached_json = df_employees[['EMPLOYEE_NUMBER', 'FIRST_NAME', 'LAST_NAME']].to_json()
        if isinstance(cache, redis.Redis):
            cache.set(cache_key, cached_json.encode('utf-8'), ex=86400) # 24 hours
        else:
            cache.set(cache_key, cached_json, expire=86400)
       
        return df_employees[['EMPLOYEE_NUMBER', 'FIRST_NAME', 'LAST_NAME']]
    except Exception as e:
        logger.error(f"Error fetching employee names: {e}")
        return pd.DataFrame(columns=['EMPLOYEE_NUMBER', 'FIRST_NAME', 'LAST_NAME'])
def get_date_range():
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    weekday = today.weekday()
    days_to_prev_sun = (weekday + 1) % 7
    if days_to_prev_sun == 0:
        days_to_prev_sun = 7
    prev_sun = today - timedelta(days=days_to_prev_sun)
    three_suns_ago = prev_sun - timedelta(weeks=2)
    dates = []
    current = three_suns_ago
    while current <= yesterday:
        dates.append(current.strftime('%d-%b-%y'))
        current += timedelta(days=1)
    return dates, three_suns_ago.strftime('%d-%b-%y'), yesterday.strftime('%d-%b-%y')
def fetch_location_data(location_code, location_name, brand, labor_date):
    try:
        location_code = location_code.zfill(4)
        url = f"https://webservices.net-chef.com/timeclock/v1/getAllTimeClockEnhanced?laborDate={labor_date}&locationCode={location_code}&includeNull=false"
        headers = {
            "accept": "application/json",
            "authenticationtoken": API_TOKEN,
            "password": API_PASSWORD,
            "sitename": API_SITENAME,
            "userid": API_USERID
        }
       
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Raw API Data for {location_code}: {data}")
       
        body = data if isinstance(data, list) else data.get('body', [])
        if not body:
            return pd.DataFrame()
       
        all_details = []
        for item in body:
            details = item.get('timeClockEnhancedDetailDetails', [])
            if details:
                df_details = pd.DataFrame(details)
                df_details['location_code'] = location_code
                df_details['location'] = location_name
                df_details['brand'] = brand
                df_details['laborDate'] = item.get('timeClockEnhancedHeaderDetails', {}).get('laborDate', labor_date)
                all_details.append(df_details)
       
        if all_details:
            df = pd.concat(all_details, ignore_index=True)
            df['refresh_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            return df
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data for {location_code}: {e} - Ignoring and continuing")
        return pd.DataFrame()
def fetch_data(force_refresh=False):
    cache_key = "timeclock_data"
    if force_refresh:
        if isinstance(cache, redis.Redis):
            cache.delete(cache_key)
        else:
            cache.delete(cache_key)
   
    try:
        if isinstance(cache, redis.Redis):
            cached_data = cache.get(cache_key)
            if cached_data:
                cached_data = cached_data.decode('utf-8')
        else:
            cached_data = cache.get(cache_key)
       
        if cached_data and not force_refresh:
            logger.info("Using cached data")
            df = pd.read_json(StringIO(cached_data), convert_dates=['clockOut_dt', 'laborDate'])
            logger.info(f"Cached DataFrame shape: {df.shape}, columns: {df.columns}")
            return df
    except Exception as e:
        logger.error(f"Cache read error: {e}")
   
    try:
        df_locations = get_location_codes()
        if df_locations.empty:
            return pd.DataFrame({
                'error': ["No valid data from API"],
                'location': ['Unknown'],
                'location_code': ['Unknown'],
                'brand': ['Unknown'],
                'laborDate': [pd.NaT],
                'clockOut': [pd.NaT],
                'employeeNumber': ['Unknown'],
                'first_name': ['Unknown'],
                'last_name': ['Unknown'],
                'refresh_time': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            })
       
        df_employees = get_employee_names()
        dates = get_date_range()[0]
       
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(fetch_location_data, str(row['LOCATION_CODE']), row['LOCATION_NAME'], row['brand'], d)
                for _, row in df_locations.iterrows() for d in dates
            ]
            all_data = [future.result() for future in as_completed(futures) if not future.result().empty]
       
        if not all_data:
            return pd.DataFrame({
                'error': ["No valid data from API"],
                'location': ['Unknown'],
                'location_code': ['Unknown'],
                'brand': ['Unknown'],
                'laborDate': [pd.NaT],
                'clockOut': [pd.NaT],
                'employeeNumber': ['Unknown'],
                'first_name': ['Unknown'],
                'last_name': ['Unknown'],
                'refresh_time': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            })
       
        df = pd.concat(all_data, ignore_index=True)
        if df.empty:
            return pd.DataFrame({
                'error': ["No late clockOut data found"],
                'location': ['Unknown'],
                'location_code': ['Unknown'],
                'brand': ['Unknown'],
                'laborDate': [pd.NaT],
                'clockOut': [pd.NaT],
                'employeeNumber': ['Unknown'],
                'first_name': ['Unknown'],
                'last_name': ['Unknown'],
                'refresh_time': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            })
       
        df['clockOut_dt'] = pd.to_datetime(df['clockOut'], format='%H:%M %m/%d/%Y', errors='coerce')
        df = df[df['clockOut_dt'].notna() & ((df['clockOut_dt'].dt.hour > 0) | ((df['clockOut_dt'].dt.hour == 0) & (df['clockOut_dt'].dt.minute > 0))) & (df['clockOut_dt'].dt.hour <= 4)]
       
        df['employeeNumber'] = df['employeeNumber'].astype(str)
        df_employees['EMPLOYEE_NUMBER'] = df_employees['EMPLOYEE_NUMBER'].astype(str)
        df = df.merge(df_employees.rename(columns={'EMPLOYEE_NUMBER': 'employeeNumber', 'FIRST_NAME': 'first_name', 'LAST_NAME': 'last_name'}),
                      on='employeeNumber', how='left')
       
        columns = ['location', 'location_code', 'brand', 'laborDate', 'employeeNumber', 'first_name', 'last_name', 'clockOut', 'clockOut_dt', 'refresh_time']
        df = df[columns]
       
        cached_json = df.to_json()
        if isinstance(cache, redis.Redis):
            cache.set(cache_key, cached_json.encode('utf-8'), ex=86400) # 24 hours
        else:
            cache.set(cache_key, cached_json, expire=86400)
       
        logger.info(f"Raw combined DataFrame shape: {df.shape}, columns: {df.columns}")
        return df
    except Exception as e:
        logger.error(f"Error in fetch_data: {e}")
        return pd.DataFrame({
            'error': [str(e)],
            'location': ['Unknown'],
            'location_code': ['Unknown'],
            'brand': ['Unknown'],
            'laborDate': [pd.NaT],
            'clockOut': [pd.NaT],
            'employeeNumber': ['Unknown'],
            'first_name': ['Unknown'],
            'last_name': ['Unknown'],
            'refresh_time': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        })
df = fetch_data()
dates, start_date, end_date = get_date_range()
# Prepare initial data for layout
if 'error' in df.columns:
    initial_table_data = []
    initial_refresh_text = "Error occurred"
    initial_alert_rows = [html.Tr(html.Td("Error fetching data: " + df['error'].iloc[0], style={'padding': '8px', 'border': '1px solid #dee2e6', 'textAlign': 'center', 'fontFamily': 'Inter', 'fontSize': '14px'}))]
else:
    filtered_df = df.copy()
    filtered_df['laborDate_dt'] = pd.to_datetime(filtered_df['laborDate'], errors='coerce')
    filtered_df = filtered_df.sort_values(['laborDate_dt', 'clockOut_dt'], ascending=[False, False])
    filtered_df.drop(columns=['laborDate_dt'], inplace=True)
    filtered_df['clockOut'] = filtered_df['clockOut_dt'].dt.strftime('%I:%M %p')
    filtered_df['laborDate'] = pd.to_datetime(filtered_df['laborDate'], errors='coerce').dt.strftime('%m/%d/%Y')
    initial_table_data = filtered_df[['location', 'employeeNumber', 'first_name', 'last_name', 'laborDate', 'clockOut']].to_dict('records')
   
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
        initial_alert_rows = [html.Tr([html.Th('Alert Message', style={'backgroundColor': '#dc3545', 'color': 'white', 'padding': '8px', 'border': '1px solid #dee2e6', 'fontFamily': 'Inter', 'textAlign': 'left', 'fontSize': '14px'})])]
        for alert in alerts:
            initial_alert_rows.append(html.Tr([html.Td(alert, style={'backgroundColor': '#fff3cd', 'padding': '8px', 'border': '1px solid #dee2e6', 'fontFamily': 'Inter', 'textAlign': 'left', 'fontSize': '14px'})]))
    else:
        initial_alert_rows = [html.Tr([html.Td("No alerts", colSpan=1, style={'padding': '8px', 'border': '1px solid #dee2e6', 'textAlign': 'center', 'fontFamily': 'Inter', 'fontSize': '14px'})])]
   
    initial_refresh_text = f"Last refreshed: {df['refresh_time'].iloc[0] if 'refresh_time' in df.columns else 'Unknown'} | {len(df)} late clockOut events across {len(dates)} days"
location_options = [{'label': loc, 'value': loc} for loc in sorted(df['location'].unique())]
app.layout = dbc.Container(fluid=True, children=[
    html.Div([
        html.H1('Timeclock Dashboard (Five Guys USA)', style={'textAlign': 'center', 'color': '#0056b3', 'fontSize': '30px', 'fontFamily': 'Poppins', 'fontWeight': '700', 'marginBottom': '10px', 'marginTop': '20px'}),
        html.Div(f"Date Range: {start_date} to {end_date}", style={'textAlign': 'center', 'fontSize': '16px', 'color': '#6c757d', 'marginBottom': '10px', 'fontWeight': '400', 'fontFamily': 'Inter'}),
        html.Div(id='refresh-time', children=initial_refresh_text, style={'textAlign': 'center', 'fontSize': '14px', 'color': '#6c757d', 'marginBottom': '10px', 'fontWeight': '400', 'fontFamily': 'Inter'}),
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
                                options=location_options,
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
                data=initial_table_data,
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
            html.Div(id='dummy', style={'display': 'none'})
        ]),
        style={'boxShadow': '0 4px 8px rgba(0,0,0,0.1)', 'borderRadius': '10px', 'backgroundColor': 'white', 'margin': '20px auto', 'maxWidth': '80%'}
    ),
    dcc.Interval(id='refresh-interval', interval=15*60*1000, n_intervals=0, disabled=True)
])
# Clientside callback to make entire header clickable for sorting
clientside_callback(
    """
    function(id) {
        const headers = document.querySelectorAll('.dash-spreadsheet-inner th.dash-header');
        headers.forEach(header => {
            header.addEventListener('click', function(e) {
                const sortIcon = header.querySelector('.column-header--sort');
                if (sortIcon) {
                    sortIcon.click();
                }
            });
        });
        return 'done';
    }
    """,
    Output('dummy', 'children'),
    Input('late-clockout-table', 'id')
)
# Clientside callback to convert refresh time to local timezone
clientside_callback(
    """
    function(text) {
        if (!text || text === 'Refresh in Progress' || text === 'Error occurred') return text;
        const parts = text.split(' | ');
        if (parts.length < 2) return text;
        const timestamp = parts[0].replace('Last refreshed: ', '');
        const date = new Date(timestamp + 'Z');  // Treat as UTC
        if (isNaN(date)) return text;
        const localTime = date.toLocaleString();
        return 'Last refreshed: ' + localTime + ' | ' + parts[1];
    }
    """,
    Output('refresh-time', 'children'),
    Input('refresh-time', 'children')
)
@app.callback(
    [
        Output('late-clockout-table', 'data'),
        Output('refresh-time', 'children'),
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
   
    if triggered_id == 'refresh-button' and n_clicks > 0:
        df = fetch_data(force_refresh=True)
        return [], "Refresh in Progress", True, False, [], None
    if triggered_id == 'refresh-interval' and n_intervals > 0:
        df = fetch_data(force_refresh=True)
    if 'error' in df.columns:
        return [], "Error occurred", False, True, [html.Tr(html.Td("Error fetching data: " + df['error'].iloc[0], style={'padding': '8px', 'border': '1px solid #dee2e6', 'textAlign': 'center', 'fontFamily': 'Inter', 'fontSize': '14px'}))], None
   
    filtered_df = df.copy() # Start with explicit copy to avoid warnings
    if selected_location:
        filtered_df = filtered_df[filtered_df['location'] == selected_location]
    if search_value:
        search_lower = search_value.lower()
        filtered_df = filtered_df[
            filtered_df['employeeNumber'].str.lower().str.contains(search_lower, na=False) |
            filtered_df['first_name'].str.lower().str.contains(search_lower, na=False) |
            filtered_df['last_name'].str.lower().str.contains(search_lower, na=False)
        ]
   
    # Handle table sorting
    if sort_by:
        sort_col = sort_by[0]['column_id']
        sort_direction = sort_by[0]['direction']
        if sort_col == 'clockOut':
            # Modified sorting for clockOut to use time of day only
            filtered_df = filtered_df.sort_values('clockOut_dt', key=lambda s: s.dt.time, ascending=(sort_direction == 'asc'))
        elif sort_col == 'laborDate':
            filtered_df = filtered_df.sort_values('laborDate', key=lambda x: pd.to_datetime(x, errors='coerce'), ascending=(sort_direction == 'asc'))
        else:
            filtered_df = filtered_df.sort_values(sort_col, ascending=(sort_direction == 'asc'))
    else:
        filtered_df['laborDate_dt'] = pd.to_datetime(filtered_df['laborDate'], errors='coerce')
        filtered_df = filtered_df.sort_values(['laborDate_dt', 'clockOut_dt'], ascending=[False, False])
        filtered_df.drop(columns=['laborDate_dt'], inplace=True)
   
    # Format clockOut and laborDate
    filtered_df['clockOut'] = filtered_df['clockOut_dt'].dt.strftime('%I:%M %p')
    filtered_df['laborDate'] = pd.to_datetime(filtered_df['laborDate'], errors='coerce').dt.strftime('%m/%d/%Y')
    table_data = filtered_df[['location', 'employeeNumber', 'first_name', 'last_name', 'laborDate', 'clockOut']].to_dict('records')
   
    # Generate alerts using unfiltered df
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
   
    refresh_text = f"Last refreshed: {df['refresh_time'].iloc[0] if 'refresh_time' in df.columns else 'Unknown'} | {len(filtered_df)} late clockOut events across {len(dates)} days"
   
    # Export to Excel logic
    export_data = None
    if triggered_id == 'export-button' and export_n_clicks > 0:
        export_df = filtered_df[['location', 'employeeNumber', 'first_name', 'last_name', 'laborDate', 'clockOut']].copy()
        export_df['clockOut'] = pd.to_datetime(export_df['clockOut'], format='%I:%M %p', errors='coerce').dt.strftime('%H:%M')
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            export_df.to_excel(writer, sheet_name='Late Clockouts', index=False)
        export_data = dcc.send_bytes(output.getvalue(), filename='late_clockouts.xlsx')
        logger.info("Export to Excel triggered successfully")
   
    return table_data, refresh_text, False, True, alert_rows, export_data
if __name__ == '__main__':
    app.run_server(debug=False)
