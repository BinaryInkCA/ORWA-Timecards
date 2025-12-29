import sys
sys.path.insert(0, "./.python_packages/lib/python3.10/site-packages")
import dash
from dash import dcc, html, Input, Output, callback_context, State, clientside_callback
import pandas as pd
import io
import aiohttp
import pyodbc
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc
from io import StringIO
from dash.exceptions import PreventUpdate
from dash import dash_table
import os
import logging
import asyncio
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Initialize app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, '/assets/style.css'])
server = app.server
# Environment variables for Azure
API_TOKEN = os.getenv('API_TOKEN', "5b67e106-173f-4281-a83f-87b2bdc3b1f1")
API_PASSWORD = os.getenv('API_PASSWORD', "Welcome1")
API_SITENAME = os.getenv('API_SITENAME', "fivestar")
API_USERID = os.getenv('API_USERID', "jeff.thompson")
SQL_SERVER = os.getenv('SQL_SERVER', "SQL-03")
SQL_DATABASE = os.getenv('SQL_DATABASE', "TECHSYS")
SQL_USERNAME = os.getenv('SQL_USERNAME')
SQL_PASSWORD = os.getenv('SQL_PASSWORD')
def get_location_codes() -> pd.DataFrame:
    try:
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
    
        logger.info(f"Retrieved {len(df_locations)} locations from SQL")
        return df_locations[['LOCATION_CODE', 'LOCATION_NAME', 'brand']]
    except Exception as e:
        logger.error(f"Error fetching location codes: {e}")
        return pd.DataFrame(columns=['LOCATION_CODE', 'LOCATION_NAME', 'brand'])
def get_employee_names() -> pd.DataFrame:
    try:
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
    
        return df_employees[['EMPLOYEE_NUMBER', 'FIRST_NAME', 'LAST_NAME']]
    except Exception as e:
        logger.error(f"Error fetching employee names: {e}")
        return pd.DataFrame(columns=['EMPLOYEE_NUMBER', 'FIRST_NAME', 'LAST_NAME'])
def get_date_range() -> tuple[list[str], str, str]:
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
async def fetch_location_data(location_code: str, location_name: str, brand: str, labor_date: str) -> pd.DataFrame:
    max_retries = 3
    retry_delay = 1  # initial delay in seconds
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
    
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, timeout=30) as response:
                        response.raise_for_status()
                        data = await response.json()
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
                    df['refresh_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S') # Assume local time; adjust to UTC if needed
                    return df
                return pd.DataFrame()
            except aiohttp.ClientError as e:
                logger.warning(f"Attempt {attempt+1} failed for {location_code}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # exponential backoff
                else:
                    logger.error(f"Max retries exceeded for {location_code}: {e} - Ignoring and continuing")
                    return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error fetching data for {location_code}: {e}")
        return pd.DataFrame()
async def fetch_data() -> pd.DataFrame:
    try:
        df_locations = get_location_codes()
        if df_locations.empty:
            logger.warning("No locations fetched from SQL - using fallback")
            return pd.DataFrame({
                'error': ["No valid locations from SQL"],
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
    
        tasks = [
            fetch_location_data(str(row['LOCATION_CODE']), row['LOCATION_NAME'], row['brand'], d)
            for _, row in df_locations.iterrows() for d in dates
        ]
        all_data = [df for df in await asyncio.gather(*tasks) if not df.empty]
    
        if not all_data:
            logger.warning("No data from any API calls - using fallback")
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
# Initial data fetch
df_locations_initial = get_location_codes()
location_options_initial = [{'label': row['LOCATION_NAME'], 'value': row['LOCATION_NAME']} for _, row in df_locations_initial.sort_values('LOCATION_NAME').iterrows()] if not df_locations_initial.empty else [{'label': 'Unknown', 'value': 'Unknown'}]

df = asyncio.run(fetch_data())
dates_initial, start_date_initial, end_date_initial = get_date_range()
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
    # Vectorized alert generation
    location_counts = df.groupby('location').size()
    high_locations = location_counts[location_counts > 5].reset_index()
    loc_alerts = high_locations.apply(lambda row: f"High number of late clockouts at {row['location']}!", axis=1).tolist()
    employee_counts = df.groupby(['employeeNumber', 'first_name', 'last_name', 'location']).size().reset_index(name='count')
    high_employees = employee_counts[employee_counts['count'] > 3]
    emp_alerts = high_employees.apply(lambda row: f"{row['first_name']} {row['last_name']} at {row['location']} has repeated late clock outs", axis=1).tolist()
    alerts = loc_alerts + emp_alerts
    if alerts:
        initial_alert_rows = [html.Tr([html.Th('Alert Message', style={'backgroundColor': '#dc3545', 'color': 'white', 'padding': '8px', 'border': '1px solid #dee2e6', 'fontFamily': 'Inter', 'textAlign': 'left', 'fontSize': '14px'})])]
        for alert in alerts:
            initial_alert_rows.append(html.Tr([html.Td(alert, style={'backgroundColor': '#fff3cd', 'padding': '8px', 'border': '1px solid #dee2e6', 'fontFamily': 'Inter', 'textAlign': 'left', 'fontSize': '14px'})]))
    else:
        initial_alert_rows = [html.Tr([html.Td("No alerts", colSpan=1, style={'padding': '8px', 'border': '1px solid #dee2e6', 'textAlign': 'center', 'fontFamily': 'Inter', 'fontSize': '14px'})])]
    initial_refresh_text = f"Last refreshed: {df['refresh_time'].iloc[0] if 'refresh_time' in df.columns else 'Unknown'} | {len(df)} late clockOut events across {len(dates_initial)} days"

app.layout = dbc.Container(fluid=True, children=[
    html.Div([
        html.H1('Timeclock Dashboard (Five Guys USA)', style={'textAlign': 'center', 'color': '#0056b3', 'fontSize': '30px', 'fontFamily': 'Poppins', 'fontWeight': '700', 'marginBottom': '10px', 'marginTop': '20px'}),
        html.Div(id='date-range-display', children=f"Date Range: {start_date_initial} to {end_date_initial}", style={'textAlign': 'center', 'fontSize': '16px', 'color': '#6c757d', 'marginBottom': '10px', 'fontWeight': '400', 'fontFamily': 'Inter'}),
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
                                options=location_options_initial,
                                value=None,
                                placeholder="Select Location",
                                searchable=True  # Improved: make searchable for better UX
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
            dcc.Loading(
                id="loading",
                type="default",
                children=[
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
                ]
            ),
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
        const date = new Date(timestamp + 'Z'); // Treat as UTC
        if (isNaN(date)) return text;
        const localDate = date.toLocaleDateString('en-CA'); // YYYY-MM-DD
        const localTime = date.toLocaleTimeString('en-GB'); // HH:MM:SS
        const localTimestamp = localDate + ' ' + localTime;
        return 'Last refreshed: ' + localTimestamp + ' | ' + parts[1];
    }
    """,
    Output('refresh-time', 'children'),
    Input('utc-refresh-time', 'children')
)
@app.callback(
    [
        Output('late-clockout-table', 'data'),
        Output('utc-refresh-time', 'children'),
        Output('refresh-button', 'disabled'),
        Output('refresh-interval', 'disabled'),
        Output('alerts-table', 'children'),
        Output('download-excel', 'data'),
        Output('date-range-display', 'children'),
        Output('location-filter', 'options')
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
    
    # Always compute current date range and locations for robustness
    dates, start_date, end_date = get_date_range()
    date_range_text = f"Date Range: {start_date} to {end_date}"
    df_locations = get_location_codes()
    location_options = [{'label': row['LOCATION_NAME'], 'value': row['LOCATION_NAME']} for _, row in df_locations.sort_values('LOCATION_NAME').iterrows()] if not df_locations.empty else [{'label': 'Unknown', 'value': 'Unknown'}]
    
    if triggered_id in ['refresh-button', 'refresh-interval'] and (n_clicks > 0 or n_intervals > 0):
        df = asyncio.run(fetch_data())
        # Compute new table data, alerts, etc.
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
        # Generate alerts using unfiltered df (vectorized)
        location_counts = df.groupby('location').size()
        high_locations = location_counts[location_counts > 5].reset_index()
        loc_alerts = high_locations.apply(lambda row: f"High number of late clockouts at {row['location']}!", axis=1).tolist()
        employee_counts = df.groupby(['employeeNumber', 'first_name', 'last_name', 'location']).size().reset_index(name='count')
        high_employees = employee_counts[employee_counts['count'] > 3]
        emp_alerts = high_employees.apply(lambda row: f"{row['first_name']} {row['last_name']} at {row['location']} has repeated late clock outs", axis=1).tolist()
        alerts = loc_alerts + emp_alerts
        if alerts:
            alert_rows = [html.Tr([html.Th('Alert Message', style={'backgroundColor': '#dc3545', 'color': 'white', 'padding': '8px', 'border': '1px solid #dee2e6', 'fontFamily': 'Inter', 'textAlign': 'left', 'fontSize': '14px'})])]
            for alert in alerts:
                alert_rows.append(html.Tr([html.Td(alert, style={'backgroundColor': '#fff3cd', 'padding': '8px', 'border': '1px solid #dee2e6', 'fontFamily': 'Inter', 'textAlign': 'left', 'fontSize': '14px'})]))
        else:
            alert_rows = [html.Tr([html.Td("No alerts", colSpan=1, style={'padding': '8px', 'border': '1px solid #dee2e6', 'textAlign': 'center', 'fontFamily': 'Inter', 'fontSize': '14px'})])]
        refresh_text = f"Last refreshed: {df['refresh_time'].iloc[0] if 'refresh_time' in df.columns else 'Unknown'} | {len(filtered_df)} late clockOut events across {len(dates)} days"
        return table_data, refresh_text, False, True, alert_rows, None, date_range_text, location_options
    if 'error' in df.columns:
        return [], "Error occurred", False, True, [html.Tr(html.Td("Error fetching data: " + df['error'].iloc[0], style={'padding': '8px', 'border': '1px solid #dee2e6', 'textAlign': 'center', 'fontFamily': 'Inter', 'fontSize': '14px'}))], None, date_range_text, location_options
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
    # Generate alerts using unfiltered df (vectorized)
    location_counts = df.groupby('location').size()
    high_locations = location_counts[location_counts > 5].reset_index()
    loc_alerts = high_locations.apply(lambda row: f"High number of late clockouts at {row['location']}!", axis=1).tolist()
    employee_counts = df.groupby(['employeeNumber', 'first_name', 'last_name', 'location']).size().reset_index(name='count')
    high_employees = employee_counts[employee_counts['count'] > 3]
    emp_alerts = high_employees.apply(lambda row: f"{row['first_name']} {row['last_name']} at {row['location']} has repeated late clock outs", axis=1).tolist()
    alerts = loc_alerts + emp_alerts
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
    return table_data, refresh_text, False, True, alert_rows, export_data, date_range_text, location_options
if __name__ == '__main__':
    app.run_server(debug=False)
