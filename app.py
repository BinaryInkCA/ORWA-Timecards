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

# Configure logging to stdout with flush
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True, format='%(asctime)s - %(levelname)s - %(message)s')
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
                    df['refresh_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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
    
        # Add semaphore to limit concurrency and prevent overload/timeout
        sem = asyncio.Semaphore(50)  # Adjust based on testing; 50 concurrent tasks
        async def limited_fetch(row, d):
            async with sem:
                return await fetch_location_data(str(row['LOCATION_CODE']), row['LOCATION_NAME'], row['brand'], d)
        
        tasks = [limited_fetch(row, d) for _, row in df_locations.iterrows() for d in dates]
        all_data = [df for df in await asyncio.gather(*tasks) if not df.empty]
    
        if not all_data:
            logger.warning("No data from any API calls - using fallback")
            return pd.DataFrame()
        
        df = pd.concat(all_data, ignore_index=True)
        df = df.merge(df_employees, left_on='employeeNumber', right_on='EMPLOYEE_NUMBER', how='left').drop(columns='EMPLOYEE_NUMBER', errors='ignore')
        df['clockOut_dt'] = pd.to_datetime(df['clockOut'], errors='coerce')
        return df
    except Exception as e:
        logger.error(f"Error in fetch_data: {e}")
        return pd.DataFrame()

clientside_callback(
    """
    function(text) {
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
    
    if triggered_id is None:
        # Initial load: empty data
        return [], "Press Refresh to load data | 0 late clockOut events", False, True, [html.Tr([html.Td("No alerts", colSpan=1, style={'padding': '8px', 'border': '1px solid #dee2e6', 'textAlign': 'center', 'fontFamily': 'Inter', 'fontSize': '14px'})])], None, date_range_text, location_options
    
    if triggered_id in ['refresh-button', 'refresh-interval'] and (n_clicks > 0 or n_intervals > 0):
        df = asyncio.run(fetch_data())
    
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

# Add the missing layout
app.layout = html.Div([
    html.H1("Late ClockOut Dashboard", style={'fontFamily': 'Poppins', 'fontWeight': '700', 'textAlign': 'center', 'marginBottom': '20px'}),
    html.Div([
        dcc.Dropdown(id='location-filter', options=[], placeholder="Select Location", className='dropdown', style={'width': '300px', 'marginRight': '10px'}),
        dcc.Input(id='search-input', type='text', placeholder="Search by Employee", className='filter-box', style={'width': '300px', 'marginRight': '10px'}),
        html.Button('Refresh', id='refresh-button', className='btn-primary', style={'marginRight': '10px'}),
        html.Button('Export to Excel', id='export-button', className='btn-success')
    ], style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px'}),
    html.Div(id='date-range-display', style={'textAlign': 'center', 'marginBottom': '10px', 'fontSize': '16px', 'fontFamily': 'Inter'}),
    html.Div(id='utc-refresh-time', style={'display': 'none'}),
    html.Div(id='refresh-time', style={'textAlign': 'center', 'marginBottom': '20px', 'fontSize': '14px', 'fontFamily': 'Inter', 'color': '#6c757d'}),
    dash_table.DataTable(
        id='late-clockout-table',
        columns=[{"name": i, "id": i} for i in ['location', 'employeeNumber', 'first_name', 'last_name', 'laborDate', 'clockOut']],
        data=[],
        sort_action='native',
        style_table={'overflowX': 'auto', 'border': '1px solid #dee2e6', 'borderRadius': '5px', 'margin': '0 auto', 'width': '80%'},
        style_cell={'textAlign': 'left', 'padding': '8px', 'border': '1px solid #dee2e6', 'fontFamily': 'Inter', 'fontSize': '14px'},
        style_header={'backgroundColor': '#007bff', 'color': 'white', 'fontWeight': '500', 'textAlign': 'left', 'padding': '8px', 'borderBottom': '2px solid #dee2e6'},
        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#f9f9f9'}, {'if': {'state': 'active'}, 'backgroundColor': '#e9ecef'}]
    ),
    html.H2("Alerts", style={'fontFamily': 'Poppins', 'fontWeight': '500', 'textAlign': 'center', 'marginTop': '30px', 'marginBottom': '10px'}),
    html.Table(id='alerts-table', style={'width': '80%', 'margin': '0 auto', 'borderCollapse': 'collapse', 'border': '1px solid #dee2e6'}),
    dcc.Download(id='download-excel'),
    dcc.Interval(id='refresh-interval', interval=60*60*1000, disabled=True)  # 1 hour auto-refresh, initially disabled
])

# Set initial empty df for lazy loading
df = pd.DataFrame()

if __name__ == '__main__':
    app.run_server(debug=False)
