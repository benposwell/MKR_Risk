import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta, date
import dateutil.relativedelta
# import scipy.optimize as sco
from pandas.tseries.offsets import BDay as BusinessDays
from numerize import numerize
import warnings
warnings.filterwarnings("ignore")
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pd.options.display.float_format = '{:.4f}'.format


def standardize_dates(date_str):
    """
    Convert dates from both MM/DD/YYYY and DD/MM/YYYY formats to a standard datetime format.
    
    Args:
        date_str: String representing a date
        
    Returns:
        datetime object in standardized format
    """
    try:
        # First try MM/DD/YYYY format
        return pd.to_datetime(date_str, format='%m/%d/%Y')
    except ValueError:
            # If that fails, try DD/MM/YYYY format
            try:
                return pd.to_datetime(date_str, format='%d/%m/%Y')
            except ValueError:
                # format for YYYY-MM-DD
                return pd.to_datetime(date_str, format='%Y-%m-%d')
    

# After processing your dates, handle NaT values in the Month-Year conversion
def safe_strftime(x):
    """
    Safely convert datetime to Month-Year string, handling NaT values
    """
    return x.strftime("%B-%Y") if pd.notna(x) else None



def process_csv_with_mixed_dates(df):
    """
    Process a dataframe with mixed date formats in the first column.
    
    Args:
        df: pandas DataFrame with dates in the first column
        
    Returns:
        DataFrame with standardized dates in the first column
    """
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Convert the first column to datetime using our custom function
    df_processed.iloc[:, 0] = df_processed.iloc[:, 0].apply(standardize_dates)
    
    # Handle any None values that might have resulted from failed conversions
    failed_conversions = df_processed.iloc[:, 0].isna()
    if failed_conversions.any():
        print(f"Warning: Failed to convert {failed_conversions.sum()} dates")
        print("Rows with failed conversions:")
        print(df[failed_conversions])
    
    return df_processed


def get_position_desc(sector, market, instrument_name):
    desc = ''
    
    if sector == 'Cash':
        desc = ' '.join(instrument_name.split('CASH')[:-1])
    elif sector in ['Commodity', 'Currency']:
        desc = ' '.join(instrument_name.split()[:2])
    elif sector == 'Fixed Income':
        if 'INTRTSWP' in instrument_name:
            desc = 'INTRTSWP '+ market
        elif 'OIS_SWAP' in instrument_name:
            desc = 'OIS_SWAP '+ market
        elif 'FXO' in instrument_name:
            desc = ' '.join(instrument_name.split()[:1])
        else:
            desc = market
    else:
        desc = market
    
    return desc

def get_pnl_desc(sector, market, instrument_name):
    desc = market
    
    if sector == 'Cash':
        desc = instrument_name.split('CASHINT')[0]
    elif sector in ['Commodity', 'Equity']:
        desc = market
    elif sector == 'Currency':
        desc = instrument_name if 'FX' in instrument_name else instrument_name.split('F')[0]
    elif sector == 'Fixed Income':
        desc = 'IRS ' + market if 'IRS' in instrument_name else market
    
    return desc

def rename_positions(instrument, type_inst, sector, market, desc):
    if sector == 'Fixed Income':
        if type_inst == 'Mutual Fund':
            desc = ''.join(instrument.split()[:-1])
        else:
            desc = desc
    elif sector == 'Currency':
        if market == 'Forward':
            desc = desc + ' ' + market
        else:
            desc = desc
    else:
        desc = desc
        
    return desc

def rolling_average(df, window): #calculate rolling average with given window size
    temp = None
    if type(df) == pd.core.series.Series: #check whether the given return stream is in a Series or a Dataframe
        temp = df.rolling(window,min_periods=window-3).mean()
    else:
        temp = df[df.columns[0]].rolling(window,min_periods=window-3).mean()
    return temp

def get_currency_format(curr):
    formatted_curr = '${0:,.2f}'.format(curr) if curr>=0 else '(${0:,.2f})'.format(abs(curr))
    return formatted_curr

def get_pm_returns(csv_dataframes):
    df = csv_dataframes['Daily_Returns.csv']
    df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
    return df

def get_trading_level_history(csv_dataframes, map_id='MLH240_27'):
    trading_level_history_df = csv_dataframes['Trading_Level.csv']
    return trading_level_history_df

def rolling_returns(df, window): #calculate rolling returns with given window size
    if type(df) == pd.core.series.Series: #check whether the given return stream is in a Series or a Dataframe
        df[str(window)+' Days Rolling Return'] = df.rolling(window,min_periods=window-3)\
        .apply(lambda x: (x+1).prod()-1, raw = False)
    else:
        df[str(window)+' Days Rolling Return'] = df[df.columns[0]].rolling(window,min_periods=window-3)\
        .apply(lambda x: (x+1).prod()-1, raw=False)
        
    #stores rolling returns stats in a Dataframe called stats
    return df[str(window)+' Days Rolling Return']

def rolling_std(df, window):
    if type(df) == pd.core.series.Series:
        df[str(window)+' Days Rolling STD'] = df.rolling(window,min_periods=window-3).std()*math.sqrt(252)
    else:
        df[str(window)+' Days Rolling STD'] = df[df.columns[0]].rolling(window,min_periods=window-3).std()*math.sqrt(252)
        
    return df[str(window)+' Days Rolling STD']

def rolling_Sharpe_Ratio(df, window):
    df[str(window)+' Days Rolling SR'] = rolling_returns(df, window)/(rolling_std(df, window)*math.sqrt(window/252))
    return df[str(window) + ' Days Rolling SR']

def rolling_corr(df,window,benchmark):
    df[str(window)+' Days Rolling Correlation with '+benchmark.name] = df[df.columns[0]]\
    .rolling(window, min_periods=window-3).corr(benchmark)
    return df[str(window)+' Days Rolling Correlation with '+benchmark.name]

def rolling_beta(df, window, benchmark):
    df[str(window)+' Days Rolling Beta with '+benchmark.name] = \
    df[df.columns[0]].rolling(window, min_periods = window-3)\
    .cov(benchmark)/benchmark\
    .rolling(window, min_periods = window-3).var()
    
    return df[str(window)+' Days Rolling Beta with '+benchmark.name]

def drawdown(df):
    df['Cumulative Returns'] = (df[df.columns[0]]+1).cumprod()
    df['Drawdown'] = df['Cumulative Returns'].div(df['Cumulative Returns'].cummax()) - 1
    df['Cumulative Returns'] = df['Cumulative Returns'] - 1
    return df['Drawdown']

def unknown_sector_mapping(instrument_type):
    if instrument_type == 'Interest Rate Future':
        sector = 'Fixed Income'
    elif 'FX' in instrument_type:
        sector = 'Currency'
    else:
        sector = 'Unknown'
    return sector

def get_pm_margins(csv_dataframes):
    df = csv_dataframes['Margins.csv']
    return df

def get_pm_name(map_name):
    map_name_dict = {
        '240_27': 'MKR - MAP 240_27',
    }
    map_id = map_name[3:]
    return map_name_dict[map_id]

def closed_positions(instrument_name, admin_id, sector):
    name = ''
    if pd.isna(instrument_name):
        if sector == 'Cash':
            name = admin_id[:3] + ' Interest'
        else:
            name = admin_id + ' (Closed)'
    else:
        name = instrument_name
    return name

def convert_to_bps(x, sector='IR'):
    t = ''
    
    limit = -2 if sector=='IR' else -3
    if 'Neg' in x:
        t = int(x[3:limit])*-1
    elif 'Pos' in x:
        t = int(x[3:limit])
    return t

def get_account_limit_type(desc, limit_type, sector, market):
    desc_limit = desc
    if 'Maximum Single Net Exposure' in desc:
        desc_limit = 'Maximum Single Net Exposure for Sector: '+ str(sector) + ', Market: '+ str(market)
    elif limit_type == '%NRA':
        desc_limit = 'Maximum Gross Exposure for Sector: '+ str(sector) + ', Market: '+ str(market)
    elif limit_type == '%Margin/NRA':
        desc_limit = 'Account Total Margin to NRA Ratio'
    return desc_limit

def get_value_exposure(desc,sector, market, value_type, positions_snapshot, current_margin_val, csv_dataframes):
    value = 0
    exp_type = 'Net Exposure' if 'Net Exposure' in desc else 'Margin/NRA' if 'Margin to NRA' in desc else 'Gross Exposure'
    pos_type = 'single' if 'Single' in desc else 'total'
    
    if value_type == '# level III contracts':
        value = positions_snapshot['Level 3 Assets'].sum()
    elif value_type == '# short options' or value_type == '# put spread':
        value = positions_snapshot['Short Options'].sum()
    elif value_type == '# futures per market':
        value = positions_snapshot['CFTC'].sum()
    elif exp_type == 'Margin/NRA':
        value = current_margin_val
    elif 'volume' in desc.lower():
        liqlimits_df = csv_dataframes['LiqLimits.csv']
        df = liquidity_limits_check(positions_snapshot, liqlimits_df)
        value = len(df[df['Satisfy Limit'] == 'No'])
    elif str(market) == 'Portfolio' and str(sector) == 'Portfolio':
        value = positions_snapshot[exp_type].sum() if pos_type == 'total' else positions_snapshot[exp_type].max()
    elif str(market) == 'All':
        if sector == 'Currency':
            currency_group_net_exp = pd.pivot_table(positions_snapshot[positions_snapshot['Sector'] == 'Currency'], 
                               values=['Net Exposure'], 
                               index=['Currency'], aggfunc="sum")
            currency_gross_exp = abs(currency_group_net_exp['Net Exposure']).sum()
            if exp_type == 'Net Exposure':
                value = currency_group_net_exp['Net Exposure'].sum() if pos_type == 'total' else currency_group_net_exp['Net Exposure'].max()
            else:
                value = abs(currency_group_net_exp['Net Exposure']).sum() if pos_type == 'total' else abs(currency_group_net_exp['Net Exposure']).max() 
        else:
            value = positions_snapshot[positions_snapshot['Sector'] == sector][exp_type].sum() if pos_type == 'total' else positions_snapshot[positions_snapshot['Sector'] == sector][exp_type].max()
    else:
        column_name = 'MarketGroup' if market == 'Metals' else 'Market'
        value = positions_snapshot[(positions_snapshot['Sector'] == sector) & (positions_snapshot[column_name].str.contains(market))][exp_type].sum() if pos_type == 'total' else positions_snapshot[(positions_snapshot['Sector'] == sector) & (positions_snapshot[column_name].str.contains(market))][exp_type].max()
    return value

def get_limit_value(limit_type, limit, current_trading_level):
    limit_value = 0
    if limit_type in ["#level3", "#options", "cftc", '#putspread']:
        limit_value = 0
    elif limit_type == '%NRA':
        limit_value = current_trading_level * limit/100
    elif limit_type == '%Margin/NRA':
        #limit_value = ("{:.2f}%").format(limit)
        limit_value = current_trading_level * limit/100
    return limit_value

def satisfy_limit(limit_type, value, limit):
    satisfy = 'N/A'
    if limit_type == '(+/-)%NRA':
        satisfy = 'Yes' if value >= limit[0] and value <= limit[1] else 'No'
    else:
        satisfy = 'Yes' if value <= limit else 'No'
    return satisfy

def liquidity_limits_check(positions_snapshot, liqlimits_df):
    liqlimits_df = liqlimits_df[['BBG Ticker', 'VOLUME_AVG_30D']]
    liqlimits_df['Ticker'] = liqlimits_df['BBG Ticker'].apply(lambda x: x.split(' ')[0][:-1])
    futures_df = positions_snapshot[positions_snapshot['Ticker'].isin(liqlimits_df['Ticker'].values)]
    futures_df = futures_df[['InstrumentName', 'Ticker', 'Sector', 'Market', 'Quantity', 'Net Exposure']]
    check_df = pd.merge(futures_df, liqlimits_df, left_on=['Ticker'], right_on=['Ticker'], how='left')
    check_df['Limit'] = round(0.03*check_df['VOLUME_AVG_30D'], 0)
    check_df['Satisfy Limit'] = check_df.apply(lambda x: 'Yes' if x['Quantity'] < x['Limit'] else 'No', axis=1)
    check_df.drop(columns=['Ticker'], inplace=True)
    return check_df

def get_level3_assets(instrument_type, quantity):
    level1_assets = ['Cash', 'Equity', 'Mutual Fund', 'Generic Bond']
    level2_assets = ['Option on Interest Rate Future', 'Commodity Future', 'Foreign Exchange Option', 'Rate Swap',
                    'Bond Future', 'Interest Rate Future', 'Foreign Exchange Futures', 'Equity Future', 'Equity Option',
                    'Foreign Exchange Forward', 'Option on Commodity Future', 'Option on Equity Future', 'FX Digital Option',
                    'Option on Bond Future', 'Credit Default Swap','Overnight Indexed Swap']
    if instrument_type in level1_assets or instrument_type in level2_assets:
        return 0
    else:
        return abs(quantity)
    
def get_value_type(desc, limit_type):
    value_type = 'gross'
    value_limit_map = {'#level3': '# level III contracts', 
                           '#options': '# short options', 
                           '#cftc': '# futures per market',
                           '#putspread': '# put spread'}
    if 'naked short options' in desc.lower():
        value_type = value_limit_map[limit_type]
    elif 'RIC' in desc:
        value_type = 'RIC'
    elif 'Margin to NRA' in desc:
        value_type = 'Margin/NRA'
    elif 'Sub-Investment' in desc:
        value_limit_map = {'#level3': '# level III contracts', 
                           '#options': '# short options', 
                           '#cftc': '# futures per market',
                           '#putspread': '# put spread'}
        value_type = value_limit_map[limit_type]
    elif 'net' in desc.lower():
        value_type = 'net'
    return value_type
    
def get_number_short_options(instrument_type, quantity):
    if 'Option' in instrument_type and quantity < 0:
        return abs(quantity)
    else:
        return 0

def get_limit_usage(value, limit, limit_type):
    usage = 0
    if limit_type == '(+/-)%NRA' and type(limit) not in [int, float]:
        usage = value/abs(limit[1])
    else:
        usage = value/limit if limit != 0 else 0
    return usage
    
def get_cftc_spot_month_limit(instrument_name, instrument_type, ticker, quantity):
    cftc_spot_month_limit_df = pd.read_excel('data/CFTC Limits.xlsx', skiprows=1)
    cftc_spot_month_limit_df = cftc_spot_month_limit_df.head(24)
    if 'Future' not in instrument_type:
        return False
    elif ticker not in cftc_spot_month_limit_df['Ticker'].values:
        return False
    else:
        expiry = instrument_name.split(' ')[-1]
        current_month = datetime.today().strftime("%B").upper()[:3]
        current_year = datetime.today().strftime("%Y")[2:]
        current = current_month + current_year
        if expiry == current:
            cftc_qty = cftc_spot_month_limit_df[cftc_spot_month_limit_df['Ticker'] == ticker]['2020 Final Rulemaking Federal Spot Month Limit Level'].tolist()[0]
            cftc = True if quantity > cftc_qty else False
            return cftc
        return 