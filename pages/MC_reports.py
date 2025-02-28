
import streamlit as st
from utils.MC_utils import *
from utils.MC_plotting import *
from utils.MC_error_logging import *

import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta, date
from pandas.tseries.offsets import BDay as BusinessDays
from numerize import numerize
import warnings
warnings.filterwarnings("ignore")
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pd.options.display.float_format = '{:.4f}'.format


# st.set_page_config(layout="wide")
st.divider()

def check_all_csvs_exist(csv_dataframes):
    sample_list = ['MKR_MC_PNL', 'MKR_MC_POSITIONS', 'MKR_MC_VAR', 'MKR_PNL_History', 'MKR_Positions_History', 'MKR_VAR_History', 'Trading_Level', 'Margins', 'LiqLimits', 'Daily_Returns', 'Benchmarks']
    retrieved_list = csv_dataframes.keys()
    for key in sample_list:
        if not any(key in retrieved_key for retrieved_key in retrieved_list):
            return False
    return True

def check_up_to_date(csv_dataframes):
    retrieved_list = csv_dataframes.keys()
    # Get yesterdays date in form YYYYMMDD
    yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y%m%d')
    for key in retrieved_list:
        if "MKR_MC_POSITIONS" in key:
            if yesterday not in key:
                date_part = key.split('_')[6]
                try:
                    formatted_date = (datetime.strptime(date_part, '%Y%m%d') - timedelta(days=1)).strftime('%B %d, %Y')
                    st.warning("Warning - Positions file is of date: " + formatted_date)
                except ValueError:
                    st.warning("Warning - Positions file is of date: " + key)
        if "MKR_MC_PNL" in key:
            if yesterday not in key:
                date_part = key.split('_')[6]
                try:
                    formatted_date = (datetime.strptime(date_part, '%Y%m%d') - timedelta(days=1)).strftime('%B %d, %Y')
                    st.warning("Warning - P&L file is of date: " + formatted_date)
                except ValueError:
                    st.warning("Warning - P&L file is of date: " + key)
        if "MKR_MC_VAR" in key:
            if yesterday not in key:
                date_part = key.split('_')[6]
                try:
                    formatted_date = (datetime.strptime(date_part, '%Y%m%d') - timedelta(days=1)).strftime('%B %d, %Y')
                    st.warning("Warning - VAR file is of date: " + formatted_date)
                except ValueError:
                    st.warning("Warning - VAR file is of date: " + key)

csv_dataframes = load_all_csvs_to_dataframe('Risk_Raw_Data')

if not check_all_csvs_exist(csv_dataframes):
    st.cache_data.clear()
    csv_dataframes = load_all_csvs_to_dataframe('Risk_Raw_Data')
    if not check_all_csvs_exist(csv_dataframes):
        st.error("Error - CSV files are not up to date - Admin has been notified")
        send_teams_message("Error - CSV files are not up to date.")
        st.stop()

check_up_to_date(csv_dataframes)


try:
    if csv_dataframes:
        positions_history = csv_dataframes['MKR_Positions_History.csv']
        pnl_history = csv_dataframes['MKR_PNL_History.csv']
        var_history = csv_dataframes['MKR_VAR_History.csv']

        positions_snapshot_name = ""
        pnl_snapshot_name = ""
        var_snapshot_name = ""
    
        for key in csv_dataframes.keys():
            if "MKR_MC_PNL" in key:
                pnl_snapshot_name = key
            if "MKR_MC_VAR" in key:
                var_snapshot_name = key
            if "MKR_MC_POSITIONS" in key:
                positions_snapshot_name = key

        positions_snapshot = csv_dataframes[positions_snapshot_name]
        pnl_snapshot = csv_dataframes[pnl_snapshot_name]
        var_snapshot = csv_dataframes[var_snapshot_name]
    else:
        st.warning("No CSV files found in the bucket.")


    # Code from Plotting IPYNB
    var_snapshot['Sector'] = var_snapshot['Sector'].apply(lambda x: 'Cash' if x == 'Inflation' else x)
    positions_snapshot['Sector'] = positions_snapshot.apply(lambda x: 'Equity' if x['Market'] == 'VIX' else x['Sector'], axis=1)
    positions_snapshot['Sector'] = positions_snapshot.apply(lambda x: unknown_sector_mapping(x['InstrumentType']) if x['Sector'] == 'Unknown' else x['Sector'], axis=1)
    positions_snapshot['Position Desc'] = positions_snapshot.apply(lambda x: get_position_desc(x['Sector'], x['Market'], x['InstrumentName']), axis=1)
    positions_snapshot = positions_snapshot[positions_snapshot['InstrumentType'] != 'Mutual Fund']
    positions_history = positions_history[positions_history['InstrumentType'] != 'Mutual Fund']
    positions_history['Sector'] = positions_history.apply(lambda x: unknown_sector_mapping(x['InstrumentType']) if x['Sector'] == 'Unknown' else x['Sector'], axis=1)
    positions_snapshot['MarketGroup'] = positions_snapshot['MarketGroup'].apply(lambda x: 'Australia & New Zealand' if x == 'Australia and New Zealand' else x)
    pnl_snapshot = pnl_snapshot[pnl_snapshot['Position Desc'] != 'UNITED STATES']
    trading_level_history_df = get_trading_level_history(csv_dataframes)
    trading_level_history_df.sort_values(by=['AsOf'])
    current_trading_level = trading_level_history_df['TradingLevel'].tail(1).tolist()[0]

    positions_snapshot_date = positions_snapshot['PositionAsOfDate'].tail(1).tolist()[0]

    # current_business_date = (datetime.today() - BusinessDays(1)).strftime("%B %d, %Y")
    current_business_date = positions_snapshot_date
    st.markdown("<h1 style='text-align: center;'>PM Exposure & Performance Summary</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Portfolio Manager: MKR Capital</h2>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'>Trading Level: ${numerize.numerize(current_trading_level)}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'>Positions as of EOD: {datetime.strptime(current_business_date, '%Y-%m-%d').strftime('%B %d, %Y')}</h3>", unsafe_allow_html=True)

    # Create refresh button which reruns load_all_csvs_to_dataframe
    refresh_button = st.button("Refresh", use_container_width=True)
    if refresh_button:
        st.cache_data.clear()
        csv_dataframes = load_all_csvs_to_dataframe('Risk_Raw_Data')

    exposure_summary = pd.pivot_table(positions_snapshot, 
                                values=['Gross Exposure', 'Net Exposure', 'MSCIBeta1d1Y', 'IR_Pos1bp_PNL'], 
                                index=['Sector'], aggfunc="sum")
    exposure_summary.rename(columns={'MSCIBeta1d1Y': 'Beta', 'IR_Pos1bp_PNL': 'DV01'}, inplace=True)
    currency_group_net_exp = pd.pivot_table(positions_snapshot[positions_snapshot['Sector'] == 'Currency'], 
                                values=['Net Exposure'], 
                                index=['Currency'], aggfunc="sum")
    currency_gross_exp = abs(currency_group_net_exp['Net Exposure']).sum()
    exposure_summary.loc['Cash', 'Beta'] = 0
    exposure_summary.loc['Currency', 'Gross Exposure'] = currency_gross_exp
    dollar_exposures = exposure_summary.copy()
    exposure_summary['Gross Exposure'] = exposure_summary['Gross Exposure']*100/current_trading_level
    exposure_summary['Net Exposure'] = exposure_summary['Net Exposure']*100/current_trading_level
    exposure_summary['Beta'] = exposure_summary['Beta']/current_trading_level
    exposure_summary['DV01'] = exposure_summary['DV01']/current_trading_level

    pnl_summary = pd.pivot_table(pnl_snapshot, values=['DTD', 'MTD'], index=['Sector'], aggfunc="sum")
    # january_pnl_snapshot = pd.pivot_table(january_pnl_snapshot, values=['MTD'], index=['Sector'], aggfunc="sum")
    # january_pnl_snapshot.rename(columns={'MTD': 'January MTD'}, inplace=True)
    # february_pnl_snapshot = pd.pivot_table(february_pnl_snapshot, values=['MTD'], index=['Sector'], aggfunc="sum")
    # february_pnl_snapshot.rename(columns={'MTD': 'February MTD'}, inplace=True)
    # march_pnl_snapshot = pd.pivot_table(march_pnl_snapshot, values=['MTD'], index=['Sector'], aggfunc="sum")
    # march_pnl_snapshot.rename(columns={'MTD': 'March MTD'}, inplace=True)
    # april_pnl_snapshot = pd.pivot_table(april_pnl_snapshot, values=['MTD'], index=['Sector'], aggfunc="sum")
    # april_pnl_snapshot.rename(columns={'MTD': 'April MTD'}, inplace=True)
    # may_pnl_snapshot = pd.pivot_table(may_pnl_snapshot, values=['MTD'], index=['Sector'], aggfunc="sum")
    # may_pnl_snapshot.rename(columns={'MTD': 'May MTD'}, inplace=True)
    # june_pnl_snapshot = pd.pivot_table(june_pnl_snapshot, values=['MTD'], index=['Sector'], aggfunc="sum")
    # june_pnl_snapshot.rename(columns={'MTD': 'June MTD'}, inplace=True)
    # july_pnl_snapshot = pd.pivot_table(july_pnl_snapshot, values=['MTD'], index=['Sector'], aggfunc="sum")
    # july_pnl_snapshot.rename(columns={'MTD': 'July MTD'}, inplace=True)
    # august_pnl_snapshot = pd.pivot_table(august_pnl_snapshot, values=['MTD'], index=['Sector'], aggfunc="sum")
    # august_pnl_snapshot.rename(columns={'MTD': 'August MTD'}, inplace=True)
    # september_pnl_snapshot = pd.pivot_table(september_pnl_snapshot, values=['MTD'], index=['Sector'], aggfunc="sum")
    # september_pnl_snapshot.rename(columns={'MTD': 'September MTD'}, inplace=True)
    # october_pnl_snapshot = pd.pivot_table(october_pnl_snapshot, values=['MTD'], index=['Sector'], aggfunc="sum")
    # october_pnl_snapshot.rename(columns={'MTD': 'October MTD'}, inplace=True)
    # november_pnl_snapshot = pd.pivot_table(november_pnl_snapshot, values=['MTD'], index=['Sector'], aggfunc="sum")
    # november_pnl_snapshot.rename(columns={'MTD': 'November MTD'}, inplace=True)
    # december_pnl_snapshot = pd.pivot_table(december_pnl_snapshot, values=['MTD'], index=['Sector'], aggfunc="sum")
    # december_pnl_snapshot.rename(columns={'MTD': 'December MTD'}, inplace=True)
    # january_2025_pnl_snapshot = pd.pivot_table(january_2025_pnl_snapshot, values=['MTD'], index=['Sector'], aggfunc="sum")
    # january_2025_pnl_snapshot.rename(columns={'MTD': 'January 2025 MTD'}, inplace=True)

    # prev_pnl_summary = pd.merge(january_pnl_snapshot, february_pnl_snapshot, left_index=True, right_index=True, how='outer')
    # prev_pnl_summary = pd.merge(prev_pnl_summary, march_pnl_snapshot, left_index=True, right_index=True, how='outer')
    # prev_pnl_summary = pd.merge(prev_pnl_summary, april_pnl_snapshot, left_index=True, right_index=True, how='outer')
    # prev_pnl_summary = pd.merge(prev_pnl_summary, may_pnl_snapshot, left_index=True, right_index=True, how='outer')
    # prev_pnl_summary = pd.merge(prev_pnl_summary, june_pnl_snapshot, left_index=True, right_index=True, how='outer')
    # prev_pnl_summary = pd.merge(prev_pnl_summary, july_pnl_snapshot, left_index=True, right_index=True, how='outer')
    # prev_pnl_summary = pd.merge(prev_pnl_summary, august_pnl_snapshot, left_index=True, right_index=True, how='outer')
    # prev_pnl_summary = pd.merge(prev_pnl_summary, september_pnl_snapshot, left_index=True, right_index=True, how='outer')
    # prev_pnl_summary = pd.merge(prev_pnl_summary, october_pnl_snapshot, left_index=True, right_index=True, how='outer')
    # prev_pnl_summary = pd.merge(prev_pnl_summary, november_pnl_snapshot, left_index=True, right_index=True, how='outer')
    # prev_pnl_summary = pd.merge(prev_pnl_summary, december_pnl_snapshot, left_index=True, right_index=True, how='outer')
    # prev_pnl_summary = pd.merge(prev_pnl_summary, january_2025_pnl_snapshot, left_index=True, right_index=True, how='outer')

    # pnl_summary = pd.merge(pnl_summary, prev_pnl_summary, left_index=True, right_index=True, how='outer')
    # pnl_summary['YTD'] = pnl_summary['January MTD'] + pnl_summary['February MTD'] + pnl_summary['March MTD'] + pnl_summary['April MTD'] \
    #                     + pnl_summary['May MTD'] + pnl_summary['June MTD'] + pnl_summary['July MTD'] + pnl_summary['August MTD'] \
    #                     + pnl_summary['September MTD'] + pnl_summary['October MTD'] + pnl_summary['November MTD'] + pnl_summary['December MTD']

    # pnl_summary['YTD'] = pnl_summary['January 2025 MTD']
    # pnl_summary.drop(columns=['January MTD', 'February MTD', 'March MTD', 'April MTD', 'May MTD', 'June MTD', 'July MTD', 
                            #   'August MTD', 'September MTD', 'October MTD', 'November MTD', 'December MTD', 'January 2025 MTD'], inplace=True)

    # Calculate YTD by summing DTD values for each Sector since July 1, 2024
    ytd_by_sector = pnl_history[pnl_history['PeriodEndDate'] >= '2024-07-01'].groupby('Sector')['DTD'].sum()
    # Merge the YTD values into pnl_summary
    pnl_summary['YTD'] = pnl_summary.index.map(lambda x: ytd_by_sector.get(x, 0))
    dollar_pnl = pnl_summary.copy()
    pnl_summary = pnl_summary*100/current_trading_level
    pm_summary = pd.merge(exposure_summary, pnl_summary, left_index=True, right_index=True, how='outer')
    pm_dollar_summary = pd.merge(dollar_exposures, dollar_pnl, left_index=True, right_index=True, how='outer')
    pm_summary.fillna(0, inplace=True)
    pm_dollar_summary.fillna(0, inplace=True)

    st.markdown("<h2 style='text-align: center;'>Exposure Summary By Sector</h2>", unsafe_allow_html=True)
    exp_summary_df = pd.merge(pm_summary, pm_dollar_summary, left_index=True, right_index=True, 
                            how='inner', suffixes=(' (%)', ' ($)'))
    exp_summary_df = exp_summary_df.reindex(columns=['Gross Exposure (%)', 'Gross Exposure ($)', 'Net Exposure (%)', 'Net Exposure ($)',
                                'DV01 (%)', 'DV01 ($)', 'Beta (%)', 'Beta ($)', 'DTD (%)', 'DTD ($)', 'MTD (%)', 'MTD ($)',
                                'YTD (%)', 'YTD ($)'])
    exp_summary_df.drop(columns={'DV01 (%)', 'Beta ($)'}, inplace=True)
    exp_summary_df.rename(columns={'Beta (%)': 'Beta'}, inplace=True)
    exp_summary_df.loc['Total',:] = exp_summary_df.sum(axis=0)
    exp_summary_df.reset_index(inplace=True)

    format_mapping = {"Gross Exposure (%)": "{:.2f}%", "Net Exposure (%)": "{:.2f}%",
                    "Beta": "{:.2f}", "DV01 (%)": "{:.2f}",
                    "MTD (%)": "{:.2f}%", "DTD (%)": "{:.2f}%", "YTD (%)": "{:.2f}%"}

    for col in exp_summary_df.columns[1:]:
        if '($)' in col:
            exp_summary_df[col] = exp_summary_df[col].apply(lambda x: get_currency_format(x))
        else:
            exp_summary_df[col] = exp_summary_df[col].apply(lambda x: format_mapping[col].format(x))

    headers = [['','<b>'+c+'</b>'] for c in exp_summary_df.columns]
    headers[5][0], headers[6][0], headers[7][0] = '<b>Exposure</b>', '<b>Summary</b>', '<b>Table</b>'

    bold_total = []
    last_index = len(exp_summary_df) - 1
    for x in exp_summary_df.tail(1).values:
        tl = '<b>'+x+'</b>'
        bold_total.append(tl)
    exp_summary_df.loc[last_index, :] = bold_total[0]

    values = []
    for col in exp_summary_df.columns:
        values.append(exp_summary_df[col])

    topHeaderColor = '#1A3357'
    nextHeaderColor = '#1A9687'

    rowOddColor = '#F2F2F2'
    rowEvenColor = '#FFFFFF'
    fig = go.Figure(
        go.Table(
            header=dict(values=headers,
                        line = dict(width=0),
                    fill_color=[[topHeaderColor, nextHeaderColor]*8],
                        align='center',
                        font=dict(family="Aptos Narrow", color="white", size=14)
                    ),
            cells=dict(values=values,
                    line = dict(width=0),
                    align='center',
                    font=dict(family="Segoe UI", color="black", size=12),
                    fill_color = [[rowOddColor, rowEvenColor,rowOddColor, rowEvenColor,rowOddColor, rowEvenColor]*8,
                                ],
                    height=30)
        )
    )
    # Set the figure layout to ensure it takes up the full width
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<h2 style='text-align: center;'>Exposure Summary By Market</h2>", unsafe_allow_html=True)

    #graph_exp_df = positions_snapshot[positions_snapshot['Sector'] != 'Cash']
    currency_exp_df = positions_snapshot[positions_snapshot['Sector'] == 'Currency']
    graph_exp_df = positions_snapshot[~positions_snapshot['Sector'].isin(['Currency', 'Cash'])]
    currency_exp_df['List Check'] = currency_exp_df['InstrumentName'].apply(lambda x: x.split())
    currency_exp_df['List Check'] = currency_exp_df['List Check'].apply(lambda x: sorted(x))
    currency_exp_df['List Check'] = currency_exp_df['List Check'].apply(lambda x: ' '.join(x))
    temp = pd.pivot_table(currency_exp_df, values=['Net Exposure'], index=['List Check'], aggfunc="sum")
    temp.reset_index(inplace=True)
    currency_exp_df.drop_duplicates(subset=['List Check'], inplace=True)
    currency_exp_df.drop(columns=['Net Exposure'], inplace=True)
    currency_exp_df = pd.merge(currency_exp_df, temp, left_on=['List Check'], right_on=['List Check'], how='inner')
    currency_exp_df['Gross Exposure'] = abs(currency_exp_df['Net Exposure'])
    currency_exp_df.drop(columns=['List Check'], inplace=True)
    graph_exp_df = pd.concat([graph_exp_df, currency_exp_df])
    graph_exp_df['Gross %'] = graph_exp_df['Gross Exposure']/current_trading_level
    graph_exp_df['Net %'] = graph_exp_df['Net Exposure']/current_trading_level
    temp = graph_exp_df[['Sector','Position Desc','Gross Exposure', 'Gross %', 'Net Exposure', 'Net %']]
    temp = temp.groupby(['Sector','Position Desc']).sum()
    temp['Gross Exposure'] = temp['Gross Exposure'].apply('${:,.2f}'.format)
    temp['Net Exposure'] = temp['Net Exposure'].apply('${:,.2f}'.format)
    temp.reset_index(inplace=True)
    x_axis = [temp['Sector'].tolist(), temp['Position Desc'].tolist()]
    temp.set_index(['Sector', 'Position Desc'], inplace=True)
    hover_gross_exposure = temp[temp['Gross %'] != 0]['Gross Exposure']
    hover_net_exposure = temp[temp['Net %'] != 0]['Net Exposure']

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=x_axis,
        y=temp['Gross %'],
        histfunc="sum",
        hovertext=hover_gross_exposure,
        hovertemplate='Market: %{x}' + '<br>Gross: %{y}' + '<br>Gross ($): %{hovertext}',
        name='Gross %', # name used in legend and hover labels
        marker_color='#1A3357',
        opacity=0.75,
    ))
    fig.add_trace(go.Histogram(
        x=x_axis,
        y=temp['Net %'],
        histfunc="sum",
        hovertext=hover_net_exposure,
        hovertemplate='Market: %{x}' + '<br>Net: %{y}' + '<br>Net ($): %{hovertext}',
        name='Net %',
        marker_color='#1A9687',
        opacity=0.75
    ))

    fig.update_layout(
        title_text='Gross and Net Exposures By Market', # title of plot
        xaxis_title_text='Market', # xaxis label
        yaxis_title_text='Exposure %', # yaxis label
        bargap=0.2, # gap between bars of adjacent location coordinates
        bargroupgap=0.1, # gap between bars of the same location coordinates
        template='plotly_white', yaxis_tickformat=',.0%', 
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    non_currency_positions_snapshot = positions_snapshot[positions_snapshot['Sector'] != 'Currency']
    non_currency_positions_snapshot = non_currency_positions_snapshot[non_currency_positions_snapshot['Sector'] != 'Cash']
    non_currency_positions_snapshot = non_currency_positions_snapshot.loc[:, ['Sector', 'MarketGroup','Investment', 'InstrumentName', 'Net Exposure', 'Gross Exposure', 'MSCIBeta1d1Y', 'IR_Pos1bp_PNL']].groupby(['Investment', 'InstrumentName', 'Sector', 'MarketGroup']).sum()
    non_currency_positions_snapshot.rename(columns={'MSCIBeta1d1Y': 'Beta', 'IR_Pos1bp_PNL': 'DV01'}, inplace=True)
    currency_frwrd_positions_snapshot = positions_snapshot[positions_snapshot['InstrumentType'] == 'Foreign Exchange Forward']
    currency_frwrd_positions_snapshot = currency_frwrd_positions_snapshot.loc[:, ['Sector', 'MarketGroup','Currency', 'Net Exposure', 'MSCIBeta1d1Y', 'IR_Pos1bp_PNL']].groupby(['Currency', 'Sector', 'MarketGroup']).sum()
    currency_frwrd_positions_snapshot['Gross Exposure'] = abs(currency_frwrd_positions_snapshot['Net Exposure'])
    currency_frwrd_positions_snapshot.reset_index(inplace=True)
    currency_frwrd_positions_snapshot['InstrumentName'] = currency_frwrd_positions_snapshot['Currency'].apply(lambda x: x + ' Forward')
    currency_frwrd_positions_snapshot['Currency'] = currency_frwrd_positions_snapshot['Currency'].apply(lambda x: x+'F')
    currency_frwrd_positions_snapshot.rename(columns={'MSCIBeta1d1Y': 'Beta', 'IR_Pos1bp_PNL': 'DV01', 'Currency': 'Investment'}, inplace=True)
    currency_frwrd_positions_snapshot.set_index(['Investment', 'InstrumentName', 'Sector', 'MarketGroup'], inplace=True)
    currency_option_positions_snapshot = positions_snapshot[positions_snapshot['InstrumentType'].isin(['Foreign Exchange Option', 'FX Digital Option'])]
    currency_option_positions_snapshot = currency_option_positions_snapshot.loc[:, ['Investment', 'InstrumentName', 'Sector', 'MarketGroup', 'Net Exposure', 'Gross Exposure', 'MSCIBeta1d1Y', 'IR_Pos1bp_PNL']].groupby(['Investment', 'InstrumentName', 'Sector', 'MarketGroup']).sum()
    currency_option_positions_snapshot['Gross Exposure'] = abs(currency_option_positions_snapshot['Net Exposure'])
    currency_option_positions_snapshot.rename(columns={'MSCIBeta1d1Y': 'Beta', 'IR_Pos1bp_PNL': 'DV01'}, inplace=True)
    cash_positions_snapshot = positions_snapshot[positions_snapshot['Sector'] == 'Cash']
    cash_positions_snapshot = cash_positions_snapshot.loc[:, ['Investment', 'InstrumentName', 'Sector', 'MarketGroup', 'Net Exposure', 'Gross Exposure', 'MSCIBeta1d1Y', 'IR_Pos1bp_PNL']].groupby(['Investment', 'InstrumentName', 'Sector', 'MarketGroup']).sum()
    cash_positions_snapshot['Gross Exposure'] = abs(cash_positions_snapshot['Net Exposure'])
    cash_positions_snapshot.rename(columns={'MSCIBeta1d1Y': 'Beta', 'IR_Pos1bp_PNL': 'DV01'}, inplace=True)
    instrument_exposures = pd.concat([non_currency_positions_snapshot, currency_frwrd_positions_snapshot, currency_option_positions_snapshot, cash_positions_snapshot])
    instrument_exposures.reset_index(inplace=True)
    instrument_exposures.rename(columns={'Investment': 'AdminInstrumentId'}, inplace=True)
    instrument_exposures.set_index(['AdminInstrumentId','Sector', 'MarketGroup'], inplace=True)
    #pnl_snapshot = pnl_snapshot[pnl_snapshot['Sector'] != 'Commission & Fees']
    instrument_pnl = pnl_snapshot.loc[:, ['AdminInstrumentId','Sector', 'MarketGroup', 'MTD', 'DTD']].groupby(['AdminInstrumentId', 'Sector', 'MarketGroup']).sum()

    positions_summary = pd.merge(instrument_exposures, instrument_pnl, left_index=True, right_index=True, how='outer')
    positions_summary.reset_index(inplace=True)
    positions_summary.set_index(['InstrumentName', 'AdminInstrumentId','Sector', 'MarketGroup'], inplace=True)
    positions_summary['Beta'] = positions_summary['Beta']/100
    positions_summary.loc['Total',:] = positions_summary.sum(axis=0).values

    for col in positions_summary.columns:
        positions_summary[col+ ' (%)'] = positions_summary[col]*100/current_trading_level
    positions_summary.reset_index(inplace=True)
    positions_summary['InstrumentName'] = positions_summary.apply(lambda x: closed_positions(x['InstrumentName'], x['AdminInstrumentId'], x['Sector']), axis=1)
    positions_summary.fillna(0, inplace=True)
    positions_summary.drop(columns=['AdminInstrumentId', 'DV01 (%)', 'Beta'], inplace=True)
    format_mapping = {"Gross Exposure (%)": "{:.2f}%", "Net Exposure (%)": "{:.2f}%",
                    "Beta (%)": "{:.2f}", "DV01 (%)": "{:.2f}",
                    "MTD (%)": "{:.2f}%", "DTD (%)": "{:.2f}%", "YTD (%)": "{:.2f}%"}

    positions_summary['Sector'] = positions_summary.apply(lambda x: 'Total' if x['InstrumentName'] == 'Total' else x['Sector'], axis=1)
    positions_summary['MarketGroup'] = positions_summary.apply(lambda x: 'Total' if x['InstrumentName'] == 'Total' else x['MarketGroup'], axis=1)
    for col in positions_summary.columns[3:]:
        if '(%)' in col:
            positions_summary[col] = positions_summary[col].apply(lambda x: format_mapping[col].format(x))
        else:
            positions_summary[col] = positions_summary[col].apply(lambda x: get_currency_format(x))
    positions_summary.rename(columns={'Beta (%)': 'Beta'}, inplace=True)

    positions_summary = positions_summary.reindex(columns=['Sector','MarketGroup','InstrumentName',
                                    'Gross Exposure','Gross Exposure (%)', 'Net Exposure', 'Net Exposure (%)', 
                                    'DV01', 'Beta',  'DTD', 'DTD (%)', 'MTD', 'MTD (%)'])
    positions_summary.sort_values(['Sector', 'MarketGroup'], inplace=True, ignore_index=True)

    headers = [['','<b>'+c+'</b>'] for c in positions_summary.columns]
    headers[5][0], headers[6][0], headers[7][0] = '<b>Position</b>', '<b>Summary</b>', '<b>Table</b>'

    last_index = len(positions_summary) - 1
    bold_total = []
    for x in positions_summary.tail(1).values:
        tl = '<b>'+x+'</b>'
        bold_total.append(tl)
    positions_summary.loc[last_index, :] = bold_total[0]

    values = []
    for col in positions_summary.columns:
        values.append(positions_summary[col])

    st.markdown("<h2 style='text-align: center;'>Position Summary</h2>", unsafe_allow_html=True)

    topHeaderColor = '#1A3357'
    nextHeaderColor = '#1A9687'

    rowOddColor = '#F2F2F2'
    rowEvenColor = '#FFFFFF'
    fig = go.Figure(
        go.Table(
            header=dict(values=headers,
                        line = dict(width=0),
                    fill_color=[[topHeaderColor, nextHeaderColor]*8],
                        align='center',
                        font=dict(family="Aptos Narrow", color="white", size=14)
                    ),
            cells=dict(values=values,
                    line = dict(width=0),
                    align='center',
                    font=dict(family="Segoe UI", color="black", size=12),
                    fill_color = [[rowOddColor, rowEvenColor,rowOddColor, rowEvenColor,rowOddColor, rowEvenColor]*8,
                                ],
                    height=30)
        )
    )
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<h2 style='text-align: center;'>VAR Summary</h2>", unsafe_allow_html=True)
    ivar_sector_summary = pd.pivot_table(var_snapshot, values=['IVaR_Hist_95','VaR_Hist_95', 'IVaR_Hist_99', 'VaR_Hist_99'], 
                index=['Sector'], aggfunc="sum")
    ivar_sector_summary.loc['Total', :] = ivar_sector_summary.sum(axis=0)
    for col in ivar_sector_summary.columns:
        ivar_sector_summary[col + ' %'] = ivar_sector_summary[col]/current_trading_level
    for col in ivar_sector_summary.columns[:-4]:
        ivar_sector_summary[col] = ivar_sector_summary[col].apply(lambda x: get_currency_format(x))


    ivar_market_summary = pd.pivot_table(var_snapshot, values=['IVaR_Hist_95','VaR_Hist_95', 'IVaR_Hist_99', 'VaR_Hist_99'], 
                index=['Market'], aggfunc="sum")
    for col in ivar_market_summary.columns:
        ivar_market_summary[col + ' %'] = ivar_market_summary[col]/current_trading_level
    for col in ivar_market_summary.columns[:-4]:
        ivar_market_summary[col] = ivar_market_summary[col].apply(lambda x: get_currency_format(x))

    fig = make_subplots(rows=1, cols=2,
                    subplot_titles=['IVAR 95% By Sector','IVAR 99% By Sector'])

    # VAR By Sector

    # fig.add_trace(go.Histogram(
    #     x=ivar_sector_summary.index.values,
    #     y=ivar_sector_summary['VaR_Hist_95 %'],
    #     histfunc="sum",
    #     hovertext=ivar_sector_summary['VaR_Hist_95'],
    #     hovertemplate='Sector: %{x}' + '<br>VAR 95: %{y}' + '<br>VAR 95 ($): %{hovertext}',
    #     name='VAR 95 %', # name used in legend and hover labels
    #     marker_color='#1A3357',
    #     opacity=0.75,
    # ), row=1, col=1)

    # fig.add_trace(go.Histogram(
    #     x=ivar_sector_summary.index.values,
    #     y=ivar_sector_summary['VaR_Hist_99 %'],
    #     histfunc="sum",
    #     hovertext=ivar_sector_summary['VaR_Hist_99'],
    #     hovertemplate='Sector: %{x}' + '<br>VAR 99: %{y}' + '<br>VAR 99 ($): %{hovertext}',
    #     name='VAR 99 %',
    #     marker_color='#1A9687',
    #     opacity=0.75
    # ),  row=1, col=1)


    # IVAR By Sector
    fig.add_trace(go.Histogram(
        x=ivar_sector_summary.index.values,
        y=ivar_sector_summary['IVaR_Hist_95 %'],
        histfunc="sum",
        hovertext=ivar_sector_summary['IVaR_Hist_95'],
        hovertemplate='Sector: %{x}' + '<br>IVAR 95: %{y}' + '<br>IVAR 95 ($): %{hovertext}',
        name='IVAR 95 %', # name used in legend and hover labels
        marker_color='#1A3357',
        opacity=0.75,
    ), row=1, col=1)

    fig.add_trace(go.Histogram(
        x=ivar_sector_summary.index.values,
        y=ivar_sector_summary['IVaR_Hist_99 %'],
        histfunc="sum",
        hovertext=ivar_sector_summary['IVaR_Hist_99'],
        hovertemplate='Sector: %{x}' + '<br>IVAR 99: %{y}' + '<br>IVAR 99 ($): %{hovertext}',
        name='IVAR 99 %',
        marker_color='#1A9687',
        opacity=0.75
    ),  row=1, col=2)


    # Update Layout
    fig.update_layout(
        #title_text='IVAR By Sector', # title of plot
        #xaxis_title_text='Sector', # xaxis label
        #yaxis_title_text='IVAR 95%', # yaxis label
        bargap=0.2, # gap between bars of adjacent location coordinates
        bargroupgap=0.1, # gap between bars of the same location coordinates
        template='plotly_white', yaxis_tickformat=',.0%', height=500,
        showlegend=False,
    )

    fig.update_yaxes(tickformat=".2%")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<h2 style='text-align: center;'>Stress Tests Snapshot</h2>", unsafe_allow_html=True)

    stress_test_snapshot = positions_snapshot[["PositionAsOfDate", "Sector", "Market", "MarketGroup", "BlackMonday1987", "SubprimeDebacle", "GulfWar1990", 
                                        "RateRise94", "MexicanPesoCrisis1995", "AsianCrisis1997", 
                                        "September11th2001Attack", "HedgeFundSelloff_010172018_11202018", 
                                        "LehmanBankruptcy", "QuantLiquidation_08072007_08092007_Hist", 
                                        "EmergingMarketSellOff_05012006_06082006", 
                                        "USDowngradeEuropeanCrisis_072511_092222"]]
    sc_dict = {
        "BlackMonday1987": "Black Monday 1987", 
        "SubprimeDebacle": "Subprime Debacle", 
        "GulfWar1990": "Gulf War 1990", 
        "RateRise94": "Rate Rise 94", "MexicanPesoCrisis1995": "Mexican Peso Crisis 1995", 
        "AsianCrisis1997": "Asian Crisis 1997", 
        "September11th2001Attack": "September 11th, 2001 Attack", 
        "HedgeFundSelloff_010172018_11202018": "Hedge Fund Selloff 2018", 
        "LehmanBankruptcy": "Lehman Bankruptcy", 
        "QuantLiquidation_08072007_08092007_Hist": "Quant Liquidation 2007", 
        "EmergingMarketSellOff_05012006_06082006": "Emerging Market SellOff 2006", 
        "USDowngradeEuropeanCrisis_072511_092222": "US Downgrade European Crisis"
    }

    stress_test_snapshot.rename(columns=sc_dict, inplace=True)
    scenarios = stress_test_snapshot.columns.tolist()[4:]

    stress_test_df = pd.pivot_table(stress_test_snapshot, values=scenarios, index=['Sector'], aggfunc="sum")
    stress_test_df = stress_test_df.T
    stress_test_df.reset_index(inplace=True)
    stress_test_df.rename(columns={'index': 'Scenario'}, inplace=True)
    stress_test_df['Total'] = stress_test_df[stress_test_df.columns[1:]].sum(axis=1)

    #fig = go.Figure()
    sector_color_map = {'Cash': '#1A9687', 'Currency': '#1A3357', 'Fixed Income': '#23CAB8', 
                        'Equity': '#03ADC9', 'Commodity': '#12D0A3', 'Unknown': 'black'}

    fig = px.bar(stress_test_df, x=stress_test_df.Scenario, y=[c for c in stress_test_df.columns[1:-1]],
                color_discrete_sequence=[sector_color_map[c] for c in stress_test_df.columns[1:-1]])

    fig.add_trace(go.Scatter(mode='markers',
                            x=stress_test_df.Scenario,
                            y=stress_test_df.Total,
                            #text=[str(x) for x in stress_test_df['Total'].tolist()],
                            #textposition='top center',
                            marker_color='grey',
                            marker_symbol='circle',
                            #showlegend=False,
                            name='Total'
                            ))

    fig.update_layout(
        title_text='Stress Test', # title of plot
        xaxis_title_text='Stress Scenario', # xaxis label
        yaxis_title_text='Portfolio ($)', # yaxis label
        bargap=0.55, # gap between bars of adjacent location coordinates
        bargroupgap=0.1, # gap between bars of the same location coordinates
        template='plotly_white', yaxis_tickprefix = '$', yaxis_tickformat = ',.0f', height=700,
        legend_title_text='Sector'
    )

    st.plotly_chart(fig, use_container_width=True)

    fi_shocks, comm_shocks, eq_shocks, fx_shocks = [],[],[],[]
    for col in positions_snapshot.columns:
        if 'IR_Neg' in col or 'IR_Pos' in col:
            fi_shocks.append(col)
        elif 'Commodity_' in col:
            comm_shocks.append(col)
        elif 'Equity_' in col:
            eq_shocks.append(col)
        elif 'FXUSD_' in col:
            fx_shocks.append(col)

    fi_df = positions_snapshot[fi_shocks + ['PositionAsOfDate', 'Sector', 'Market']]
    fi_pivot_df = pd.pivot_table(fi_df, values=fi_shocks, index=['PositionAsOfDate'], aggfunc="sum")
    fi_stress_df = fi_pivot_df.T*100/current_trading_level
    fi_stress_df = fi_pivot_df.T
    fi_stress_df.rename(columns={fi_stress_df.columns[0]: 'Portfolio ($)'}, inplace=True)
    fi_stress_df.reset_index(inplace=True)
    fi_stress_df['Type (BPS)'] = fi_stress_df['index'].apply(lambda x: x.split('_')[1])
    fi_stress_df['Change (BPS)'] = fi_stress_df['Type (BPS)'].apply(lambda x: convert_to_bps(x))
    fi_stress_df['Shock (%)'] = fi_stress_df['Change (BPS)'].apply(lambda x: x/100)
    fi_stress_df = fi_stress_df.sort_values(by=['Change (BPS)'])
    fi_stress_df['Sector'] = 'Rates'
    fi_stress_df['Portfolio ($)'] = fi_stress_df['Portfolio ($)']*2
    fi_stress_df['Portfolio Change(%)'] = fi_stress_df['Portfolio ($)']*100/current_trading_level
    fi_stress_df.rename(columns={'index': 'IR Shock'}, inplace=True)

    comm_df = positions_snapshot[comm_shocks + ['PositionAsOfDate', 'Sector', 'Market']]
    comm_pivot_df = pd.pivot_table(comm_df, values=comm_shocks, index=['PositionAsOfDate'], aggfunc="sum")
    comm_stress_df = comm_pivot_df.T
    comm_stress_df.rename(columns={comm_stress_df.columns[0]: 'Portfolio ($)'}, inplace=True)
    comm_stress_df.reset_index(inplace=True)
    comm_stress_df['Type (BPS)'] = comm_stress_df['index'].apply(lambda x: x.split('_')[1])
    comm_stress_df['Change (BPS)'] = comm_stress_df['Type (BPS)'].apply(lambda x: convert_to_bps(x, 'Commodity'))
    comm_stress_df['Shock (%)'] = comm_stress_df['Change (BPS)'].apply(lambda x: x)
    comm_stress_df = comm_stress_df.sort_values(by=['Change (BPS)'])
    comm_stress_df['Sector'] = 'Commodity'
    comm_stress_df['Portfolio ($)'] = comm_stress_df['Portfolio ($)']*2
    comm_stress_df['Portfolio Change(%)'] = comm_stress_df['Portfolio ($)']*100/current_trading_level
    comm_stress_df.rename(columns={'index': 'IR Shock'}, inplace=True)

    eq_df = positions_snapshot[eq_shocks + ['PositionAsOfDate', 'Sector', 'Market']]
    eq_pivot_df = pd.pivot_table(eq_df, values=eq_shocks, index=['PositionAsOfDate'], aggfunc="sum")
    eq_stress_df = eq_pivot_df.T
    eq_stress_df.rename(columns={eq_stress_df.columns[0]: 'Portfolio ($)'}, inplace=True)
    eq_stress_df.reset_index(inplace=True)
    eq_stress_df['Type (BPS)'] = eq_stress_df['index'].apply(lambda x: x.split('_')[1])
    eq_stress_df['Change (BPS)'] = eq_stress_df['Type (BPS)'].apply(lambda x: convert_to_bps(x, 'Equity'))
    eq_stress_df['Shock (%)'] = eq_stress_df['Change (BPS)'].apply(lambda x: x)
    eq_stress_df = eq_stress_df.sort_values(by=['Change (BPS)'])
    eq_stress_df['Sector'] = 'Equity'
    eq_stress_df['Portfolio ($)'] = eq_stress_df['Portfolio ($)']*2
    eq_stress_df['Portfolio Change(%)'] = eq_stress_df['Portfolio ($)']*100/current_trading_level
    eq_stress_df.rename(columns={'index': 'IR Shock'}, inplace=True)

    fx_df = positions_snapshot[fx_shocks + ['PositionAsOfDate', 'Sector', 'Market']]
    fx_pivot_df = pd.pivot_table(fx_df, values=fx_shocks, index=['PositionAsOfDate'], aggfunc="sum")
    fx_stress_df = fx_pivot_df.T
    fx_stress_df.rename(columns={fx_stress_df.columns[0]: 'Portfolio ($)'}, inplace=True)
    fx_stress_df.reset_index(inplace=True)
    fx_stress_df['Type (BPS)'] = fx_stress_df['index'].apply(lambda x: x.split('_')[1])
    fx_stress_df['Change (BPS)'] = fx_stress_df['Type (BPS)'].apply(lambda x: convert_to_bps(x, 'Equity'))
    fx_stress_df['Shock (%)'] = fx_stress_df['Change (BPS)'].apply(lambda x: x)
    fx_stress_df = fx_stress_df.sort_values(by=['Change (BPS)'])
    fx_stress_df['Sector'] = 'FX'
    fx_stress_df['Portfolio ($)'] = fx_stress_df['Portfolio ($)']*2
    fx_stress_df['Portfolio Change(%)'] = fx_stress_df['Portfolio ($)']*100/current_trading_level
    fx_stress_df.rename(columns={'index': 'IR Shock'}, inplace=True)

    fig = make_subplots(rows=2, cols=2,
                    subplot_titles=['Rates', 'Commodity', 'Equity', 'FX'],
                    x_title='Shock (%)',
                        y_title='Portfolio Change(%)')

    fig.add_trace(
        go.Scatter(x=fi_stress_df['Shock (%)'], y=fi_stress_df['Portfolio Change(%)'], name='Rates', 
                text=fi_stress_df['Portfolio ($)'],
                hovertemplate='Shock: %{x}%' + '<br>Portfolio Change: %{y:.2f}%' + '<br>Portfolio ($): $%{text:,.2f}',
                mode='lines', marker_color='#23CAB8'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=comm_stress_df['Shock (%)'], y=comm_stress_df['Portfolio Change(%)'], name='Commodity',
                text=comm_stress_df['Portfolio ($)'],
                hovertemplate='Shock: %{x}%' + '<br>Portfolio Change: %{y:.2f}%' + '<br>Portfolio ($): $%{text:,.2f}',
                mode='lines', marker_color='#1A3357'),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=eq_stress_df['Shock (%)'], y=eq_stress_df['Portfolio Change(%)'], name='Equity', 
                text=eq_stress_df['Portfolio ($)'],
                hovertemplate='Shock: %{x}%' + '<br>Portfolio Change: %{y:.2f}%' + '<br>Portfolio ($): $%{text:,.2f}',
                mode='lines', marker_color='#1A9687'),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=fx_stress_df['Shock (%)'], y=fx_stress_df['Portfolio Change(%)'], name='FX',
                text=fx_stress_df['Portfolio ($)'],
                hovertemplate='Shock: %{x}%' + '<br>Portfolio Change: %{y:.2f}%' + '<br>Portfolio ($): $%{text:,.2f}',
                mode='lines', marker_color='#03ADC9'),
        row=2, col=2
    )


    fig.update_layout(height=600, title_text="Portfolio Performance Under Stress Shocks - By Sector",
                    template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<h2 style='text-align: center;'>Historical Performance Statistics</h2>", unsafe_allow_html=True)

    returns_df = get_pm_returns(csv_dataframes)
    benchmarks = csv_dataframes['Benchmarks.csv']
    benchmarks["Date"] = pd.to_datetime(benchmarks['Date'].astype(str), format='%Y%m%d')
    benchmarks_df = benchmarks.pivot(index='Date', columns='BBG Ticker', values='PX_LAST').div(100)
    benchmarks_df.rename(columns={"CO1 Comdty": "WTI Crude Oil", "CRY Index": "CRB", "DXY Index": "DXY",
                                "GC1 Comdty": "Gold", "SPX Index": "S&P 500", "TY1 Comdty": "US Treasury Note, 10Yr"}, 
                        inplace=True)

    #get MAP240 manager returns
    gt_table = pd.merge(returns_df, benchmarks_df, on='Date', how='left')
    #combine MAP240 manager returns with prospect's return 
    gt_table.set_index('Date',inplace=True)
    gt_table.ffill(inplace=True)
    gt_table.dropna(inplace=True)

    rolling_returns(gt_table,22)
    rolling_returns(gt_table,66)
    rolling_std(gt_table,22)
    rolling_std(gt_table,66)
    rolling_Sharpe_Ratio(gt_table, 22)
    rolling_Sharpe_Ratio(gt_table, 66)
    #x = rolling_Sharpe_Ratio(gt_table, 22)
    x = drawdown(gt_table)

    for i in range(1,9):
        rolling_corr(gt_table,22,gt_table[gt_table.columns[i]])
        rolling_corr(gt_table,66,gt_table[gt_table.columns[i]])
    for i in range(1,9):
        rolling_beta(gt_table,22,gt_table[gt_table.columns[i]])
        rolling_beta(gt_table,66,gt_table[gt_table.columns[i]])

    fig = make_subplots(rows=3, cols=1,
                    subplot_titles=['Daily Returns', 'Cumulative Returns', 'Drawdown'],
                        y_title='Return(%)')

    fig.add_trace(
        go.Bar(
            x=gt_table.index.values, y=gt_table['MKR - MAP 240_27'],
            marker_color='#1A3357', name='Daily Return (%)'
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(x=gt_table.index.values, y=gt_table['Cumulative Returns'], 
                hovertemplate='Date: %{x}' + '<br>Cumulative Return: %{y}',
                mode='lines', marker_color='#1A3357', name='Cumulative Return (%)'),
        row=2, col=1,
    )

    fig.add_trace(
        go.Scatter(x=gt_table.index.values, y=gt_table['Drawdown'], fill='tozeroy', 
                marker_color='#1A3357', name='Drawdown (%)'),
        row=3, col=1,
    )
    fig.update_yaxes(tickformat=".2%")
    fig.update_layout(height=1500, #title_text="Portfolio Performance Under Stress Shocks - By Sector",
                    template='plotly_white', showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

    fig = make_subplots(rows=3, cols=1,
                    subplot_titles=['Historical Rolling Returns',
                                    'Historical Rolling Volatility',
                                    'Rolling Sharpe Ratio'],)

    fig.add_trace(
        go.Scatter(
            x=gt_table['22 Days Rolling Return'].dropna().index.values, y=gt_table['22 Days Rolling Return'].dropna(),
            marker_color='#1A3357',mode='lines', name='22D Rolling Return', legendgroup='1',
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=gt_table['66 Days Rolling Return'].dropna().index.values, y=gt_table['66 Days Rolling Return'].dropna(),
            marker_color='#1A9687',mode='lines', name='66D Rolling Return', legendgroup='1',
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=gt_table['22 Days Rolling STD'].dropna().index.values, y=gt_table['22 Days Rolling STD'].dropna(),
            marker_color='#1A3357',mode='lines', name='22D Rolling Volatility', legendgroup='2',
        ),
        row=2, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=gt_table['66 Days Rolling STD'].dropna().index.values, y=gt_table['66 Days Rolling STD'].dropna(),
            marker_color='#1A9687',mode='lines', name='66D Rolling Volatility', legendgroup='2',
        ),
        row=2, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=gt_table['22 Days Rolling SR'].dropna().index.values, y=gt_table['22 Days Rolling SR'].dropna(),
            marker_color='#1A3357',mode='lines', name='22D Rolling Sharpe', legendgroup='3',
        ),
        row=3, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=gt_table['66 Days Rolling SR'].dropna().index.values, y=gt_table['66 Days Rolling SR'].dropna(),
            marker_color='#1A9687',mode='lines', name='66D Rolling Sharpe', legendgroup='3',
        ),
        row=3, col=1,
    )

    fig.update_yaxes(tickformat=".2%", row=1, col=1)
    fig.update_yaxes(tickformat=".2%", row=1, col=2)
    fig.update_yaxes(tickformat=".2%", row=2, col=1)
    fig.update_yaxes(tickformat=".2%", row=2, col=2)
    fig.update_yaxes(tickformat=".2", row=3, col=1)
    fig.update_yaxes(tickformat=".2", row=3, col=1)
    fig.update_layout(height=1500, #title_text="Rolling Returns, Std and Sharpe Ratio",
                    template='plotly_white', showlegend=True, legend_tracegroupgap=500)

    st.plotly_chart(fig, use_container_width=True)

    corr_col_22d, corr_col_66d = [], []
    for col in gt_table.columns:
        if '22 Days Rolling Correlation' in col:
            corr_col_22d.append(col)
        elif '66 Days Rolling Correlation' in col:
            corr_col_66d.append(col)

    fig = make_subplots(rows=2, cols=1,
                    subplot_titles=['Rolling 22D Correlation With Markets', 'Rolling 66D Correlation With Markets'])
    market_color_map = {'WTI Crude Oil': '#1A9687', 'Gold': '#1A3357', 'CRB': '#23CAB8', 
                        'DXY': '#03ADC9', 'S&P 500': '#12D0A3', 'US Treasury Note, 10Yr': '#04CDF1'}
    for col in corr_col_22d[:-2]:
        market = col.split('22 Days Rolling Correlation with')[1].strip(' ')
        fig.add_trace(
            go.Scatter(
                x=gt_table[col].dropna().index.values, y=gt_table[col].dropna(),
                mode='lines', name=col, hovertemplate='Date: %{x}' + '<br>Correlation: %{y:.2f}',
                marker_color=market_color_map[market], legendgroup = '1',
            ),
            row=1, col=1,
        )

    for col in corr_col_66d[:-2]:
        market = col.split('66 Days Rolling Correlation with')[1].strip(' ')
        fig.add_trace(
            go.Scatter(
                x=gt_table[col].dropna().index.values, y=gt_table[col].dropna(),
                mode='lines', name=col, hovertemplate='Date: %{x}' + '<br>Correlation: %{y:.2f}',
                marker_color=market_color_map[market], legendgroup = '2',
            ),
            row=2, col=1,
        )

    #fig.update_yaxes(tickformat=".2%")
    fig.update_layout(height=1000,
                    template='plotly_white', showlegend=True, legend_tracegroupgap=400)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<h2 style='text-align: center;'>Historical Summary</h2>", unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center;'>Historical Cumulative Performance Summary</h3>", unsafe_allow_html=True)
    # st.write count of the types of the PeriodEndDate column in pnl_history
    pnl_history['PeriodEndDate'] = pnl_history['PeriodEndDate'].apply(standardize_dates)
    pnl_aggregate_history = pd.pivot_table(pnl_history, values=['MTD', 'DTD'], index=['PeriodEndDate'], aggfunc="sum")
    pnl_aggregate_history['ITD'] = pnl_aggregate_history['DTD'].cumsum()
    pnl_aggregate_history.reset_index(inplace=True)
    pnl_aggregate_history.rename(columns={'PeriodEndDate': 'Date'}, inplace=True)
    pnl_aggregate_history['Date'] = pd.to_datetime(pnl_aggregate_history['Date'], format="mixed")
    # pnl_aggregate_history['Date'] = pnl_aggregate_history['Date'].apply(standardize_dates)
    pnl_aggregate_history['Mon-Year'] = pnl_aggregate_history['Date'].apply(lambda x: x.strftime("%B-%Y"))
    # trading_level_history_df['AsOf'] = pd.to_datetime(trading_level_history_df['AsOf'], format="%Y-%m-%d")
    trading_level_history_df['AsOf'] = pd.to_datetime(trading_level_history_df['AsOf'], format="mixed")

    trading_level_history_df['Mon-Year'] = trading_level_history_df['AsOf'].apply(lambda x: x.strftime("%B-%Y"))
    pnl_aggregate_history = pd.merge(pnl_aggregate_history, trading_level_history_df[['Mon-Year', 'TradingLevel']], left_on=['Mon-Year'], right_on=['Mon-Year'], how='left')
    pnl_aggregate_history['ITD %'] = pnl_aggregate_history['ITD']/pnl_aggregate_history['TradingLevel']
    pnl_aggregate_history['MTD'] = pnl_aggregate_history['MTD'].apply(lambda x: get_currency_format(x))
    pnl_aggregate_history['DTD'] = pnl_aggregate_history['DTD'].apply(lambda x: get_currency_format(x))
    pnl_aggregate_history['ITD'] = pnl_aggregate_history['ITD'].apply(lambda x: get_currency_format(x))
    pnl_aggregate_history.set_index(["Date"], inplace=True)

    fig = go.Figure()

    fig.add_trace(go.Bar(
            x = pnl_aggregate_history.index.values,
            y = pnl_aggregate_history['ITD %'],
            hovertext=pnl_aggregate_history['ITD'],
            hovertemplate='Date: %{x}' + '<br>ITD Return: %{y}' + '<br>ITD PNL: %{hovertext}</br>',
            marker_color='#1A9687',
            name='ITD'
        ))

    fig.update_layout(height=500, title_text="Historical Cumulative Performance", yaxis_tickformat=',.2%',
                    template='plotly_white', showlegend=False, clickmode='select')
    st.plotly_chart(fig, use_container_width=True)

    mtd_aggregate_sector_history = pnl_history.groupby(['PeriodEndDate', 'Sector'])['DTD'].sum().groupby(level=1).cumsum().reset_index()
    mtd_aggregate_sector_history['PeriodEndDate'] = pd.to_datetime(mtd_aggregate_sector_history['PeriodEndDate'])
    mtd_aggregate_sector_history['Mon-Year'] = mtd_aggregate_sector_history['PeriodEndDate'].apply(lambda x: x.strftime("%B-%Y"))
    trading_level_history_df['Mon-Year'] = trading_level_history_df['AsOf'].apply(lambda x: x.strftime("%B-%Y"))
    mtd_aggregate_sector_history = pd.merge(mtd_aggregate_sector_history, trading_level_history_df[['Mon-Year', 'TradingLevel']], left_on=['Mon-Year'], right_on=['Mon-Year'], how='left')
    mtd_aggregate_sector_history.rename(columns={'DTD': 'ITD'}, inplace=True)
    mtd_aggregate_sector_history['ITD (%)'] = mtd_aggregate_sector_history['ITD']/mtd_aggregate_sector_history['TradingLevel']
    mtd_aggregate_sector_history['ITD'] = mtd_aggregate_sector_history['ITD'].apply(lambda x: get_currency_format(x))

    temp = mtd_aggregate_sector_history.pivot(values=['ITD', 'ITD (%)'], index=['PeriodEndDate'], columns=['Sector'])
    temp.fillna(0, inplace=True)
    mtd_dollar_sector_df = temp.loc[:, "ITD"]
    mtd_percentage_sector_df = temp.loc[:, "ITD (%)"]

    sector_color_map = {'Cash': '#1A9687', 'Currency': '#1A3357', 'Fixed Income': '#23CAB8', 
                        'Equity': '#03ADC9', 'Commodity': '#12D0A3', 'Commission & Fees': 'black', 'Inflation': 'grey'}

    fig = px.bar(mtd_percentage_sector_df, x=mtd_percentage_sector_df.index.values, y=[c for c in mtd_percentage_sector_df.columns],
                color_discrete_sequence=[sector_color_map[c] for c in mtd_percentage_sector_df.columns])

    fig.add_trace(go.Scatter(
            x = pnl_aggregate_history.index.values,
            y = pnl_aggregate_history['ITD %'],
            hovertext=pnl_aggregate_history['ITD'],
            hovertemplate='Date: %{x}' + '<br>ITD Return: %{y}' + '<br>ITD PNL: %{hovertext}</br>',
            name='ITD',
            marker_color='rgb(255, 217, 47)'
        ))
    fig.update_layout(height=500, title_text="Historical Cumulative Performance by Sector", yaxis_tickformat=',.2%',
                    template='plotly_white', legend_title_text='Sector', yaxis_title=None, xaxis_title=None,
                    clickmode='select')
    st.plotly_chart(fig, use_container_width=True)

    mtd_aggregate_subsector_history = pnl_history.groupby(['PeriodEndDate', 'MarketGroup'])['DTD'].sum().groupby(level=1).cumsum().reset_index()
    mtd_aggregate_subsector_history['PeriodEndDate'] = pd.to_datetime(mtd_aggregate_subsector_history['PeriodEndDate'])
    mtd_aggregate_subsector_history['Mon-Year'] = mtd_aggregate_subsector_history['PeriodEndDate'].apply(lambda x: x.strftime("%B-%Y"))
    trading_level_history_df['Mon-Year'] = trading_level_history_df['AsOf'].apply(lambda x: x.strftime("%B-%Y"))
    mtd_aggregate_subsector_history = pd.merge(mtd_aggregate_subsector_history, trading_level_history_df[['Mon-Year', 'TradingLevel']], left_on=['Mon-Year'], right_on=['Mon-Year'], how='left')
    mtd_aggregate_subsector_history.rename(columns={'DTD': 'ITD'}, inplace=True)
    mtd_aggregate_subsector_history['ITD (%)'] = mtd_aggregate_subsector_history['ITD']/mtd_aggregate_subsector_history['TradingLevel']
    mtd_aggregate_subsector_history['ITD'] = mtd_aggregate_subsector_history['ITD'].apply(lambda x: get_currency_format(x))

    temp = mtd_aggregate_subsector_history.pivot(values=['ITD', 'ITD (%)'], index=['PeriodEndDate'], columns=['MarketGroup'])
    temp.fillna(0, inplace=True)
    mtd_dollar_subsector_df = temp.loc[:, "ITD"]
    mtd_percentage_subsector_df = temp.loc[:, "ITD (%)"]

    fig = px.bar(mtd_percentage_subsector_df, x=mtd_percentage_subsector_df.index.values, y=[c for c in mtd_percentage_subsector_df.columns],
                color_discrete_sequence=px.colors.sequential.Teal)

    fig.add_trace(go.Scatter(
            x = pnl_aggregate_history.index.values,
            y = pnl_aggregate_history['ITD %'],
            hovertext=pnl_aggregate_history['ITD'],
            hovertemplate='Date: %{x}' + '<br>ITD Return: %{y}' + '<br>ITD PNL: %{hovertext}</br>',
            name='ITD',
            marker_color='rgb(255, 217, 47)'
        ))
    fig.update_layout(height=500, title_text="Historical Cumulative Performance by Sub Sector", yaxis_tickformat=',.2%',
                    template='plotly_white', legend_title_text='Sub Sector', yaxis_title=None, xaxis_title=None,
                    clickmode='select')
    st.plotly_chart(fig, use_container_width=True)

    mtd_aggregate_market_history = pnl_history.groupby(['PeriodEndDate', 'Market'])['DTD'].sum().groupby(level=1).cumsum().reset_index()
    mtd_aggregate_market_history['PeriodEndDate'] = pd.to_datetime(mtd_aggregate_market_history['PeriodEndDate'])
    mtd_aggregate_market_history['Mon-Year'] = mtd_aggregate_market_history['PeriodEndDate'].apply(lambda x: x.strftime("%B-%Y"))
    trading_level_history_df['Mon-Year'] = trading_level_history_df['AsOf'].apply(lambda x: x.strftime("%B-%Y"))
    mtd_aggregate_market_history = pd.merge(mtd_aggregate_market_history, trading_level_history_df[['Mon-Year', 'TradingLevel']], left_on=['Mon-Year'], right_on=['Mon-Year'], how='left')
    mtd_aggregate_market_history.rename(columns={'DTD': 'ITD'}, inplace=True)
    mtd_aggregate_market_history['ITD (%)'] = mtd_aggregate_market_history['ITD']/mtd_aggregate_market_history['TradingLevel']
    mtd_aggregate_market_history['ITD'] = mtd_aggregate_market_history['ITD'].apply(lambda x: get_currency_format(x))

    temp = mtd_aggregate_market_history.pivot(values=['ITD', 'ITD (%)'], index=['PeriodEndDate'], columns=['Market'])
    temp.fillna(0, inplace=True)
    mtd_dollar_market_df = temp.loc[:, "ITD"]
    mtd_percentage_market_df = temp.loc[:, "ITD (%)"]

    fig = px.bar(mtd_percentage_market_df, x=mtd_percentage_market_df.index.values, y=[c for c in mtd_percentage_market_df.columns],
                color_discrete_sequence=px.colors.sequential.Teal)

    fig.add_trace(go.Scatter(
            x = pnl_aggregate_history.index.values,
            y = pnl_aggregate_history['ITD %'],
            hovertext=pnl_aggregate_history['ITD'],
            hovertemplate='Date: %{x}' + '<br>ITD Return: %{y}' + '<br>ITD PNL: %{hovertext}</br>',
            name='ITD',
            marker_color='rgb(255, 217, 47)'
        ))
    fig.update_layout(height=500, title_text="Historical Cumulative Performance by Market", yaxis_tickformat=',.2%',
                    template='plotly_white', legend_title_text='Market', yaxis_title=None, xaxis_title=None,
                    clickmode='select')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<h3 style='text-align: center;'>Historical Exposure Summary</h3>", unsafe_allow_html=True)

    gross_exp_currency_df = pd.pivot_table(positions_history[positions_history['Sector']=='Currency'], values=['Net Exposure'], 
                index=['PositionAsOfDate', 'Currency'], aggfunc="sum")

    gross_exp_currency_df.reset_index(inplace=True)
    gross_exp_currency_df['Abs Net Exposure'] = abs(gross_exp_currency_df['Net Exposure'])
    gross_exp_currency_df  = pd.pivot_table(gross_exp_currency_df, values=['Abs Net Exposure'], 
                                            index=['PositionAsOfDate'], aggfunc="sum")
    gross_exp_currency_df.rename(columns={'Abs Net Exposure': 'Currency Gross Exposure'}, inplace=True)
    gross_exp_currency_df.fillna(0, inplace=True)
    gross_exp_non_currency_df = pd.pivot_table(positions_history[positions_history['Sector']!='Currency'], 
                                            values=['Gross Exposure'], index=['PositionAsOfDate'], aggfunc="sum")
    gross_exp_non_currency_df.rename(columns={'Gross Exposure': 'Non Currency Gross Exposure'}, inplace=True)
    gross_exp_aggregate_history = pd.merge(gross_exp_non_currency_df, gross_exp_currency_df, 
                                        left_index=True, right_index=True, how='outer')
    gross_exp_aggregate_history.fillna(0, inplace=True)
    gross_exp_aggregate_history['Gross Exposure'] = gross_exp_aggregate_history['Non Currency Gross Exposure'] + gross_exp_aggregate_history['Currency Gross Exposure']
    gross_exp_aggregate_history.reset_index(inplace=True)
    gross_exp_aggregate_history['PositionAsOfDate'] = pd.to_datetime(gross_exp_aggregate_history['PositionAsOfDate'], format="mixed")
    gross_exp_aggregate_history['Mon-Year'] = gross_exp_aggregate_history['PositionAsOfDate'].apply(lambda x: x.strftime("%B-%Y"))
    trading_level_history_df['Mon-Year'] = trading_level_history_df['AsOf'].apply(lambda x: x.strftime("%B-%Y"))
    gross_exp_aggregate_history = pd.merge(gross_exp_aggregate_history, trading_level_history_df[['Mon-Year', 'TradingLevel']], left_on=['Mon-Year'], right_on=['Mon-Year'], how='left')
    gross_exp_aggregate_history['Gross (%)'] = gross_exp_aggregate_history['Gross Exposure']/gross_exp_aggregate_history['TradingLevel']
    gross_exp_aggregate_history['Gross'] = gross_exp_aggregate_history['Gross Exposure'].apply(lambda x: get_currency_format(x))



    exposure_aggregate_history = pd.pivot_table(positions_history, values=['Net Exposure'], index=['PositionAsOfDate'], aggfunc="sum")
    exposure_aggregate_history.reset_index(inplace=True)
    exposure_aggregate_history['PositionAsOfDate'] = pd.to_datetime(exposure_aggregate_history['PositionAsOfDate'], format="mixed")
    exposure_aggregate_history['Mon-Year'] = exposure_aggregate_history['PositionAsOfDate'].apply(lambda x: x.strftime("%B-%Y"))
    #trading_level_history_df['Mon-Year'] = trading_level_history_df['AsOf'].apply(lambda x: x.strftime("%B-%Y"))
    exposure_aggregate_history = pd.merge(exposure_aggregate_history, trading_level_history_df[['Mon-Year', 'TradingLevel']], left_on=['Mon-Year'], right_on=['Mon-Year'], how='left')
    exposure_aggregate_history['Net (%)'] = exposure_aggregate_history['Net Exposure']/exposure_aggregate_history['TradingLevel']
    exposure_aggregate_history['Net'] = exposure_aggregate_history['Net Exposure'].apply(lambda x: get_currency_format(x))
    exposure_aggregate_history.set_index(['PositionAsOfDate'], inplace=True)

    fig = go.Figure()

    fig.add_trace(go.Bar(
            x = exposure_aggregate_history.index.values,
            y = exposure_aggregate_history['Net (%)'],
            hovertext=exposure_aggregate_history['Net'],
            hovertemplate='Date: %{x}' + '<br>Net Exposure: %{y}' + '<br>Net: %{hovertext}</br>',
            marker_color='#1A9687',
            name='Net'
        ))


    fig.update_layout(height=500, title_text="Historical Net Exposure", yaxis_tickformat=',.0%',
                    template='plotly_white', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()

    fig.add_trace(go.Bar(
            x = gross_exp_aggregate_history['PositionAsOfDate'],
            y = gross_exp_aggregate_history['Gross (%)'],
            hovertext=gross_exp_aggregate_history['Gross'],
            hovertemplate='Date: %{x}' + '<br>Gross Exposure: %{y}' + '<br>Gross: %{hovertext}</br>',
            marker_color='#1A3357',
            name='Gross'
        ))

    fig.update_layout(height=500, title_text="Historical Gross Exposure", yaxis_tickformat=',.0%',
                    template='plotly_white', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    exposure_aggregate_sector_history = pd.pivot_table(positions_history, values=['Net Exposure'], 
                                                    index=['PositionAsOfDate','Sector'], aggfunc="sum")
    exposure_aggregate_sector_history.reset_index(inplace=True)
    exposure_aggregate_sector_history['PositionAsOfDate'] = pd.to_datetime(exposure_aggregate_sector_history['PositionAsOfDate'], format="mixed")
    exposure_aggregate_sector_history['Mon-Year'] = exposure_aggregate_sector_history['PositionAsOfDate'].apply(lambda x: x.strftime("%B-%Y"))
    trading_level_history_df['Mon-Year'] = trading_level_history_df['AsOf'].apply(lambda x: x.strftime("%B-%Y"))
    exposure_aggregate_sector_history = pd.merge(exposure_aggregate_sector_history, trading_level_history_df[['Mon-Year', 'TradingLevel']], left_on=['Mon-Year'], right_on=['Mon-Year'], how='left')
    exposure_aggregate_sector_history['Net (%)'] = exposure_aggregate_sector_history['Net Exposure']/exposure_aggregate_sector_history['TradingLevel']
    exposure_aggregate_sector_history['Net'] = exposure_aggregate_sector_history['Net Exposure'].apply(lambda x: get_currency_format(x))
    exposure_aggregate_sector_history.reset_index(inplace=True)

    temp = exposure_aggregate_sector_history.pivot(values=['Net', 'Net (%)'], index=['PositionAsOfDate'], columns=['Sector'])
    temp.fillna(0, inplace=True)
    exp_dollar_sector_df = temp.loc[:, "Net"]
    exp_percentage_sector_df = temp.loc[:, "Net (%)"]

    exposure_aggregate_history.sort_index(inplace=True)

    sector_color_map = {'Cash': '#1A9687', 'Currency': '#1A3357', 'Fixed Income': '#23CAB8', 
                        'Equity': '#03ADC9', 'Commodity': '#12D0A3', 'Commission & Fees': 'black', 'Unknown': 'grey'}

    fig = px.bar(exp_percentage_sector_df, x=exp_percentage_sector_df.index.values, y=[c for c in exp_percentage_sector_df.columns],
                color_discrete_sequence=[sector_color_map[c] for c in exp_percentage_sector_df.columns])

    fig.add_trace(go.Scatter(
            x = exposure_aggregate_history.index.values,
            y = exposure_aggregate_history['Net (%)'],
            hovertext=exposure_aggregate_history['Net'],
            hovertemplate='Date: %{x}' + '<br>Net Exposure: %{y:.2f}' + '<br>Net: %{hovertext}</br>',
            name='Net',
            marker_color='rgb(255, 217, 47)'
        ))
    fig.update_layout(height=500, title_text="Historical Net Exposure by Sector", yaxis_tickformat=',.0%',
                    template='plotly_white', legend_title_text='Sector', yaxis_title=None, xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

    exposure_aggregate_subsector_history = pd.pivot_table(positions_history, values=['Net Exposure'], 
                                                        index=['PositionAsOfDate','MarketGroup'], aggfunc="sum")
    exposure_aggregate_subsector_history.reset_index(inplace=True)
    exposure_aggregate_subsector_history['PositionAsOfDate'] = pd.to_datetime(exposure_aggregate_subsector_history['PositionAsOfDate'], format="mixed")
    exposure_aggregate_subsector_history['Mon-Year'] = exposure_aggregate_subsector_history['PositionAsOfDate'].apply(lambda x: x.strftime("%B-%Y"))
    trading_level_history_df['Mon-Year'] = trading_level_history_df['AsOf'].apply(lambda x: x.strftime("%B-%Y"))
    exposure_aggregate_subsector_history = pd.merge(exposure_aggregate_subsector_history, trading_level_history_df[['Mon-Year', 'TradingLevel']], left_on=['Mon-Year'], right_on=['Mon-Year'], how='left')
    exposure_aggregate_subsector_history['Net (%)'] = exposure_aggregate_subsector_history['Net Exposure']/exposure_aggregate_subsector_history['TradingLevel']
    exposure_aggregate_subsector_history['Net'] = exposure_aggregate_subsector_history['Net Exposure'].apply(lambda x: get_currency_format(x))
    exposure_aggregate_subsector_history.reset_index(inplace=True)
    temp = exposure_aggregate_subsector_history.pivot(values=['Net', 'Net (%)'], index=['PositionAsOfDate'], columns=['MarketGroup'])
    temp.fillna(0, inplace=True)
    exp_dollar_subsector_df = temp.loc[:, "Net"]
    exp_percentage_subsector_df = temp.loc[:, "Net (%)"]

    fig = px.bar(exp_percentage_subsector_df, x=exp_percentage_subsector_df.index.values, y=[c for c in exp_percentage_subsector_df.columns],
                color_discrete_sequence=px.colors.sequential.Teal)

    fig.add_trace(go.Scatter(
            x = exposure_aggregate_history.index.values,
            y = exposure_aggregate_history['Net (%)'],
            hovertext=exposure_aggregate_history['Net'],
            hovertemplate='Date: %{x}' + '<br>Net Exposure: %{y}' + '<br>Net: %{hovertext}</br>',
            name='Net',
            marker_color='rgb(255, 217, 47)'
        ))
    fig.update_layout(height=500, title_text="Historical Net Exposure by Sub Sector", yaxis_tickformat=',.0%',
                    template='plotly_white', legend_title_text='Sub Sector', yaxis_title=None, xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

    exposure_aggregate_market_history = pd.pivot_table(positions_history, values=['Net Exposure'], 
                                                    index=['PositionAsOfDate','Market'], aggfunc="sum")
    exposure_aggregate_market_history.reset_index(inplace=True)
    exposure_aggregate_market_history['PositionAsOfDate'] = pd.to_datetime(exposure_aggregate_market_history['PositionAsOfDate'], format="mixed")
    exposure_aggregate_market_history['Mon-Year'] = exposure_aggregate_market_history['PositionAsOfDate'].apply(lambda x: x.strftime("%B-%Y"))
    trading_level_history_df['Mon-Year'] = trading_level_history_df['AsOf'].apply(lambda x: x.strftime("%B-%Y"))
    exposure_aggregate_market_history = pd.merge(exposure_aggregate_market_history, trading_level_history_df[['Mon-Year', 'TradingLevel']], left_on=['Mon-Year'], right_on=['Mon-Year'], how='left')
    exposure_aggregate_market_history['Net (%)'] = exposure_aggregate_market_history['Net Exposure']/exposure_aggregate_market_history['TradingLevel']
    exposure_aggregate_market_history['Net'] = exposure_aggregate_market_history['Net Exposure'].apply(lambda x: get_currency_format(x))
    exposure_aggregate_market_history.reset_index(inplace=True)
    temp = exposure_aggregate_market_history.pivot(values=['Net', 'Net (%)'], index=['PositionAsOfDate'], columns=['Market'])
    temp.fillna(0, inplace=True)
    exp_dollar_market_df = temp.loc[:, "Net"]
    exp_percentage_market_df = temp.loc[:, "Net (%)"]

    fig = px.bar(exp_percentage_market_df, x=exp_percentage_market_df.index.values, y=[c for c in exp_percentage_market_df.columns],
                color_discrete_sequence=px.colors.sequential.Teal)

    fig.add_trace(go.Scatter(
            x = exposure_aggregate_history.index.values,
            y = exposure_aggregate_history['Net (%)'],
            hovertext=exposure_aggregate_history['Net'],
            hovertemplate='Date: %{x}' + '<br>Net Exposure: %{y}' + '<br>Net: %{hovertext}</br>',
            name='Net',
            marker_color='rgb(255, 217, 47)'
        ))
    fig.update_layout(height=500, title_text="Historical Net Exposure by Market", yaxis_tickformat=',.2%',
                    template='plotly_white', legend_title_text='Market', yaxis_title=None, xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<h3 style='text-align: center;'>Historical VAR Summary</h3>", unsafe_allow_html=True)

    var_history['PositionAsOfDate'] = pd.to_datetime(var_history['PositionAsOfDate'].astype(str), format='%m/%d/%Y')
    var_history['Sector'] = var_history['Sector'].apply(lambda x: 'Cash' if x=='Inflation' else x)

    var_aggregate_history = pd.pivot_table(var_history, values=['IVaR_Hist_95','VaR_Hist_95', 'IVaR_Hist_99', 'VaR_Hist_99'], 
                                        index=['PositionAsOfDate'], aggfunc="sum")
    var_aggregate_history.reset_index(inplace=True)
    var_aggregate_history['PositionAsOfDate'] = pd.to_datetime(var_aggregate_history['PositionAsOfDate'])
    var_aggregate_history['Mon-Year'] = var_aggregate_history['PositionAsOfDate'].apply(lambda x: x.strftime("%B-%Y"))
    trading_level_history_df['Mon-Year'] = trading_level_history_df['AsOf'].apply(lambda x: x.strftime("%B-%Y"))
    var_aggregate_history = pd.merge(var_aggregate_history, trading_level_history_df[['Mon-Year', 'TradingLevel']], left_on=['Mon-Year'], right_on=['Mon-Year'], how='left')
    var_columns = ['IVaR_Hist_95','VaR_Hist_95', 'IVaR_Hist_99', 'VaR_Hist_99']
    for col in var_columns:
        var_aggregate_history[col + ' %'] = var_aggregate_history[col]/var_aggregate_history['TradingLevel']
    for col in var_columns:
        var_aggregate_history[col] = var_aggregate_history[col].apply(lambda x: get_currency_format(x))
    var_aggregate_history.set_index(['PositionAsOfDate'], inplace=True)

    fig = make_subplots(rows=1, cols=2,
                    subplot_titles=['IVAR 95%', 'IVAR 99%'],)

    # IVAR 95
    fig.add_trace(go.Scatter(
            x = var_aggregate_history.index.values,
            y = var_aggregate_history['IVaR_Hist_95 %'],
            hovertext=var_aggregate_history['IVaR_Hist_95'],
            hovertemplate='Date: %{x}' + '<br>IVAR 95: %{y:.4f}' + '<br>IVAR 95: %{hovertext}</br>',
            name='IVAR 95',
            marker_color='#1A3357'
        ), row=1, col=1)

    fig.add_trace(go.Bar(
            x = var_aggregate_history.index.values,
            y = var_aggregate_history['IVaR_Hist_95 %'],
            hovertext=var_aggregate_history['IVaR_Hist_95'],
            hovertemplate='Date: %{x}' + '<br>IVAR 95: %{y:.4f}' + '<br>IVAR 95: %{hovertext}</br>',
            marker_color='#1A9687',
            name='IVAR 95'
        ), row=1, col=1)

    # IVAR 99
    fig.add_trace(go.Scatter(
            x = var_aggregate_history.index.values,
            y = var_aggregate_history['IVaR_Hist_99 %'],
            hovertext=var_aggregate_history['IVaR_Hist_99'],
            hovertemplate='Date: %{x}' + '<br>IVAR 99: %{y:.4f}' + '<br>IVAR 99: %{hovertext}</br>',
            name='IVAR 99',
            marker_color='#1A3357'
        ), row=1, col=2)

    fig.add_trace(go.Bar(
            x = var_aggregate_history.index.values,
            y = var_aggregate_history['IVaR_Hist_99 %'],
            hovertext=var_aggregate_history['IVaR_Hist_99'],
            hovertemplate='Date: %{x}' + '<br>IVAR 99: %{y:.4f}' + '<br>IVAR 99: %{hovertext}</br>',
            marker_color='#1A9687',
            name='IVAR 99'
        ), row=1, col=2)


    fig.update_yaxes(tickformat=".2%")
    fig.update_layout(height=500, #title_text="Historical VAR Performance (Overall)",
                    template='plotly_white', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    var_aggregate_sector_history = pd.pivot_table(var_history, values=['IVaR_Hist_95'], index=['PositionAsOfDate','Sector'], aggfunc="sum")
    var_aggregate_sector_history.reset_index(inplace=True)
    var_aggregate_sector_history['PositionAsOfDate'] = pd.to_datetime(var_aggregate_sector_history['PositionAsOfDate'])
    var_aggregate_sector_history['Mon-Year'] = var_aggregate_sector_history['PositionAsOfDate'].apply(lambda x: x.strftime("%B-%Y"))
    trading_level_history_df['Mon-Year'] = trading_level_history_df['AsOf'].apply(lambda x: x.strftime("%B-%Y"))
    var_aggregate_sector_history = pd.merge(var_aggregate_sector_history, trading_level_history_df[['Mon-Year', 'TradingLevel']], left_on=['Mon-Year'], right_on=['Mon-Year'], how='left')
    var_aggregate_sector_history['IVAR 95(%)'] = var_aggregate_sector_history['IVaR_Hist_95']/var_aggregate_sector_history['TradingLevel']
    var_aggregate_sector_history['IVAR 95'] = var_aggregate_sector_history['IVaR_Hist_95'].apply(lambda x: get_currency_format(x))
    temp = var_aggregate_sector_history.pivot(values=['IVAR 95', 'IVAR 95(%)'], index=['PositionAsOfDate'], columns=['Sector'])
    temp.fillna(0, inplace=True)
    var_dollar_sector_df = temp.loc[:, "IVAR 95"]
    var_percentage_sector_df = temp.loc[:, "IVAR 95(%)"]

    sector_color_map = {'Cash': '#1A9687', 'Currency': '#1A3357', 'Fixed Income': '#23CAB8', 
                        'Equity': '#03ADC9', 'Commodity': '#12D0A3', 'Commission & Fees': 'black', 'Inflation': 'grey'}

    fig = px.bar(var_percentage_sector_df, x=var_percentage_sector_df.index.values, y=[c for c in var_percentage_sector_df.columns],
                color_discrete_sequence=[sector_color_map[c] for c in var_percentage_sector_df.columns])

    fig.add_trace(go.Scatter(
            x = var_aggregate_history.index.values,
            y = var_aggregate_history['IVaR_Hist_95 %'],
            hovertext=var_aggregate_history['IVaR_Hist_95'],
            hovertemplate='Date: %{x}' + '<br>IVAR 95 (%): %{y}' + '<br>IVAR 95 ($): %{hovertext}</br>',
            name='Net',
            marker_color='rgb(255, 217, 47)'
        ))

    fig.update_layout(height=500, title_text="Historical IVAR 95% by Sector", yaxis_tickformat=',.2%',
                    template='plotly_white', legend_title_text='Sector', yaxis_title=None, xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

    var_aggregate_sector_history = pd.pivot_table(var_history, values=['IVaR_Hist_99'], index=['PositionAsOfDate','Sector'], aggfunc="sum")
    var_aggregate_sector_history.reset_index(inplace=True)
    var_aggregate_sector_history['PositionAsOfDate'] = pd.to_datetime(var_aggregate_sector_history['PositionAsOfDate'])
    var_aggregate_sector_history['Mon-Year'] = var_aggregate_sector_history['PositionAsOfDate'].apply(lambda x: x.strftime("%B-%Y"))
    trading_level_history_df['Mon-Year'] = trading_level_history_df['AsOf'].apply(lambda x: x.strftime("%B-%Y"))
    var_aggregate_sector_history = pd.merge(var_aggregate_sector_history, trading_level_history_df[['Mon-Year', 'TradingLevel']], left_on=['Mon-Year'], right_on=['Mon-Year'], how='left')
    var_aggregate_sector_history['IVAR 99(%)'] = var_aggregate_sector_history['IVaR_Hist_99']/var_aggregate_sector_history['TradingLevel']
    var_aggregate_sector_history['IVAR 99'] = var_aggregate_sector_history['IVaR_Hist_99'].apply(lambda x: get_currency_format(x))
    temp = var_aggregate_sector_history.pivot(values=['IVAR 99', 'IVAR 99(%)'], index=['PositionAsOfDate'], columns=['Sector'])
    temp.fillna(0, inplace=True)
    var_dollar_sector_df = temp.loc[:, "IVAR 99"]
    var_percentage_sector_df = temp.loc[:, "IVAR 99(%)"]

    sector_color_map = {'Cash': '#1A9687', 'Currency': '#1A3357', 'Fixed Income': '#23CAB8', 
                        'Equity': '#03ADC9', 'Commodity': '#12D0A3', 'Commission & Fees': 'black', 'Inflation': 'grey'}

    fig = px.bar(var_percentage_sector_df, x=var_percentage_sector_df.index.values, y=[c for c in var_percentage_sector_df.columns],
                color_discrete_sequence=[sector_color_map[c] for c in var_percentage_sector_df.columns])

    fig.add_trace(go.Scatter(
            x = var_aggregate_history.index.values,
            y = var_aggregate_history['IVaR_Hist_99 %'],
            hovertext=var_aggregate_history['IVaR_Hist_99'],
            hovertemplate='Date: %{x}' + '<br>IVAR 99 (%): %{y}' + '<br>IVAR 99 ($): %{hovertext}</br>',
            name='Net',
            marker_color='rgb(255, 217, 47)'
        )) 

    fig.update_layout(height=500, title_text="Historical IVAR 99% by Sector", yaxis_tickformat=',.2%',
                    template='plotly_white', legend_title_text='Sector', yaxis_title=None, xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

    # st.markdown("<h3 style='text-align: center;'>IAA Guidelines Summary</h3>", unsafe_allow_html=True)

    margins_df = get_pm_margins(csv_dataframes)
    margins_df['FundName'] = margins_df['map'].apply(lambda x: get_pm_name(x))
    historical_margins_df = pd.pivot_table(margins_df, columns=['FundName'], index=['Date'], values=['MarginRequirement'],  aggfunc="sum")
    #historical_margins_df.fillna(0, inplace=True)
    historical_margins_df = historical_margins_df['MarginRequirement']
    historical_margins_df.reset_index(inplace=True)
    historical_margins_df['Date'] = pd.to_datetime(historical_margins_df['Date'], format="%Y-%m-%d")
    historical_margins_df.set_index(['Date'], inplace=True)
    # #historical_margins_df.to_excel(r"C:\Users\hnair\Documents\Portfolio Construction\Historical_Margins.xlsx")

    margin_pnl_df = pd.pivot_table(pnl_history, values=['MTD', 'DTD'], index=['PeriodEndDate'], aggfunc="sum")
    margin_pnl_df['ITD'] = margin_pnl_df['DTD'].cumsum()
    margin_pnl_df.reset_index(inplace=True)
    margin_pnl_df.rename(columns={'PeriodEndDate': 'Date'}, inplace=True)
    margin_pnl_df['Date'] = pd.to_datetime(margin_pnl_df['Date'], format="%Y-%m-%d")
    margin_pnl_df['Mon-Year'] = margin_pnl_df['Date'].apply(lambda x: x.strftime("%B-%Y"))
    trading_level_history_df['Mon-Year'] = trading_level_history_df['AsOf'].apply(lambda x: x.strftime("%B-%Y"))
    margin_pnl_df = pd.merge(margin_pnl_df, trading_level_history_df[['Mon-Year', 'TradingLevel']], left_on=['Mon-Year'], right_on=['Mon-Year'], how='left')
    margin_pnl_df.set_index(['Date'], inplace=True)
    margin_pnl_df = pd.merge(margin_pnl_df, historical_margins_df, left_index=True, right_index=True, how='outer')
    margin_pnl_df.rename(columns={'MKR - MAP 240_27': 'Margin'}, inplace=True)

    # # st.write(margin_pnl_df.index)

    current_margin_val = margin_pnl_df.tail(1)['Margin'].values[0]
    margin_pnl_df['Margin (%)'] = round(margin_pnl_df['Margin']/margin_pnl_df['TradingLevel'], 4)
    margin_pnl_df['Margin 3M Average %'] = rolling_average(margin_pnl_df['Margin (%)'], 66)
    margin_pnl_df['Margin 3M Average'] = margin_pnl_df['Margin 3M Average %'] * margin_pnl_df['TradingLevel']
    margin_pnl_df['ROC'] = margin_pnl_df['ITD']/margin_pnl_df['Margin 3M Average']
    #margin_pnl_df['ROC'] = margin_pnl_df['ITD']/margin_pnl_df['Margin']
    margin_pnl_df['ITD'] = margin_pnl_df['ITD'].apply(lambda x: get_currency_format(x))
    margin_pnl_df['Margin'] = margin_pnl_df['Margin'].apply(lambda x: get_currency_format(x))

    positions_snapshot['Level 3 Assets'] = positions_snapshot.apply(lambda x:get_level3_assets(x['InstrumentType'], x['Quantity']) , axis=1)
    positions_snapshot['Short Options'] = positions_snapshot.apply(lambda x:get_number_short_options(x['InstrumentType'], x['Quantity']) , axis=1)
    positions_snapshot['Ticker'] = positions_snapshot['Investment'].apply(lambda x: str(x).split('_')[0] if pd.notnull(x) else '')
    positions_snapshot['CFTC'] = positions_snapshot.apply(lambda x: get_cftc_spot_month_limit(x['InstrumentName'], x['InstrumentType'], x['Ticker'], x['Quantity']), axis=1)

    # iaa_limits_path = 'data/IAA_Limits.xlsx'
    # iaa_df = pd.read_excel(iaa_limits_path)
    # iaa_df = iaa_df[iaa_df['ShortName'] == 'PMAP 240 MKR CAPITAL ']
    # iaa_df['Account Limit Type'] = iaa_df.apply(lambda x: get_account_limit_type(x['Description'],x['Limit1Type'], x['Sector'], x['Market']), axis=1)
    # iaa_df['Value Type'] = iaa_df.apply(lambda x: get_value_type(x['Description'], x['Limit1Type']), axis=1)
    # iaa_df['Value'] = iaa_df.apply(lambda x: get_value_exposure(x['Description'], x['Sector'], x['Market'], x['Value Type'], positions_snapshot, current_margin_val, csv_dataframes), axis=1)
    # iaa_df['Limit Value'] = iaa_df.apply(lambda x: get_limit_value(x['Limit1Type'], x['Limit1'], current_trading_level), axis=1)
    # iaa_df.fillna(0,inplace=True)
    # iaa_df['Satisfy Limit'] = iaa_df.apply(lambda x: satisfy_limit(x['Limit1Type'], x['Value'],x['Limit Value']), axis=1)
    # iaa_df['% Limit Usage'] = iaa_df.apply(lambda x: get_limit_usage(x['Value'], x['Limit Value'], x['Limit1Type']), axis=1)
    # iaa_df['Value'] = iaa_df['Value'].apply(lambda x: get_currency_format(x))
    # iaa_df['Limit Value'] = iaa_df['Limit Value'].apply(lambda x: get_currency_format(x))
    # iaa_df['% Limit Usage'] =  iaa_df['% Limit Usage'].apply(lambda x: "{:.2f}%".format(x*100))
    # iaa_df['Limit1'] = iaa_df['Limit1'].apply(lambda x: "{:.2f}%".format(x))
    # iaa_df.rename(columns={'Limit1': 'Limit', 'Limit1Type': 'Limit Type'}, inplace=True)
    # iaa_df['Sector'] = iaa_df['Sector'].apply(lambda x: 'All' if x in ['Market', 'Portfolio'] else x)
    # iaa_df['Market'] = iaa_df['Market'].apply(lambda x: 'All' if x in ['Market', 'Portfolio'] else x)
    # iaa_df = iaa_df.sort_values(by=['Sector', 'Market'])
    # iaa_df = iaa_df[['Sector', 'Market','Account Limit Type', 'Limit', 'Limit Type', 'Limit Value', 'Value', 'Satisfy Limit', '% Limit Usage']]

    # headers = [['','<b>'+c+'</b>'] for c in iaa_df.columns]
    # values = []
    # for col in iaa_df.columns:
    #     values.append(iaa_df[col])

    # topHeaderColor = '#1A3357'
    # nextHeaderColor = '#1A9687'

    # rowOddColor = '#F2F2F2'
    # rowEvenColor = '#FFFFFF'
    # fig = go.Figure(
    #     go.Table(
    #         header=dict(values=headers,
    #                     line = dict(width=0),
    #                    fill_color=[[topHeaderColor, nextHeaderColor]*6],
    #                     align='center',
    #                     font=dict(family="Aptos Narrow", color="white", size=14)
    #                    ),
    #         cells=dict(values=values,
    #                    line = dict(width=0),
    #                   align='center',
    #                   font=dict(family="Segoe UI", color="black", size=12),
    #                   fill_color = [[rowOddColor, rowEvenColor,rowOddColor, rowEvenColor,rowOddColor, rowEvenColor]*6,
    #                                ],
    #                   height=30)
    #     )
    # )
    # fig.update_layout(height=700)
    # st.plotly_chart(fig, use_container_width=True)

    df = liquidity_limits_check(positions_snapshot, csv_dataframes['LiqLimits.csv'])
    df = df.groupby(df['BBG Ticker']).agg({'InstrumentName': ', '.join,
                                'Sector':'first', 
                                'Market': 'first', 
                                'Quantity':'sum',
                                'Net Exposure': 'sum', 
                                'VOLUME_AVG_30D': 'first', 
                                'Limit': 'first', 
                                'Satisfy Limit': 'first'}).reset_index()
    df['Net Exposure'] = df['Net Exposure'].apply(lambda x: get_currency_format(x))
    df['VOLUME_AVG_30D'] = df['VOLUME_AVG_30D'].apply(lambda x: '{0:,.2f}'.format(x))
    df['Limit'] = df['Limit'].apply(lambda x: '{0:,.0f}'.format(x))
    headers = [['','<b>'+c+'</b>'] for c in df.columns]
    values = []
    for col in df.columns:
        values.append(df[col])

    st.markdown("<h3 style='text-align: center;'>Liquidity Limits</h3>", unsafe_allow_html=True)

    topHeaderColor = '#1A3357'
    nextHeaderColor = '#1A9687'

    rowOddColor = '#F2F2F2'
    rowEvenColor = '#FFFFFF'
    fig = go.Figure(
        go.Table(
            header=dict(values=headers,
                        line = dict(width=0),
                    fill_color=[[topHeaderColor, nextHeaderColor]*6],
                        align='center',
                        font=dict(family="Aptos Narrow", color="white", size=14)
                    ),
            cells=dict(values=values,
                    line = dict(width=0),
                    align='center',
                    font=dict(family="Segoe UI", color="black", size=12),
                    fill_color = [[rowOddColor, rowEvenColor,rowOddColor, rowEvenColor,rowOddColor, rowEvenColor]*6,
                                ],
                    height=30)
        )
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<h3 style='text-align: center;'>Margin Statistics</h3>", unsafe_allow_html=True)

    fig = go.Figure()

    fig.add_trace(go.Bar(
            x = margin_pnl_df.index.values,
            y = margin_pnl_df['Margin (%)'],
            hovertext=margin_pnl_df['Margin'],
            hovertemplate='Date: %{x}' + '<br>Margin (%): %{y}' + '<br>Margin ($): %{hovertext}</br>',
            name='Margin',
            marker_color='#1A3357'
        ))

    fig.update_layout(height=500, title_text="Historical Margin Usage", yaxis_tickformat=',.2%',
                    template='plotly_white', legend_title_text='Sector', yaxis_title=None, xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

    # roc_df = margin_pnl_df[['ROC', 'ITD']]
    # roc_df.dropna(inplace=True)
    # fig = go.Figure()

    # fig.add_trace(go.Bar(
    #         x = roc_df.index.values,
    #         y = roc_df['ROC'],
    #         hovertext=roc_df['ITD'],
    #         hovertemplate='Date: %{x}' + '<br>ROC : %{y}' + '<br>ITD PNL ($): %{hovertext}</br>',
    #         name='ROC',
    #         marker_color='#1A9687'
    #     ))

    # fig.update_layout(height=500, title_text="Historical Cumulative Return on Margin (%)", yaxis_tickformat=',.2%',
    #                  template='plotly_white', legend_title_text='Sector', yaxis_title=None, xaxis_title=None)
    # st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error("Error Processing PM Report Visualisations - Admin has been notified")
    send_teams_message(f"Error processing PM Report Visualisations - {e}")