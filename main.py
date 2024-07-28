# Import packages
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

pd.set_option('future.no_silent_downcasting', True)


# Project directories paths
paths = {'main': Path.cwd()}
paths.update({'data': Path.joinpath(paths.get('main'), 'data')})
paths.update({'scripts': Path.joinpath(paths.get('main'), 'scripts')})



# %%
# **********************************************************
# *** Section: DATA MANAGEMENT                           ***
# **********************************************************

dic_fm_data = pd.read_excel(Path.joinpath(paths.get('data'), 'Thesis_Data_Importable.xlsx'), sheet_name=None)

'''
for name, df in dic_fm_data.items():
    print(name)
'''

ls_fundamental_sheet_name = ['LT_DEBT', 'ST_DEBT', 'CFO', 'INTEREST_EXP', 'EBITDA', 'NET_DEBT', 'TOT_ASSET',
                             'WORK_CAP', 'RET_EARN', 'EBIT', 'SALES', 'CUR_ASSET', 'CUR_LIAB', 'BOOK_EQUITY',
                             'CASH', 'NET_INCOME']
dic_issuer_fundamental_quarterly = {i: dic_fm_data[i] for i in ls_fundamental_sheet_name}

ls_issuer_market_sheet_name = ['MKT_CAP', 'TOT_RETURN', 'SHARE_PRICE', 'CDS_SPREAD_5Y']
dic_issuer_market_daily = {i: dic_fm_data[i] for i in ls_issuer_market_sheet_name}

ls_issuer_market_sheet_name = ['RATING']
dic_issuer_rating_daily = {i: dic_fm_data[i] for i in ls_issuer_market_sheet_name}

ls_market_sheet_name = ['RATES', 'SPX']
dic_market_data_daily = {i: dic_fm_data[i] for i in ls_market_sheet_name}

ls_issuer_description_sheet_name = ['ISSUER', 'CDS_TICKER']
dic_issuer_description = {i: dic_fm_data[i] for i in ls_issuer_description_sheet_name}


def get_market_open_date(s_ref_data):
    s_ref_data = s_ref_data.dropna()
    s_market_open_date = pd.Series(s_ref_data.index)
    return s_market_open_date


s_market_open_date = get_market_open_date(dic_market_data_daily['SPX'].set_index('DATES')['px_last'])


# Set the index of the dataframe
def set_index_df_in_dic(dic, index):
    for i in dic:
        dic[i] = dic[i].set_index(index)
    return dic


dic_issuer_fundamental_quarterly = set_index_df_in_dic(dic_issuer_fundamental_quarterly, 'DATES')
dic_issuer_market_daily = set_index_df_in_dic(dic_issuer_market_daily, 'DATES')
dic_issuer_rating_daily = set_index_df_in_dic(dic_issuer_rating_daily, 'DATES')
dic_market_data_daily = set_index_df_in_dic(dic_market_data_daily, 'DATES')


# *** Fundamental data preprocessing ***

# Convert quarterly data to daily data
def df_conv_quarterly_to_daily(df):
    na_to_nb_value = -9999999999999
    df = df.fillna(na_to_nb_value)

    # Create a date range that includes all daily dates within the range of the quarterly data
    daily_index = pd.date_range(start=df.index.min(), end=df.index.max() + timedelta(days=90), freq='D')

    # Reindex the DataFrame to the daily date range and forward-fill the values
    df_daily = df.reindex(daily_index).ffill()
    df_daily = df_daily.replace(na_to_nb_value, np.nan)

    return df_daily


def dic_conv_quarterly_to_daily(dic):
    for i in dic:
        dic[i] = df_conv_quarterly_to_daily(dic[i])
    return dic


dic_issuer_fundamental_quarterly = dic_conv_quarterly_to_daily(dic_issuer_fundamental_quarterly)

# Lag the fundamental data by 90 days
def dic_lag_data(dic,lag):
    for i in dic:
        dic[i] = dic[i].shift(lag)
    return dic


dic_issuer_fundamental_quarterly = dic_lag_data(dic_issuer_fundamental_quarterly, lag=90)


# *** Issuer rating preprocessing ***
def dic_rating_preprocessing_1(dic):
    for i in dic:
        # Forward-fill the values
        dic[i] = dic[i].ffill()
        dic[i] = dic[i].replace('NR', np.nan)
    return dic


# Date filtering
def df_filtered_date(df, s_date):
    df = df[df.index.isin(s_date)]
    return df


def dic_filtered_date(dic, s_date):
    for i in dic:
        dic[i] = df_filtered_date(dic[i], s_date)
    return dic


dic_issuer_fundamental_quarterly = dic_filtered_date(dic_issuer_fundamental_quarterly, s_market_open_date)
dic_issuer_market_daily = dic_filtered_date(dic_issuer_market_daily, s_market_open_date)
dic_issuer_rating_daily = dic_filtered_date(dic_issuer_rating_daily, s_market_open_date)
dic_market_data_daily = dic_filtered_date(dic_market_data_daily, s_market_open_date)


