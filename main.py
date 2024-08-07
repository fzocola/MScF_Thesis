# Import packages
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from tqdm import tqdm

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


dic_issuer_rating_daily = dic_rating_preprocessing_1(dic_issuer_rating_daily)


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



# Issuer rating preprocessing_2
df_issuer_rating_daily = dic_issuer_rating_daily['RATING']

# Check the values in the rating dataframe
s_issuer_rating_daily_unique = pd.unique(df_issuer_rating_daily.values.ravel())

# Define a mapping from BBG ratings to S&P rating
bbg_rating_to_sp_rating = {
    'AAA': 'AAA',
    'AA+': 'AA+', 'AA': 'AA', 'AA-': 'AA-',
    'A+': 'A+', 'A': 'A', 'A-': 'A-',
    'BBB+': 'BBB+', 'BBB': 'BBB', 'BBB-': 'BBB-',
    'BB+': 'BB+', 'BB': 'BB', 'BB-': 'BB-',
    'B+': 'B+', 'B': 'B', 'B-': 'B-',
    'CCC+': 'CCC+', 'CCC': 'CCC', 'CCC-': 'CCC-',
    'CC+': 'CC', 'CC': 'CC', 'CC-': 'CC',
    'C+': 'C', 'C': 'C', 'C-': 'C',
    'DDD+': 'D', 'DDD': 'D', 'DDD-': 'D',
    'DD+': 'D', 'DD': 'D', 'DD-': 'D',
    'D+': 'D', 'D': 'D', 'D-': 'D',
}


# Apply the mapping to each column of the dataframe
df_issuer_rating_daily = df_issuer_rating_daily.apply(lambda col: col.map(bbg_rating_to_sp_rating))

# Check the values in the rating dataframe
s_issuer_rating_daily_unique = pd.unique(df_issuer_rating_daily.values.ravel())

# Define a mapping from S&P rating to numeric values
sp_rating_to_numeric = {
    'AAA': 1,
    'AA+': 2, 'AA': 3, 'AA-': 4,
    'A+': 5, 'A': 6, 'A-': 7,
    'BBB+': 8, 'BBB': 9, 'BBB-': 10,
    'BB+': 11, 'BB': 12, 'BB-': 13,
    'B+': 14, 'B': 15, 'B-': 16,
    'CCC+': 17, 'CCC': 18, 'CCC-': 19,
    'CC': 20,
    'C': 21,
    'D': 22,
}

df_issuer_rating_numeric_daily = df_issuer_rating_daily.apply(lambda col: col.map(sp_rating_to_numeric))

s_issuer_rating_daily_numeric_unique = pd.unique(df_issuer_rating_daily.values.ravel())


# ****


# Create a new DataFrame with the same index and columns, filled with zeros
df_issuer_rating_downgrade_daily = pd.DataFrame(np.nan,
                                             index=df_issuer_rating_numeric_daily.index,
                                             columns=df_issuer_rating_numeric_daily.columns)


# Set last_date
last_date = df_issuer_rating_numeric_daily.index[-1] - pd.DateOffset(years=1)


for issuer in tqdm(df_issuer_rating_numeric_daily.iloc[:, :1].columns, desc='Preprocessing (2)'):

    # Use NumPy arrays for faster operations
    ratings = df_issuer_rating_numeric_daily[issuer].values
    dates = df_issuer_rating_numeric_daily.index

    # Pre-calculate index range for the last date to optimize loop
    max_index = np.searchsorted(dates, last_date, side='right')

    # Iterate over the index, stopping at the calculated range
    for i in range(max_index):
        current_date = dates[i]
        rating_tmp = ratings[i]

        # Find the end index for the next year period
        end_date = current_date + pd.DateOffset(years=1)
        end_index = np.searchsorted(dates, end_date, side='right')

        # Use slicing and vectorized comparisons
        future_ratings = ratings[i + 1:end_index]

        # Check if any rating is less than the current rating
        downgrade_detected = np.any(future_ratings < rating_tmp)

        # Assign the result
        df_issuer_rating_downgrade_daily.iat[i, df_issuer_rating_numeric_daily.columns.get_loc(issuer)] = 1 if downgrade_detected else 0





'''
for issuer in tqdm(df_issuer_rating_numeric_daily.iloc[:, :1].columns, desc='Preprocessing (2)'):

    print(issuer)

    last_date = df_issuer_rating_numeric_daily.index[-1] - pd.DateOffset(years=1)
    #print(last_date)

    for i in range(len(df_issuer_rating_numeric_daily.index)):

        #print(i)

        if df_issuer_rating_numeric_daily.index[i] <= last_date:

            print(df_issuer_rating_numeric_daily.index[i])

            rating_tmp = df_issuer_rating_numeric_daily.iloc[i][issuer]

            #print(rating_tmp)

            # Calculate the end date for the next year period
            end_date = df_issuer_rating_numeric_daily.index[i] + pd.DateOffset(years=1)

            print(end_date)

            # Check for any changes in the credit rating within the next year
            j = i +1
            while df_issuer_rating_numeric_daily.index[j] <= end_date:
                print(j)
                j = j+1
                if df_issuer_rating_numeric_daily.iloc[j][issuer] < rating_tmp:
                    print('ok')
                    df_issuer_rating_downgrade_daily.at[df_issuer_rating_numeric_daily.index[i], issuer] = 1
                    break
                else:
                    df_issuer_rating_downgrade_daily.at[df_issuer_rating_numeric_daily.index[i], issuer] = 0
                j = j + 1
'''






