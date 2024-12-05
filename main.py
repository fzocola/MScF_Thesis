# Import packages
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from tqdm import tqdm
import copy
from scipy import stats
from scipy.optimize import minimize
from scipy.optimize import root
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
from imblearn.over_sampling import SMOTE
import warnings
import statsmodels.api as sm
from arch import arch_model

# Options
pd.set_option('future.no_silent_downcasting', True)

# Project directories paths
paths = {'main': Path.cwd()}
paths.update({'data': Path.joinpath(paths.get('main'), 'data')})
paths.update({'scripts': Path.joinpath(paths.get('main'), 'scripts')})

# Warnings management
#warnings.simplefilter("ignore", category=UserWarning)




# %%
# **********************************************************
# *** Section: DATA MANAGEMENT                           ***
# **********************************************************


# Import raw data
dic_fm_data = pd.read_excel(Path.joinpath(paths.get('data'), 'Thesis_Data_Importable.xlsx'), sheet_name=None)


# Select data
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


# Get market open days
def get_market_open_date(s_ref_data):
    s_ref_data_copy = copy.deepcopy(s_ref_data)
    s_ref_data_copy = s_ref_data_copy.dropna()
    s_date = pd.Series(s_ref_data_copy.index)
    return s_date

s_market_open_date = get_market_open_date(dic_market_data_daily['SPX'].set_index('DATES')['px_last'])


# Set the index of the dataframe
def set_index_df_in_dic(dic, index):
    dic_copy = copy.deepcopy(dic)
    for i in dic_copy:
        dic_copy[i] = dic_copy[i].set_index(index)
    return dic_copy

dic_issuer_fundamental_quarterly = set_index_df_in_dic(dic_issuer_fundamental_quarterly, 'DATES')
dic_issuer_market_daily = set_index_df_in_dic(dic_issuer_market_daily, 'DATES')
dic_issuer_rating_daily = set_index_df_in_dic(dic_issuer_rating_daily, 'DATES')
dic_market_data_daily = set_index_df_in_dic(dic_market_data_daily, 'DATES')


# *** Fundamental data preprocessing ***

# Convert quarterly data to daily data
def df_conv_quarterly_to_daily(df):
    df_copy = copy.deepcopy(df)
    # Fill all the nan with na_to_nb_value, we don't want to replace nan with last value
    na_to_nb_value = -9999999999999
    df_copy = df_copy.fillna(na_to_nb_value)

    # Create a date range that includes all daily dates within the range of the quarterly data
    daily_index = pd.date_range(start=df_copy.index.min(), end=df_copy.index.max() + timedelta(days=90), freq='D')

    # Reindex the DataFrame to the daily date range and forward-fill the values
    df_daily = df_copy.reindex(daily_index).ffill()
    df_daily = df_daily.replace(na_to_nb_value, np.nan)

    return df_daily


def dic_conv_quarterly_to_daily(dic):
    dic_copy = copy.deepcopy(dic)
    for i in dic_copy:
        dic_copy[i] = df_conv_quarterly_to_daily(dic_copy[i])
    return dic_copy


dic_issuer_fundamental_daily = dic_conv_quarterly_to_daily(dic_issuer_fundamental_quarterly)

# Convert quarterly data to monthly data
def df_conv_quarterly_to_monthly(df):
    df_copy = copy.deepcopy(df)

    # Align dates to the end of the month - Make sure that the date of the quarterly data is month end
    df_copy.index = df_copy.index + pd.offsets.MonthEnd(0)

    # Fill all the nan with na_to_nb_value, we don't want to replace nan with last value
    na_to_nb_value = -9999999999999
    df_copy = df_copy.fillna(na_to_nb_value)

    # Create a date range that includes all monthly dates within the range of the quarterly data
    monthly_index = pd.date_range(start=df_copy.index.min(), end=df_copy.index.max() + pd.DateOffset(months=2), freq='ME')

    # Reindex the DataFrame to the monthly date range and forward-fill the values
    df_monthly = df_copy.reindex(monthly_index).ffill()
    df_monthly = df_monthly.replace(na_to_nb_value, np.nan)

    return df_monthly


def dic_conv_quarterly_to_monthly(dic):
    dic_copy = copy.deepcopy(dic)
    for i in dic_copy:
        dic_copy[i] = df_conv_quarterly_to_monthly(dic_copy[i])
    return dic_copy

dic_issuer_fundamental_monthly = dic_conv_quarterly_to_monthly(dic_issuer_fundamental_quarterly)

# Lag the fundamental data by 90 days
def dic_lag_data(dic,lag):
    dic_copy = copy.deepcopy(dic)
    for i in dic_copy:
        dic_copy[i] = dic_copy[i].shift(lag)
    return dic_copy

dic_issuer_fundamental_daily = dic_lag_data(dic_issuer_fundamental_daily, lag=90)

# Lag the fundamental data by 3 months
dic_issuer_fundamental_monthly = dic_lag_data(dic_issuer_fundamental_monthly, lag=3)


# *** Issuer rating preprocessing 1 - before date filtering **

def dic_rating_preprocessing_1(dic):
    dic_copy = copy.deepcopy(dic)
    for i in dic_copy:
        # Forward-fill the values
        dic_copy[i] = dic_copy[i].ffill()
        dic_copy[i] = dic_copy[i].replace('NR', np.nan)
    return dic_copy

dic_issuer_rating_daily = dic_rating_preprocessing_1(dic_issuer_rating_daily)


# *** Date filtering ***

def df_filtered_date(df, s_date):
    df_copy = copy.deepcopy(df)
    df_copy = df_copy[df_copy.index.isin(s_date)]
    return df_copy


def dic_filtered_date(dic, s_date):
    dic_copy = copy.deepcopy(dic)
    for i in dic_copy:
        dic_copy[i] = df_filtered_date(dic_copy[i], s_date)
    return dic_copy


dic_issuer_fundamental_daily = dic_filtered_date(dic_issuer_fundamental_daily, s_market_open_date)
dic_issuer_market_daily = dic_filtered_date(dic_issuer_market_daily, s_market_open_date)
dic_issuer_rating_daily = dic_filtered_date(dic_issuer_rating_daily, s_market_open_date)
dic_market_data_daily = dic_filtered_date(dic_market_data_daily, s_market_open_date)


# *** Issuer rating preprocessing_2 ***
df_issuer_rating_daily = dic_issuer_rating_daily['RATING']

# Define a mapping from BBG ratings to numeric values
def rating_to_numeric(df):
    df_copy = copy.deepcopy(df)
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
    df_copy = df_copy.apply(lambda col: col.map(bbg_rating_to_sp_rating))

    # Define a mapping from S&P rating to numeric values
    '''
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
    '''
    '''
    sp_rating_to_numeric = {
        'AAA': 1,
        'AA+': 1, 'AA': 1, 'AA-': 1,
        'A+': 1, 'A': 1, 'A-': 1,
        'BBB+': 2, 'BBB': 2, 'BBB-': 2,
        'BB+': 3, 'BB': 3, 'BB-': 3,
        'B+': 4, 'B': 4, 'B-': 4,
        'CCC+': 4, 'CCC': 4, 'CCC-': 4,
        'CC': 5,
        'C': 5,
        'D': 5,
    }
    '''
    '''
    # 3 categories: IG, HY, Distress
    sp_rating_to_numeric = {
        'AAA': 1,
        'AA+': 1, 'AA': 1, 'AA-': 1,
        'A+': 1, 'A': 1, 'A-': 1,
        'BBB+': 1, 'BBB': 1, 'BBB-': 1,
        'BB+': 2, 'BB': 2, 'BB-': 2,
        'B+': 2, 'B': 2, 'B-': 2,
        'CCC+': 3, 'CCC': 3, 'CCC-': 3,
        'CC': 3,
        'C': 3,
        'D': 3,
    }
    '''

    # 3 categories: IG_High Grade, IG_Low Grade, HY
    sp_rating_to_numeric = {
        'AAA': 1,
        'AA+': 1, 'AA': 1, 'AA-': 1,
        'A+': 1, 'A': 1, 'A-': 1,
        'BBB+': 2, 'BBB': 2, 'BBB-': 2,
        'BB+': 3, 'BB': 3, 'BB-': 3,
        'B+': 3, 'B': 3, 'B-': 3,
        'CCC+': 3, 'CCC': 3, 'CCC-': 3,
        'CC': 3,
        'C': 3,
        'D': 3,
    }

    '''
    # 2 categories: IG, HY
    sp_rating_to_numeric = {
        'AAA': 1,
        'AA+': 1, 'AA': 1, 'AA-': 1,
        'A+': 1, 'A': 1, 'A-': 1,
        'BBB+': 1, 'BBB': 1, 'BBB-': 1,
        'BB+': 2, 'BB': 2, 'BB-': 2,
        'B+': 2, 'B': 2, 'B-': 2,
        'CCC+': 2, 'CCC': 2, 'CCC-': 2,
        'CC': 2,
        'C': 2,
        'D': 2,
    }
    '''
    # Apply the mapping to each column of the dataframe
    df_numeric = df_copy.apply(lambda col: col.map(sp_rating_to_numeric))

    return df_numeric

df_issuer_rating_numeric_daily = rating_to_numeric(df_issuer_rating_daily)


# Creation of the downgrade, upgrade dataframe
def create_rating_change_df(df, credit_event_type):
    df_copy = copy.deepcopy(df)
    # Create a new DataFrame with the same index and columns, filled with nan
    df_credit_change = pd.DataFrame(np.nan, index=df_copy.index, columns=df_copy.columns)

    # Set last_date
    last_date = df_copy.index[-1] - pd.DateOffset(years=1)

    for issuer in tqdm(df_copy.iloc[:, :].columns, desc='Creation of {} df'.format(credit_event_type)):

        # Use NumPy arrays for faster operations
        ratings = df_copy[issuer].values
        dates = df_copy.index

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

            # Check change in rating
            if credit_event_type == 'downgrade':
                rating_change_detected = np.any(future_ratings > rating_tmp)
            if credit_event_type == 'upgrade':
                rating_change_detected = np.any(future_ratings < rating_tmp)
            # Assign the result
            if np.isnan(rating_tmp):
                # if the actual rating is nan put nan for the rating change
                df_credit_change.iat[i, df_copy.columns.get_loc(issuer)] = np.nan
            else:
                df_credit_change.iat[i, df_copy.columns.get_loc(issuer)] = 1 if rating_change_detected else 0

    return df_credit_change

df_issuer_rating_downgrade_daily = create_rating_change_df(df_issuer_rating_numeric_daily, credit_event_type='downgrade')
df_issuer_rating_upgrade_daily = create_rating_change_df(df_issuer_rating_numeric_daily, credit_event_type='upgrade')

# Creation number of days since last change dataframe
def df_nd_days_since_last_change(df):
    df_copy = copy.deepcopy(df)

    for issuer in tqdm(df_copy.iloc[:, :].columns, desc='Creation nb days since last change df'):
        # Identify rating changes (True where the rating has changed)
        rating_change = df_copy[issuer] != df_copy[issuer].shift(1)

        # Count number of rating change
        count_rating_change = rating_change.cumsum()

        # For each group, subtract the first date of the group from all dates in that group
        time_since_last_change = df_copy[issuer].index.to_series().groupby(count_rating_change).transform(
            lambda x: x - x.iloc[0])

        # Convert to days
        time_since_last_change_in_days = time_since_last_change.dt.days

        df_copy[issuer] = time_since_last_change_in_days

    return df_copy

df_issuer_rating_nb_days_last_change_daily = df_nd_days_since_last_change(df_issuer_rating_numeric_daily)

# Creation number of days since last credit event dataframe
def df_nd_days_since_last_credit_event(df, credit_event_type):
    df_copy = copy.deepcopy(df)

    for issuer in tqdm(df_copy.iloc[:, :].columns, desc='Creation nb days since last {} df'.format(credit_event_type)):
        # Identify rating changes (True where the rating has changed)
        if credit_event_type == 'Downgrade':
            rating_change = (df_copy[issuer] > df_copy[issuer].shift(1))
        if credit_event_type == 'Upgrade':
            rating_change = (df_copy[issuer] < df_copy[issuer].shift(1))

        # Count number of rating change
        count_rating_change = rating_change.cumsum()

        # For each group, subtract the first date of the group from all dates in that group
        time_since_last_change = df_copy[issuer].index.to_series().groupby(count_rating_change).transform(
            lambda x: x - x.iloc[0])

        # Convert to days
        time_since_last_change_in_days = time_since_last_change.dt.days

        df_copy[issuer] = time_since_last_change_in_days

    return df_copy

df_issuer_rating_nb_days_last_downgrade_daily = df_nd_days_since_last_credit_event(df_issuer_rating_numeric_daily, credit_event_type= 'Downgrade')
df_issuer_rating_nb_days_last_upgrade_daily = df_nd_days_since_last_credit_event(df_issuer_rating_numeric_daily, credit_event_type= 'Upgrade')


# *** Convert daily data to monthly data ***

df_issuer_rating_numeric_monthly = df_issuer_rating_numeric_daily.resample('ME').ffill()

df_issuer_rating_downgrade_monthly = df_issuer_rating_downgrade_daily.resample('ME').ffill()
df_issuer_rating_upgrade_monthly = df_issuer_rating_upgrade_daily.resample('ME').ffill()

# Create a df with the nb of months since last rating change
def df_nb_months_since_last_rating_change(df):
    df_copy_daily = copy.deepcopy(df)
    df_copy_monthly = df_copy_daily.resample('ME').ffill()

    # Create a new DataFrame with the same index and columns, filled with nan
    df_output = pd.DataFrame(np.nan, index=df_copy_monthly.index, columns=df_copy_monthly.columns)

    for issuer in tqdm(df_copy_monthly.iloc[:, :].columns, desc='Creation of months since last change df'):
        # Find the last available date within each month
        s_last_available_dates = df_copy_daily[issuer].groupby(df_copy_daily[issuer].index.to_period('M')).apply(lambda x: x.index.max())
        # Set the index of the resampled data to the last available date within each month
        df_copy_monthly[issuer].index = s_last_available_dates
        # Find the rating change day
        s_rating_change_date = (df_copy_monthly[issuer].index - pd.to_timedelta(df_copy_monthly[issuer], unit='D')).resample('ME').ffill()

        for index_date, change_date in s_rating_change_date.items():
            # Calculate the nb of months since last rating change
            months_difference = relativedelta(index_date, change_date).months + (relativedelta(index_date, change_date).years * 12)

            df_output.loc[index_date, issuer] = months_difference

    return df_output

df_issuer_rating_nb_months_last_change_monthly = df_nb_months_since_last_rating_change(df_issuer_rating_nb_days_last_change_daily)

df_issuer_rating_nb_months_last_downgrade_monthly = df_nb_months_since_last_rating_change(df_issuer_rating_nb_days_last_downgrade_daily)
df_issuer_rating_nb_months_last_upgrade_monthly = df_nb_months_since_last_rating_change(df_issuer_rating_nb_days_last_upgrade_daily)

# TODO: Clean after
# Create a df with the nb of months until next rating change
def df_nb_months_until_next_rating_change(df):
    # Replace all values with -9999999 except where values are 0
    df_issuer_rating_nb_months_next_change = df.where(df == 0, -999999)

    for issuer in tqdm(df_issuer_rating_nb_months_next_change.iloc[:, :].columns, desc='Creation nb of months until next change df'):

        count = 0
        start_counting = False
        col = df_issuer_rating_nb_months_next_change[issuer]
        # Work backwards through the column
        for i in reversed(range(len(col))):
            if col.iloc[i] == 0:
                start_counting = True

            if start_counting == True:
                if col.iloc[i] == 0:
                    count = 0
                else:
                    count += 1
                    col.iloc[i] = count

    return df_issuer_rating_nb_months_next_change

df_issuer_rating_nb_months_next_downgrade_monthly = df_nb_months_until_next_rating_change(df_issuer_rating_nb_months_last_downgrade_monthly)
df_issuer_rating_nb_months_next_upgrade_monthly = df_nb_months_until_next_rating_change(df_issuer_rating_nb_months_last_upgrade_monthly)

# Create a df to identify the different credit events (create an id for each credit event)

def df_credit_event_id(df):
    # Replace all values with -999999 except where values are between 1 and 12
    df_issuer_credit_event_id = df.where((df >= 1) & (df <= 12), -999999)
    id = 0

    for issuer in tqdm(df_issuer_credit_event_id.iloc[:, :].columns, desc='Creation credit events id df'):

        #id = 0
        col = df_issuer_credit_event_id[issuer]
        # Work backwards through the column
        for i in reversed(range(len(col))):

            if col.iloc[i] == 1:
                id += 1
                col.iloc[i] = id
            elif (col.iloc[i] > 1) & (col.iloc[i] <= 12):
                col.iloc[i] = id

    return df_issuer_credit_event_id

#df_issuer_credit_event_id_monthly = df_credit_event_id(df_issuer_rating_nb_months_next_change_monthly)
df_issuer_downgrade_id_monthly = df_credit_event_id(df_issuer_rating_nb_months_next_downgrade_monthly)
df_issuer_upgrade_id_monthly = df_credit_event_id(df_issuer_rating_nb_months_next_upgrade_monthly)



# %%
# **********************************************************
# *** Section: Distance to Default                       ***
# **********************************************************


# *** Import data for DD computation ***

df_lt_debt_monthly = dic_issuer_fundamental_monthly['LT_DEBT']
df_st_debt_monthly = dic_issuer_fundamental_monthly['ST_DEBT']
# Computing debt level according to KMV assumption for DD computation and replace all 0 by nan
df_kmv_debt_monthly = df_st_debt_monthly + 0.5*df_lt_debt_monthly
df_kmv_debt_monthly = df_kmv_debt_monthly.replace(0, np.nan)

df_mkt_cap_daily = dic_issuer_market_daily['MKT_CAP']
# Resample by month and take the last available value within the month
df_mkt_cap_monthly = df_mkt_cap_daily.resample('ME').ffill()

# *** Volatility measurement ***

# Get the company share price
df_share_price_daily = dic_issuer_market_daily['SHARE_PRICE']
# Get the equity simple total return (not log return)
df_equity_tot_return_daily = dic_issuer_market_daily['TOT_RETURN']

# 1) Historical volatility

# Compute the volatility (annualised) of the equity total return using a rolling windows of one year,
# nan if less than 100 values available
df_historical_volatility_daily = df_equity_tot_return_daily.rolling(window=252, min_periods=100).std() * np.sqrt(252)
# Resample by month and take the last available value within the month
df_historical_volatility_daily_monthly = df_historical_volatility_daily.resample('ME').ffill()

# 2) AR(1)-GARCH(1,1) (in sample)
# TODO: Compute log return

def garch_volatility(df_returns):
    # Create a new DataFrame with the same index and columns, filled with nan
    df_conditional_volatility_t_1_daily = pd.DataFrame(np.nan, index=df_returns.index, columns=df_returns.columns)

    for issuer in df_returns:
        print(issuer)

        returns = df_returns[issuer]

        if returns.count() > 750:

            returns = returns.dropna()
            # Rescaling the returns to fit the GARCH
            returns = 100 * returns

            # Fit the AR(1)-GARCH(1,1) model
            am = arch_model(y=returns, mean='AR', lags=1, vol='GARCH', p=1, o=0, q=1, dist='normal')
            res = am.fit()
            # res.summary()
            if res.convergence_flag == 0:
                # Get the 1 period ahead conditional volatility (annualised)
                conditional_volatility_t_1_daily = res.conditional_volatility.shift(-1) / 100 * np.sqrt(252)

                df_conditional_volatility_t_1_daily[issuer] = conditional_volatility_t_1_daily

    # Take a rolling average of the conditional volatility to remove the noise
    df_conditional_volatility_t_1_daily = df_conditional_volatility_t_1_daily.rolling(window=90).mean()
    # Resample by month and take the last available value within the month
    df_conditional_volatility_t_1_monthly = df_conditional_volatility_t_1_daily.resample('ME').ffill()

    return df_conditional_volatility_t_1_daily, df_conditional_volatility_t_1_monthly

df_garch_conditional_volatility_t_1_daily, df_garch_conditional_volatility_t_1_monthly = garch_volatility(df_equity_tot_return_daily)

'''
plt.plot(df_garch_conditional_volatility_t_1_daily['AAL US Equity'])
plt.plot(df_historical_volatility_daily['AAL US Equity'])
plt.show()
'''

# Volatility measurement method choice
df_equity_return_vol_monthly = df_garch_conditional_volatility_t_1_monthly


# *** Risk free rate ***

# 3m US treasury bill rate (annualised)
s_3m_us_treasury_bill_rate_daily = dic_market_data_daily['RATES']['GB3 Govt'] / 100
# Resample by month and take the last available value within the month
s_3m_us_treasury_bill_rate_monthly = s_3m_us_treasury_bill_rate_daily.resample('ME').ffill()

# Get a constant risk-free rate
treasury_bill_rate_daily_constant = 0.02
# Replace all the values in the series with the constant
s_3m_us_treasury_bill_rate_const_daily = s_3m_us_treasury_bill_rate_daily.apply(lambda x: treasury_bill_rate_daily_constant)
s_3m_us_treasury_bill_rate_const_monthly = s_3m_us_treasury_bill_rate_const_daily.resample('ME').ffill()


# *** Select data between a date range ***

# Define the date range
start_date = '2010-12-31'
end_date = '2023-12-31'

df_kmv_debt_monthly = df_kmv_debt_monthly.loc[start_date:end_date]
df_mkt_cap_monthly = df_mkt_cap_monthly.loc[start_date:end_date]
df_equity_return_vol_monthly = df_equity_return_vol_monthly.loc[start_date:end_date]
s_3m_us_treasury_bill_rate_monthly = s_3m_us_treasury_bill_rate_monthly.loc[start_date:end_date]
s_3m_us_treasury_bill_rate_const_monthly = s_3m_us_treasury_bill_rate_const_monthly.loc[start_date:end_date]

# *** DD computation ***

def d1(V, K, r, g, sV, T, t):
    d1 = (np.log(V/K) + ((r-g) + 0.5*sV**2)*(T-t))/(sV*np.sqrt(T-t))
    return d1

def d2(V, K, r, g, sV, T, t):
    d2 = d1(V, K, r, g, sV, T, t) - sV*np.sqrt(T-t)
    return d2

'''
d1(V=100,K=50,r=0.05,g=0,sV=0.1,T=1,t=0)
d2(V=100,K=50,r=0.05,g=0,sV=0.1,T=1,t=0)
'''

# Estimation of Vt and sV - Method 1: Fixing Vt


# Estimation of Vt and sV - Method 2: Implied Vt and sV from Merton Model
def get_merton_implied_V_sV(E, K, r, g, sE, T, t):

    def merton_V_sV_equations(vars):
        V, sV = vars
        eq1 = np.exp(-r*(T-t)) * (V * np.exp((r-g) * (T-t)) * stats.norm.cdf(d1(V=V, K=K, r=r, g=g, sV=sV, T=T, t=t))
                                 - K * stats.norm.cdf(d2(V=V, K=K, r=r, g=g, sV=sV, T=T, t=t))) - E
        eq2 = sE * E - sV * V * stats.norm.cdf(d1(V=V, K=K, r=r, g=g, sV=sV, T=T, t=t))
        return [eq1, eq2]

    x0 = np.array([E+K, (E / (E+K)) * sE])

    solution = root(fun=merton_V_sV_equations, x0=x0, method='hybr', tol=None, options=None)

    return solution

'''
E = 20
K = 99.46
r = 0.1
g = 0
sE = 0.4
T = 1
t = 0

E = 12540467717.28516
K = 1357000000.00000
r = 0.10902
g = 0
sE = 0.34231
T = 1
t = 0

solution = get_merton_implied_V_sV(E,K,r,g,sE,T,t)
print(solution)
print(solution.x)
print(solution.success)
'''

def get_df_V_sV(df_E, df_K, df_sE, s_r, g, T, t):
    df_V_output = pd.DataFrame(np.nan, index=df_E.index, columns=df_E.columns)
    df_sV_output = pd.DataFrame(np.nan, index=df_E.index, columns=df_E.columns)

    for c in tqdm(df_E.columns, desc='Merton implied V and sV'):
        for i in df_E.index:
            E = df_E.loc[i, c]
            K = df_K.loc[i, c]
            sE = df_sE.loc[i, c]
            r = s_r.loc[i]

            solution = get_merton_implied_V_sV(E=E, K=K, r=r, g=g, sE=sE, T=T, t=t)
            #if not solution.success:
                #print(c, i)
            V = solution.x[0]
            sV = solution.x[1]

            df_V_output.loc[i, c] = V
            df_sV_output.loc[i, c] = sV

    return df_V_output, df_sV_output


df_ev_monthly, df_ev_vol_monthly = get_df_V_sV(df_E=df_mkt_cap_monthly,
                                               df_K=df_kmv_debt_monthly,
                                               df_sE=df_equity_return_vol_monthly,
                                               s_r=s_3m_us_treasury_bill_rate_const_monthly, g=0, T=1, t=0)

def df_DD(df_V, df_K, df_sV, s_r, g, T, t):
    df_DD_output = pd.DataFrame(np.nan, index=df_V.index, columns=df_V.columns)

    for c in tqdm(df_V.columns, desc='DD Computation'):
        for i in df_V.index:
            V = df_V.loc[i, c]
            K = df_K.loc[i, c]
            sV = df_sV.loc[i, c]
            r = s_r.loc[i]
            DD = d1(V, K, r, g, sV, T, t)

            df_DD_output.loc[i, c] = DD

    return df_DD_output

df_DD_monthly = df_DD(df_V=df_ev_monthly,
                      df_K=df_kmv_debt_monthly,
                      df_sV=df_ev_vol_monthly,
                      s_r=s_3m_us_treasury_bill_rate_const_monthly, g=0, T=1, t=0)


# %%
# **********************************************************
# *** Section: Variables definition                      ***
# **********************************************************


# *** Firm specific variables ***

# Z-score variables
df_wc_to_ta_monthly = dic_issuer_fundamental_monthly['WORK_CAP'] / dic_issuer_fundamental_monthly['TOT_ASSET']
df_re_to_ta_monthly = dic_issuer_fundamental_monthly['RET_EARN'] / dic_issuer_fundamental_monthly['TOT_ASSET']
df_ebit_to_ta_monthly = dic_issuer_fundamental_monthly['EBIT'] / dic_issuer_fundamental_monthly['TOT_ASSET']
df_me_to_td_monthly = df_mkt_cap_monthly / ((dic_issuer_fundamental_monthly['LT_DEBT'] + dic_issuer_fundamental_monthly['ST_DEBT']).replace(0, np.nan))
df_sales_to_ta_monthly = dic_issuer_fundamental_monthly['SALES'] / dic_issuer_fundamental_monthly['TOT_ASSET']

# Duan (2012)
df_cash_to_ta_monthly = dic_issuer_fundamental_monthly['CASH'] / dic_issuer_fundamental_monthly['TOT_ASSET']
df_ev_to_ta_monthly = (df_mkt_cap_monthly + dic_issuer_fundamental_monthly['LT_DEBT'] + dic_issuer_fundamental_monthly['ST_DEBT']) / dic_issuer_fundamental_monthly['TOT_ASSET']
df_size_monthly = np.log(df_mkt_cap_monthly)

# Other credit metrics
df_coverage_monthly = dic_issuer_fundamental_monthly['EBIT'] / dic_issuer_fundamental_monthly['INTEREST_EXP']
df_net_leverage_monthly = (dic_issuer_fundamental_monthly['LT_DEBT'] + dic_issuer_fundamental_monthly['ST_DEBT'] - dic_issuer_fundamental_monthly['CASH']) / dic_issuer_fundamental_monthly['EBITDA']


# *** Common variables ***
s_spx_level_daily = dic_market_data_daily['SPX']['px_last']
s_spx_level_monthly = s_spx_level_daily.resample('ME').ffill()
s_spx_1y_trailing_return_monthly = s_spx_level_monthly.pct_change(periods=12)

s_3m_us_treasury_bill_rate_monthly = s_3m_us_treasury_bill_rate_monthly


# *** Add all variables in a dictionary ***

dic_variables_monthly = {}
dic_dep_variables_firm_monthly = {}
dic_ind_variables_firm_monthly = {}
dic_ind_variables_common_monthly = {}

# Dependent variables
dic_dep_variables_firm_monthly['next_12m_downgrade'] = df_issuer_rating_downgrade_monthly
dic_dep_variables_firm_monthly['next_12m_upgrade'] = df_issuer_rating_upgrade_monthly

#dic_dep_variables_firm_monthly['credit_event_id'] = df_issuer_credit_event_id_monthly
#dic_dep_variables_firm_monthly['time_until_next_change'] = df_issuer_rating_nb_months_next_change_monthly
dic_dep_variables_firm_monthly['time_until_next_downgrade'] = df_issuer_rating_nb_months_next_downgrade_monthly
dic_dep_variables_firm_monthly['time_until_next_upgrade'] = df_issuer_rating_nb_months_next_upgrade_monthly
dic_dep_variables_firm_monthly['downgrade_id'] = df_issuer_downgrade_id_monthly
dic_dep_variables_firm_monthly['upgrade_id'] = df_issuer_upgrade_id_monthly
dic_dep_variables_firm_monthly['rating_cat_id'] = df_issuer_rating_numeric_monthly

# Independent variables - Firm specific variables
dic_ind_variables_firm_monthly['DD'] = df_DD_monthly

dic_ind_variables_firm_monthly['WC/TA'] = df_wc_to_ta_monthly
dic_ind_variables_firm_monthly['RE/TA'] = df_re_to_ta_monthly
dic_ind_variables_firm_monthly['EBIT/TA'] = df_ebit_to_ta_monthly
#dic_ind_variables_firm_monthly['ME/TD'] = df_me_to_td_monthly
dic_ind_variables_firm_monthly['SALES/TA'] = df_sales_to_ta_monthly

dic_ind_variables_firm_monthly['EV/TA'] = df_ev_to_ta_monthly
dic_ind_variables_firm_monthly['CASH/TA'] = df_cash_to_ta_monthly
dic_ind_variables_firm_monthly['SIZE'] = df_size_monthly

#dic_ind_variables_firm_monthly['coverage'] = df_coverage_monthly
#dic_ind_variables_firm_monthly['net_leverage'] = df_net_leverage_monthly

# Independent variables - Common variables
dic_ind_variables_common_monthly['spx_1y_trailing_return'] = s_spx_1y_trailing_return_monthly
dic_ind_variables_common_monthly['3m_us_treasury_bill_rate'] = s_3m_us_treasury_bill_rate_monthly

# Combine the 2 dictionary in one
dic_variables_monthly['dep_var_firm'] = dic_dep_variables_firm_monthly
dic_variables_monthly['ind_var_firm'] = dic_ind_variables_firm_monthly
dic_variables_monthly['ind_var_common'] = dic_ind_variables_common_monthly

# Select data between a date range
def dic_define_date_range(dic, start, end):
    dic_copy = copy.deepcopy(dic)
    for i in dic_copy:
        dic_copy[i] = dic_copy[i].loc[start:end]
    return dic_copy

def dic_parent_define_date_range(dic, start, end):
    dic_copy = copy.deepcopy(dic)
    for i in dic_copy:
        dic_copy[i] = dic_define_date_range(dic_copy[i], start, end)
    return dic_copy


start_date = '2010-12-31'
end_date = '2023-12-31'

dic_variables_monthly = dic_parent_define_date_range(dic_variables_monthly, start_date, end_date)


# *** Compute trend for firm specific variables: difference between current value and 12 months ago value ***

def dic_ind_var_firm_trend(dic_variables, df_lag):

    dic = dic_variables['ind_var_firm']
    dic_tmp = {}

    for var in dic:
        df_tmp = pd.DataFrame(np.nan, index=dic[var].index, columns=dic[var].columns)

        for firm in tqdm(dic[var].iloc[:, :], desc="Creating df trend for {}".format(var)):

            for i in dic[var][firm].index:
                # Get the position of the specific index in the Series
                position = dic[var][firm].index.get_loc(i)

                if df_lag[firm].loc[i] < 12:
                    if position - int(df_lag[firm].loc[i]) >= 0:
                        var_delta = dic[var][firm].loc[i] - dic[var][firm].iloc[position - int(df_lag[firm].loc[i])]
                        df_tmp.loc[i, firm] = var_delta
                else:
                    if position - 12 >= 0:
                        var_delta = dic[var][firm].loc[i] - dic[var][firm].iloc[position - 12]
                        df_tmp.loc[i, firm] = var_delta

        dic_tmp['{}_trend'.format(var)] = df_tmp

    dic_variables['ind_var_firm_trend'] = dic_tmp

    return dic_variables

dic_variables_monthly = dic_ind_var_firm_trend(dic_variables_monthly, df_issuer_rating_nb_months_last_change_monthly)

# *** Compute level for firm specific variables: take the 12 months ago level of the variable ***

def dic_ind_var_firm_level(dic_variables, lag):

    dic = dic_variables['ind_var_firm']

    for var in dic:
        dic[var] = dic[var].shift(lag)

    return dic_variables

dic_variables_monthly = dic_ind_var_firm_level(dic_variables_monthly, lag=-12)

'''
zzz = dic_variables_monthly['ind_var_firm']['DD']
yyy = dic_variables_monthly['ind_var_firm']['DD'].shift(-12)
'''

# Date filtering
start_date = '2011-12-31'
end_date = '2022-12-31'

dic_variables_monthly = dic_parent_define_date_range(dic_variables_monthly, start_date, end_date)


# *** Combine all data in a dataframe ***

def df_combined_all_data(dic):
    # firm variables
    ls_output = []
    data_type = ['dep_var_firm', 'ind_var_firm', 'ind_var_firm_trend']
    for i in data_type:
        dic_tpm = dic[i]

        ls_reshaped_data = []
        for variable, df in dic_tpm.items():
            df.index.name = 'DATES'
            df_melt = pd.melt(df.reset_index(), id_vars=['DATES'], var_name='Issuer', value_vars=df.columns,
                              value_name=variable)
            ls_reshaped_data.append(df_melt)

        # Merge all reshaped DataFrames on the date 'DATES' and 'Issuer' columns
        df_tpm = ls_reshaped_data[0]
        for df in ls_reshaped_data[1:]:
            df_tpm = pd.merge(df_tpm, df, on=['DATES', 'Issuer'])

        ls_output.append(df_tpm)

    df_output = ls_output[0]
    for df in ls_output[1:]:
        df_output = pd.merge(df_output, df, on=['DATES', 'Issuer'])

    # common variables
    ls_output = []
    data_type = ['ind_var_common']

    for i in data_type:
        dic_tpm = dic[i]

        ls_reshaped_data = []
        for variable, df in dic_tpm.items():
            df.index.name = 'DATES'
            df.name = variable
            ls_reshaped_data.append(df)

        # Merge all reshaped DataFrames on the date 'DATES' and 'Issuer' columns
        df_tpm = ls_reshaped_data[0]
        for df in ls_reshaped_data[1:]:
            df_tpm = pd.merge(df_tpm, df, on=['DATES'])

        ls_output.append(df_tpm)

    for df in ls_output[:]:
        df_output = pd.merge(df_output, df, on=['DATES'])

    return df_output

df_data = df_combined_all_data(dic_variables_monthly)

# Drop all rows that contain any NaN values
df_data = df_data.dropna()

# Save data (uncomment)
with open(Path.joinpath(paths.get('data'), 'df_data.pkl'), 'wb') as file:
     pickle.dump(df_data, file)



# %%
# **********************************************************
# *** Section: Summary Statics - Data Exploration        ***
# **********************************************************


# Load data
with open(Path.joinpath(paths.get('data'), 'df_data.pkl'), 'rb') as file:
    df_data = pickle.load(file)


# *** Data visualisation ***

# Check if the dependent variables are balanced
print('Check if the dependent variables are balanced:')
# Downgrade
count_no_downgrade = len(df_data[df_data['next_12m_downgrade'] == 0])
count_downgrade = len(df_data[df_data['next_12m_downgrade'] == 1])
pct_of_no_downgrade = count_no_downgrade/(count_no_downgrade + count_downgrade)
print('percentage of no downgrade:', pct_of_no_downgrade*100)
pct_of_downgrade = count_downgrade/(count_no_downgrade + count_downgrade)
print('percentage of downgrade:', pct_of_downgrade*100)
# Upgrade
count_no_upgrade = len(df_data[df_data['next_12m_upgrade'] == 0])
count_upgrade = len(df_data[df_data['next_12m_upgrade'] == 1])
pct_of_no_upgrade = count_no_upgrade/(count_no_upgrade + count_upgrade)
print('percentage of no upgrade:', pct_of_no_upgrade*100)
pct_of_upgrade = count_upgrade/(count_no_upgrade + count_upgrade)
print('percentage of upgrade:', pct_of_upgrade*100)

# Plot Downgrade distribution
sns.set(context='paper', style='ticks', palette='bright', font_scale=1.0)
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
ax.set_title('Proportion of Downgrade', size=28)
sns.countplot(data=df_data, x='next_12m_downgrade', stat='proportion')
ax.tick_params(axis='both', labelsize=18)
ax.set_xlabel('', size=20)
ax.set_ylabel('Proportion', size=20)
ax.grid(axis='y', alpha=0.4)
fig.tight_layout()
plt.show()
#fig.savefig(Path.joinpath(paths.get('output'), 'XXXXXXXXX'.png'))
plt.close()

# Plot Upgrade distribution
sns.set(context='paper', style='ticks', palette='bright', font_scale=1.0)
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
ax.set_title('Proportion of Upgrade', size=28)
sns.countplot(data=df_data, x='next_12m_upgrade', stat='proportion')
ax.tick_params(axis='both', labelsize=18)
ax.set_xlabel('', size=20)
ax.set_ylabel('Proportion', size=20)
ax.grid(axis='y', alpha=0.4)
fig.tight_layout()
plt.show()
#fig.savefig(Path.joinpath(paths.get('output'), 'XXXXXXXXX'.png'))
plt.close()

# Plot the distribution of the different independent variables
df_data_ind_variables = df_data.drop(['DATES', 'Issuer', 'next_12m_downgrade', 'next_12m_upgrade',
                                      'downgrade_id', 'upgrade_id', 'rating_cat_id',
                                      'time_until_next_downgrade', 'time_until_next_upgrade'], axis=1)
for i in df_data_ind_variables:
    print(i)
    sns.set(context='paper', style='ticks', palette='bright', font_scale=1.0)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    ax.set_title('Distribution of {}'.format(i), size=28)
    sns.histplot(df_data[i], stat='density', kde=False)
    ax.tick_params(axis='both', labelsize=18)
    ax.set_xlabel('{}'.format(i), size=20)
    ax.set_ylabel('Density', size=20)
    ax.grid(axis='y', alpha=0.4)
    fig.tight_layout()
    plt.show()
    #fig.savefig(Path.joinpath(paths.get('output'), 'XXXXXXXXX'.png'))
    plt.close()
    time.sleep(0.5)


# *** Summary statistics ***
df_data_variables = df_data.drop(['DATES', 'Issuer',
                                  'downgrade_id', 'upgrade_id', 'rating_cat_id',
                                  'time_until_next_downgrade', 'time_until_next_upgrade'], axis=1)
df_data_describe = df_data_variables.describe()

#df_data_downgrade = df_data.drop(['DATES', 'Issuer', 'next_12m_upgrade'], axis=1)
#df_data_upgrade = df_data.drop(['DATES', 'Issuer', 'next_12m_downgrade'], axis=1)
df_downgrade_mean = df_data_variables.groupby('next_12m_downgrade').mean()
df_upgrade_mean = df_data_variables.groupby('next_12m_upgrade').mean()


# %%
# **********************************************************
# *** Section: Preprocessing - Resampling: Over-sampling ***
# **********************************************************

# TODO: change text for upgrade for check data

def resampling_data(dep_var, df_data, dep_var_to_remove):

    # Drop all rows where 'rating_cat_id' equals max id (downgrade) or min id (upgrade)
    if dep_var == 'next_12m_downgrade':
        df_data = df_data[df_data['rating_cat_id'] != df_data['rating_cat_id'].max()]
    if dep_var == 'next_12m_upgrade':
        df_data = df_data[df_data['rating_cat_id'] != df_data['rating_cat_id'].min()]

    df_data_dep_var = df_data.drop(['DATES', 'Issuer',
                                    'downgrade_id', 'upgrade_id',
                                    'time_until_next_downgrade', 'time_until_next_upgrade', dep_var_to_remove], axis=1)
    df_data_dep_var_X = df_data_dep_var.loc[:, df_data_dep_var.columns != dep_var]
    df_data_dep_var_y = df_data_dep_var[dep_var]

    # Oversampling the data using SMOTE method
    sm = SMOTE(random_state=42)
    df_data_dep_var_X_res, df_data_dep_var_y_res = sm.fit_resample(df_data_dep_var_X, df_data_dep_var_y)

    # Check data after resampling
    print('{} - Check data after resampling:'.format(dep_var))
    print('Number of observations after resampling: ', len(df_data_dep_var_X_res))
    print('Number of no {} after resampling:'.format(dep_var), len(df_data_dep_var_y_res[df_data_dep_var_y_res == 0]))
    print('Number of {} after resampling:'.format(dep_var), len(df_data_dep_var_y_res[df_data_dep_var_y_res == 1]))
    print('Proportion of no {} in oversampled data is '.format(dep_var), len(df_data_dep_var_y_res[df_data_dep_var_y_res == 0]) / len(df_data_dep_var_X_res))
    print('Proportion of {} in oversampled data is '.format(dep_var), len(df_data_dep_var_y_res[df_data_dep_var_y_res == 1]) / len(df_data_dep_var_X_res))

    df_data_dep_var_res = df_data_dep_var_X_res.join(df_data_dep_var_y_res)
    # Reorder the columns to put the y column first
    cols = [df_data_dep_var_y_res.name] + [col for col in df_data_dep_var_res.columns if col != df_data_dep_var_y_res.name]
    df_data_dep_var_res = df_data_dep_var_res[cols]

    # Round rating category
    df_data_dep_var_res['rating_cat_id'] = df_data_dep_var_res['rating_cat_id'].round()

    # ***
    # Scatter plot df_data_dep_var_res before and after resampling
    # Before resampling
    sns.set(context='paper', style='ticks', palette='bright', font_scale=1.0)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    # ax.set_title('', size=28)
    ax.scatter(x=df_data_dep_var[df_data_dep_var[dep_var] == 0]['EBIT/TA_trend'],
               y=df_data_dep_var[df_data_dep_var[dep_var] == 0]['DD_trend'],
               label='No {}'.format(dep_var))
    ax.scatter(x=df_data_dep_var[df_data_dep_var[dep_var] == 1]['EBIT/TA_trend'],
               y=df_data_dep_var[df_data_dep_var[dep_var] == 1]['DD_trend'],
               label='{}'.format(dep_var))
    ax.tick_params(axis='both', labelsize=18)
    ax.set_xlabel('EBIT/TA_trend', size=20)
    ax.set_ylabel('DD_trend', size=20)
    # ax.grid(axis='y', alpha=0.4)
    plt.legend()
    fig.tight_layout()
    plt.show()
    # fig.savefig(Path.joinpath(paths.get('output'), 'XXXXXXXXX'.png'))
    plt.close()

    # After resampling
    sns.set(context='paper', style='ticks', palette='bright', font_scale=1.0)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    # ax.set_title('', size=28)
    ax.scatter(x=df_data_dep_var_res[df_data_dep_var_res[dep_var] == 0]['EBIT/TA_trend'],
               y=df_data_dep_var_res[df_data_dep_var_res[dep_var] == 0]['DD_trend'],
               label='No {}'.format(dep_var))
    ax.scatter(x=df_data_dep_var_res[df_data_dep_var_res[dep_var] == 1]['EBIT/TA_trend'],
               y=df_data_dep_var_res[df_data_dep_var_res[dep_var] == 1]['DD_trend'],
               label='{}'.format(dep_var))
    ax.tick_params(axis='both', labelsize=18)
    ax.set_xlabel('EBIT/TA_trend', size=20)
    ax.set_ylabel('DD_trend', size=20)
    # ax.grid(axis='y', alpha=0.4)
    plt.legend()
    fig.tight_layout()
    plt.show()
    # fig.savefig(Path.joinpath(paths.get('output'), 'XXXXXXXXX'.png'))
    plt.close()

    return df_data_dep_var, df_data_dep_var_res


# Downgrade
df_data_downgrade, df_data_downgrade_res = resampling_data(dep_var='next_12m_downgrade', df_data=df_data, dep_var_to_remove='next_12m_upgrade')

# Upgrade
df_data_upgrade, df_data_upgrade_res = resampling_data(dep_var='next_12m_upgrade', df_data=df_data, dep_var_to_remove='next_12m_downgrade')


# %%
# **********************************************************
# *** Section:  Logit regression Fit                     ***
# **********************************************************


# Creation of data test dataframe (real data)
df_data_test_downgrade = df_data[df_data['rating_cat_id'] != df_data['rating_cat_id'].max()]
df_data_test_upgrade = df_data[df_data['rating_cat_id'] != df_data['rating_cat_id'].min()]

# Addition of interaction effect (resample + real data)
def addition_interaction(df, var_range):
    df_copy = copy.deepcopy(df)

    # Creation of dummy variables, with rating == 2 as the reference
    df_copy = pd.get_dummies(df_copy, columns=['rating_cat_id'], prefix='rating', dtype=int).drop(columns=['rating_2.0'])
    binary_var = df_copy.columns[-1]
    variables_for_interaction = df_copy.columns[var_range[0]:var_range[1]].tolist()

    # Create interaction terms and add them to the DataFrame
    for var in variables_for_interaction:
        interaction_column_name = f'{var}:{binary_var}'
        df_copy[interaction_column_name] = df_copy[var] * df_copy[binary_var]
    return df_copy

# Resampled data
df_data_downgrade_res = addition_interaction(df=df_data_downgrade_res, var_range = [1,1+8])
df_data_upgrade_res = addition_interaction(df=df_data_upgrade_res, var_range = [1,1+8])

# Real data
# Data not resampled
df_data_downgrade = addition_interaction(df=df_data_downgrade, var_range = [1,1+8])
df_data_upgrade = addition_interaction(df=df_data_upgrade, var_range = [1,1+8])

# Test data
df_data_test_downgrade = addition_interaction(df=df_data_test_downgrade, var_range = [8,8+8])
df_data_test_upgrade = addition_interaction(df=df_data_test_upgrade, var_range = [8,8+8])


# *** Downgrade ***

# Fitting the logit regression (resample data)
df_data_downgrade_res_X = df_data_downgrade_res.loc[:, df_data_downgrade_res.columns != 'next_12m_downgrade']
df_data_downgrade_res_X = sm.add_constant(df_data_downgrade_res_X)
df_data_downgrade_res_y = df_data_downgrade_res['next_12m_downgrade']

logit_mod = sm.Logit(df_data_downgrade_res_y, df_data_downgrade_res_X)
logit_res = logit_mod.fit(cov_type='HC3')
print(logit_res.summary2())
print(logit_res.summary())

df_downgrade_params = pd.Series(logit_res.params)

# Model estimated rating downgrade probability within next 12m (real data)
#df_data_test = copy.deepcopy(df_data)
#df_data_test_y = df_data_test['next_12m_downgrade']

df_data_test_downgrade['y_index_downgrade_predict'] = (sm.add_constant(df_data_downgrade.loc[:, df_data_downgrade.columns != 'next_12m_downgrade']) * df_downgrade_params).sum(axis=1)
df_data_test_downgrade['estimated_proba_downgrade_next_12m'] = 1 / (1 + np.exp(-df_data_test_downgrade['y_index_downgrade_predict']))

# Marginal effects



# *** Upgrade ***

# Fitting the logit regression (resample data)
df_data_upgrade_res_X = df_data_upgrade_res.loc[:, df_data_upgrade_res.columns != 'next_12m_upgrade']
df_data_upgrade_res_X = sm.add_constant(df_data_upgrade_res_X)
df_data_upgrade_res_y = df_data_upgrade_res['next_12m_upgrade']

logit_mod = sm.Logit(df_data_upgrade_res_y, df_data_upgrade_res_X)
logit_res = logit_mod.fit(cov_type='HC3')
print(logit_res.summary2())
print(logit_res.summary())

df_upgrade_params = pd.Series(logit_res.params)

# Model estimated rating upgrade probability within next 12m (real data)
df_data_test_upgrade['y_index_upgrade_predict'] = (sm.add_constant(df_data_upgrade.loc[:, df_data_upgrade.columns != 'next_12m_upgrade']) * df_upgrade_params).sum(axis=1)
df_data_test_upgrade['estimated_proba_upgrade_next_12m'] = 1 / (1 + np.exp(-df_data_test_upgrade['y_index_upgrade_predict']))


# Marginal effects


# %%
# ******************************************************************************
# *** Section:  Evaluation of the models (real data) - In sample validations ***
# ******************************************************************************

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

# *** Downgrade: Evaluation of the model (real data) ***
# *** Classification model performance ***

df_data_test_downgrade_y_true = df_data_test_downgrade['next_12m_downgrade']

# *** Tuning the decision threshold ***

# ROC
df_data_test_downgrade_estimated_proba = df_data_test_downgrade['estimated_proba_downgrade_next_12m']

fpr, tpr, thresholds = roc_curve(df_data_test_downgrade_y_true, df_data_test_downgrade_estimated_proba)
logit_roc_auc = roc_auc_score(df_data_test_downgrade_y_true, df_data_test_downgrade_estimated_proba)

sns.set(context='paper', style='ticks', palette='bright', font_scale=1.0)
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
ax.set_title('ROC Curve', size=28)
ax.plot(fpr, tpr, label='Logit Model (AUC = %0.2f)' % logit_roc_auc, linewidth=2)
ax.plot([0, 1], [0, 1],'r--', label='Random Classifier (AUC = 0.50)')
ax.tick_params(axis='both', labelsize=18)
ax.set_xlabel('False Positive Rate (1 - Specificity)', size=20)
ax.set_ylabel('True Positive Rate (Sensitivity)', size=20)
ax.grid(True, alpha=0.4)
plt.legend(loc='lower right', fontsize=15)
fig.tight_layout()
plt.show()
# fig.savefig(Path.joinpath(paths.get('output'), 'XXXXXXXXX'.png'))
plt.close()

# Precision-Sensitivity (Recall) Curve
precision, recall, thresholds = precision_recall_curve(df_data_test_downgrade_y_true, df_data_test_downgrade_estimated_proba)
logit_average_precision = average_precision_score(df_data_test_downgrade_y_true, df_data_test_downgrade_estimated_proba)
positive_class_frequency = df_data_test_downgrade_y_true.sum() / len(df_data_test_downgrade_y_true)

sns.set(context='paper', style='ticks', palette='bright', font_scale=1.0)
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
ax.set_title('Precision-Sensitivity Curve', size=28)
ax.plot(recall, precision, label='Logit Model (AP = %0.2f)' % logit_average_precision, linewidth=2)
ax.plot([0, 1], [positive_class_frequency, positive_class_frequency],'r--', label='Random Classifier (AP = %0.2f)' % positive_class_frequency)
ax.tick_params(axis='both', labelsize=18)
ax.set_xlabel('Sensitivity', size=20)
ax.set_ylabel('Precision', size=20)
ax.grid(True, alpha=0.4)
plt.legend(loc='upper right', fontsize=15)
fig.tight_layout()
plt.show()
# fig.savefig(Path.joinpath(paths.get('output'), 'XXXXXXXXX'.png'))
plt.close()

# Find the optimal threshold - F1 score

thresholds = np.append(thresholds, 1)

df_precision_recall_downgrade = pd.DataFrame({
    'precision': precision,
    'recall': recall,
    'thresholds': thresholds
})

df_precision_recall_downgrade['f1_score'] = 2 * (precision * recall) / (precision + recall)

# Plot F1-score, Precision, and Recall
sns.set(context='paper', style='ticks', palette='bright', font_scale=1.0)
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
#ax.set_title('', size=28)
ax.plot(df_precision_recall_downgrade['thresholds'], df_precision_recall_downgrade['f1_score'], label= 'F1-score', linewidth=2)
ax.plot(df_precision_recall_downgrade['thresholds'], df_precision_recall_downgrade['precision'], label= 'Precision', linestyle='-',  linewidth=2)
ax.plot(df_precision_recall_downgrade['thresholds'], df_precision_recall_downgrade['recall'], label= 'Sensitivity (Recall)', linestyle='-',  linewidth=2)

# Add a horizontal line at y=0.75
recall_target = 0.75
ax.axhline(y=recall_target, linestyle='--', linewidth=1.5, label=f'Sensitivity (Recall) = {recall_target}')

# Find the threshold where Recall crosses y=0.75
recall = df_precision_recall_downgrade['recall']
precision = df_precision_recall_downgrade['precision']
thresholds = df_precision_recall_downgrade['thresholds']

# Find closest threshold
closest_idx = (np.abs(recall - recall_target)).idxmin()  # Index where recall is closest to y=0.75
threshold_at_y = thresholds[closest_idx]
recall_value_at_threshold = recall[closest_idx]
precision_value_at_threshold = precision[closest_idx]

# Add horizontal line at recall_value_at_threshold
ax.axhline(y=precision_value_at_threshold, linestyle='--', linewidth=1.5, label=f'Precision = {precision_value_at_threshold:.2f}')

# Add vertical line at the threshold
ax.axvline(x=threshold_at_y, linestyle='--', linewidth=1.5, label=f'Threshold = {threshold_at_y:.2f}')

# Adjust labels, grid, and legend
ax.tick_params(axis='both', labelsize=18)
ax.set_xlabel('Thresholds', size=20)
#ax.set_ylabel('F1-score', size=20)
ax.grid(True, alpha=0.4)
plt.legend(fontsize=15)
fig.tight_layout()
plt.show()
# fig.savefig(Path.joinpath(paths.get('output'), 'XXXXXXXXX'.png'))
plt.close()


# *** Other mesures of performance ***

# Evolution of proba of downgrade between 36 months to 1 month to downgrade
df_data_downgrade_test_stats_monthly = df_data_test_downgrade.drop(['DATES', 'Issuer'], axis=1).groupby('time_until_next_downgrade').mean()
y = df_data_downgrade_test_stats_monthly['estimated_proba_downgrade_next_12m'][(df_data_downgrade_test_stats_monthly.index >= 1) & (df_data_downgrade_test_stats_monthly.index <= 36)]

sns.set(context='paper', style='ticks', palette='bright', font_scale=1.0)
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
# ax.set_title('', size=28)
ax.plot(y, marker='o', linestyle='-', linewidth=2)
ax.tick_params(axis='both', labelsize=18)
ax.set_xticks(np.arange(0, len(y), 6))
ax.set_xlabel('Nb of months until downgrade', size=20)
ax.set_ylabel('Estimated probability of downgrade within 12m', size=20)
ax.grid(True, alpha=0.4)
#plt.legend()
fig.tight_layout()
plt.show()
# fig.savefig(Path.joinpath(paths.get('output'), 'XXXXXXXXX'.png'))
plt.close()


# *** Class prediction (for a given threshold) ***
threshold = threshold_at_y
df_data_test_downgrade['y_pred'] = [1 if p > threshold else 0 for p in df_data_test_downgrade['estimated_proba_downgrade_next_12m']]
df_data_test_downgrade_y_pred = df_data_test_downgrade['y_pred']

# Confusion Matrix (for a given threshold)
confusion_matrix = confusion_matrix(df_data_test_downgrade_y_true, df_data_test_downgrade_y_pred)

# Classification report (for a given threshold)
classification_report = classification_report(df_data_test_downgrade_y_true, df_data_test_downgrade_y_pred)

# Sensitivity (Recall) (for a given threshold)
recall_score = recall_score(df_data_test_downgrade_y_true, df_data_test_downgrade_y_pred, average='binary')

# Precision (for a given threshold)
precision_score = precision_score(df_data_test_downgrade_y_true, df_data_test_downgrade_y_pred, average='binary')

# Accuracy (for a given threshold)
accuracy_score = accuracy_score(df_data_test_downgrade_y_true, df_data_test_downgrade_y_pred)

# Balance accuracy (for a given threshold)
balanced_accuracy = balanced_accuracy_score(df_data_test_downgrade_y_true, df_data_test_downgrade_y_pred)

# Coefficient of discriminations
df_data_test_downgrade_mean = df_data_test_downgrade.drop(['DATES', 'Issuer'], axis=1).groupby('next_12m_downgrade').mean()
s_data_test_downgrade_mean_proba = pd.Series(df_data_test_downgrade_mean['estimated_proba_downgrade_next_12m'], name='average_estimated_proba_downgrade')
s_data_test_downgrade_mean_proba.loc['Coefficient of discrimination'] = s_data_test_downgrade_mean_proba.loc[1] - s_data_test_downgrade_mean_proba.loc[0]
df_data_test_downgrade_mean_proba = pd.DataFrame(s_data_test_downgrade_mean_proba)


# *** Upgrade: Evaluation of the model (real data) ***
# *** Classification model performance ***
