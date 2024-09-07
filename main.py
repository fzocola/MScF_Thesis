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

# Options
pd.set_option('future.no_silent_downcasting', True)

# Project directories paths
paths = {'main': Path.cwd()}
paths.update({'data': Path.joinpath(paths.get('main'), 'data')})
paths.update({'scripts': Path.joinpath(paths.get('main'), 'scripts')})

# Warnings management




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
    s_ref_data_copy = s_ref_data
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
    df_copy = df
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

'''
df_test_2 = dic_issuer_fundamental_quarterly['LT_DEBT']
df_test_3 = dic_issuer_fundamental_quarterly['LT_DEBT'].copy()
df_test_3.index = df_test_3.index + pd.offsets.MonthEnd(0)
'''

# Convert quarterly data to monthly data
def df_conv_quarterly_to_monthly(df):
    df_copy = df

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

'''
df_test_q = dic_issuer_fundamental_quarterly['LT_DEBT']
df_test_d = dic_issuer_fundamental_daily['LT_DEBT']
df_test_m = dic_issuer_fundamental_monthly['LT_DEBT']
'''

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
# TODO: check functions output - add dic copy

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
    df_copy = df
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

'''
# Check the values in the rating dataframe
s_issuer_rating_daily_unique = pd.unique(df_issuer_rating_daily.values.ravel())
'''

# Define a mapping from BBG ratings to numeric values
def rating_to_numeric(df):
    df_copy = df
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

    # Apply the mapping to each column of the dataframe
    df_numeric = df_copy.apply(lambda col: col.map(sp_rating_to_numeric))

    return df_numeric

df_issuer_rating_numeric_daily = rating_to_numeric(df_issuer_rating_daily)


# Creation of the downgrade, upgrade dataframe
def create_rating_change_df(df, credit_event_type):
    df_copy = df
    # Create a new DataFrame with the same index and columns, filled with nan
    df_credit_change = pd.DataFrame(np.nan,
                                index=df_copy.index,
                                columns=df_copy.columns)

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
            df_credit_change.iat[i, df_copy.columns.get_loc(issuer)] = 1 if rating_change_detected else 0

    return df_credit_change

df_issuer_rating_downgrade_daily = create_rating_change_df(df_issuer_rating_numeric_daily, credit_event_type='downgrade')
df_issuer_rating_upgrade_daily = create_rating_change_df(df_issuer_rating_numeric_daily, credit_event_type='upgrade')

# *** Convert daily data to monthly data ***

'''
def dic_conv_daily_to_monthly(dic):
    dic_copy = copy.deepcopy(dic)
    for i in dic_copy:
        dic_copy[i] = dic_copy[i].resample('ME').ffill()
    return dic_copy
'''''

df_issuer_rating_downgrade_monthly = df_issuer_rating_downgrade_daily.resample('ME').ffill()
df_issuer_rating_upgrade_monthly = df_issuer_rating_upgrade_daily.resample('ME').ffill()



# %%
# **********************************************************
# *** Section: Distance to Default                       ***
# **********************************************************


# *** Import data for DD computation ***

df_lt_debt_monthly = dic_issuer_fundamental_monthly['LT_DEBT']
df_st_debt_monthly = dic_issuer_fundamental_monthly['ST_DEBT']
# Computing debt level according to KMV assumption for DD computation
df_kmv_debt_monthly = df_st_debt_monthly + 0.5*df_lt_debt_monthly

df_mkt_cap_daily = dic_issuer_market_daily['MKT_CAP']
# Resample by month and take the last available value within the month
df_mkt_cap_monthly = df_mkt_cap_daily.resample('ME').ffill()

# Compute the volatility (annualised) of the equity total return using a rolling windows of one year,
# nan if less than 100 values available
# Get the company share price
df_share_price_daily = dic_issuer_market_daily['SHARE_PRICE']
# Get the equity simple total return (not log return)
df_equity_tot_return_daily = dic_issuer_market_daily['TOT_RETURN']
df_equity_return_vol_daily = df_equity_tot_return_daily.rolling(window=252, min_periods=100).std() * np.sqrt(252)
# Resample by month and take the last available value within the month
df_equity_return_vol_monthly = df_equity_return_vol_daily.resample('ME').ffill()
# TODO: Compute log return
# TODO: Estimate GARCH ?

# 3m US treasury bill rate (annualised)
s_3m_us_treasury_bill_rate_daily = dic_market_data_daily['RATES']['GB3 Govt'] / 100
# TODO: change 3m rate with long term average ?
'''
# Calculate the average of the 3m US treasury bill rate series
treasury_bill_rate_daily_average = s_3m_us_treasury_bill_rate_daily.mean()
# Replace all the values in the series with the series average
s_3m_us_treasury_bill_rate_daily = s_3m_us_treasury_bill_rate_daily.apply(lambda x: treasury_bill_rate_daily_average)
'''
# Resample by month and take the last available value within the month
s_3m_us_treasury_bill_rate_monthly = s_3m_us_treasury_bill_rate_daily.resample('ME').ffill()


# *** Select data between a date range ***

# Define the date range
start_date = '2010-12-31'
end_date = '2023-12-31'

df_kmv_debt_monthly = df_kmv_debt_monthly.loc[start_date:end_date]
df_mkt_cap_monthly = df_mkt_cap_monthly.loc[start_date:end_date]
df_equity_return_vol_monthly = df_equity_return_vol_monthly.loc[start_date:end_date]
s_3m_us_treasury_bill_rate_monthly = s_3m_us_treasury_bill_rate_monthly.loc[start_date:end_date]


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
            V = solution.x[0]
            sV = solution.x[1]

            df_V_output.loc[i, c] = V
            df_sV_output.loc[i, c] = sV

    return df_V_output, df_sV_output


df_ev_monthly, df_ev_vol_monthly = get_df_V_sV(df_E=df_mkt_cap_monthly,
                                               df_K=df_kmv_debt_monthly,
                                               df_sE=df_equity_return_vol_monthly,
                                               s_r=s_3m_us_treasury_bill_rate_monthly, g=0, T=1, t=0)

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
                      s_r=s_3m_us_treasury_bill_rate_monthly, g=0, T=1, t=0)


# %%
# **********************************************************
# *** Section: Variables definition                      ***
# **********************************************************


# *** Firm specific variables ***

# Z-score variables
df_wc_to_ta_monthly = dic_issuer_fundamental_monthly['WORK_CAP'] / dic_issuer_fundamental_monthly['TOT_ASSET']
df_re_to_ta_monthly = dic_issuer_fundamental_monthly['RET_EARN'] / dic_issuer_fundamental_monthly['TOT_ASSET']
df_ebit_to_ta_monthly = dic_issuer_fundamental_monthly['EBIT'] / dic_issuer_fundamental_monthly['TOT_ASSET']
df_me_to_be_monthly = df_mkt_cap_monthly / dic_issuer_fundamental_monthly['BOOK_EQUITY']
df_sales_to_ta_monthly = dic_issuer_fundamental_monthly['SALES'] / dic_issuer_fundamental_monthly['TOT_ASSET']

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

# Independent variables - Firm specific variables
dic_ind_variables_firm_monthly['DD'] = df_DD_monthly

dic_ind_variables_firm_monthly['wc_to_ta'] = df_wc_to_ta_monthly
dic_ind_variables_firm_monthly['re_to_ta'] = df_re_to_ta_monthly
dic_ind_variables_firm_monthly['ebit_to_ta'] = df_ebit_to_ta_monthly
dic_ind_variables_firm_monthly['me_to_be'] = df_me_to_be_monthly
dic_ind_variables_firm_monthly['sales_to_ta'] = df_sales_to_ta_monthly

dic_ind_variables_firm_monthly['coverage'] = df_coverage_monthly
dic_ind_variables_firm_monthly['net_leverage'] = df_net_leverage_monthly

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






# %%
# **********************************************************
# *** Section: Summary Statics                           ***
# **********************************************************





