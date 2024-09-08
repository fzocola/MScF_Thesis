# *** Scraps ***

'''
for name, df in dic_fm_data.items():
    print(name)
'''

'''
df_test = dic_market_data_daily['SPX']['px_last']
s_spx_price = dic_market_data_daily['SPX'].set_index('DATES')['px_last'].dropna()
s_market_open_date = pd.Series(s_spx_price.index)
'''

'''
df_test = dic_issuer_fundamental_quarterly['LT_DEBT']
na_to_nb_value = -9999999999999
df_test = dic_issuer_fundamental_quarterly['LT_DEBT'].fillna(na_to_nb_value)

# Create a date range that includes all daily dates within the range of the quarterly data
daily_index = pd.date_range(start=df_test.index.min(), end=df_test.index.max() + timedelta(days=60), freq='D')

# Reindex the DataFrame to the daily date range and forward-fill the values
df_daily = df_test.reindex(daily_index).ffill()
df_daily = df_daily.replace(na_to_nb_value, np.nan)
'''

'''
df_test_1 = dic_issuer_fundamental_quarterly['LT_DEBT']
df_test_2 = df_conv_quarterly_to_daily(dic_issuer_fundamental_quarterly['LT_DEBT'])
'''

'''
df_test_1 = dic_issuer_fundamental_quarterly['LT_DEBT']
df_filtered = df_filtered_date(df_test_1, s_market_open_date)
'''


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


'''
# Create a new DataFrame with the same index and columns, filled with zeros
df_issuer_rating_downgrade_daily = pd.DataFrame(np.nan,
                                             index=df_issuer_rating_numeric_daily.index,
                                             columns=df_issuer_rating_numeric_daily.columns)


# Set last_date
last_date = df_issuer_rating_numeric_daily.index[-1] - pd.DateOffset(years=1)


for issuer in tqdm(df_issuer_rating_numeric_daily.iloc[:, :1].columns, desc='Downgrade'):

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


'''
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
'''

'''
# Check the values in the rating dataframe
s_issuer_rating_daily_unique = pd.unique(df_issuer_rating_daily.values.ravel())
'''
'''
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
'''

'''
# Check the values in the rating dataframe
s_issuer_rating_daily_numeric_unique = pd.unique(df_issuer_rating_daily.values.ravel())
'''


'''
g = 0
T = 1
t = 0

df_enterprise_value_monthly = pd.DataFrame(-999999.999, index=df_kmv_debt_monthly.index, columns=df_kmv_debt_monthly.columns)
df_enterprise_value_vol_monthly = pd.DataFrame(-999999.999, index=df_kmv_debt_monthly.index, columns=df_kmv_debt_monthly.columns)

for c in tqdm(df_kmv_debt_monthly.iloc[:,:].columns, desc='Merton implied V and sV'):
    print(c)
    for i in df_kmv_debt_monthly.index[:]:
        print(i)
        K = df_kmv_debt_monthly.loc[i, c]
        E = df_mkt_cap_monthly.loc[i, c]
        sE = df_equity_return_vol_monthly.loc[i, c]
        r = df_3m_us_treasury_bill_rate_monthly.loc[i]
        print(K, E, sE, r)

        solution = get_merton_implied_V_sV(E=E, K=K, r=r, g=g, sE=sE, T=T, t=t)
        V = solution.x[0]
        sV = solution.x[1]
        #print(solution)
        print(V, sV)
        df_enterprise_value_monthly.loc[i, c] = V
        df_enterprise_value_vol_monthly.loc[i, c] = sV
'''

'''
df_copy = df_issuer_rating_numeric_daily


# Create a new DataFrame with the same index and columns, filled with nan
df_days_since_change = pd.DataFrame(np.nan,
                            index=df_copy.index,
                            columns=df_copy.columns)


for issuer in tqdm(df_copy.iloc[:, :].columns, desc='Creation df nb days since last change'):
    # Identify rating changes (True where the rating has changed)
    rating_change = df_copy[issuer] != df_copy[issuer].shift(1)

    # Count number of rating change
    count_rating_change = rating_change.cumsum()

    # For each group, subtract the first date of the group from all dates in that group
    time_since_last_change = df_copy[issuer].index.to_series().groupby(count_rating_change).transform(lambda x: x - x.iloc[0])

    # Convert to days
    time_since_last_change_in_days = time_since_last_change.dt.days

    df_copy[issuer] = time_since_last_change_in_days
'''

'''
#df_issuer_rating_nb_days_last_change_monthly = df_issuer_rating_nb_days_last_change_daily.resample('ME').ffill()


df_copy_daily = copy.deepcopy(df_issuer_rating_nb_days_last_change_daily)
df_copy_monthly = df_copy_daily.resample('ME').ffill()


# Create a new DataFrame with the same index and columns, filled with nan
df_output = pd.DataFrame(np.nan, index=df_copy_monthly.index, columns=df_copy_monthly.columns)


for issuer in tqdm(df_copy_monthly.iloc[:, :].columns, desc='Creation of nb of months since last rating change df'):
    # Find the last available date within each month
    s_last_available_dates = df_copy_daily[issuer].groupby(df_copy_daily[issuer].index.to_period('M')).apply(lambda x: x.index.max())
    # Set the index of the resampled data to the last available date within each month
    df_copy_monthly[issuer].index = s_last_available_dates
    # Find the rating change day
    s_rating_change_date = (df_copy_monthly[issuer].index - pd.to_timedelta(df_copy_monthly[issuer], unit='D')).resample('ME').ffill()

    for index_date, change_date in s_rating_change_date.items():
        # Calculate the nb of months since last rating change
        months_difference = relativedelta(index_date, change_date).months + (relativedelta(index_date, change_date).years * 12)
        print(months_difference)
        df_output[issuer][index_date] = months_difference

'''

'''
df_issuer_rating_nb_months_last_change_monthly
'''
'''
dic = dic_variables_monthly['ind_var_firm']
df_lag = df_issuer_rating_nb_months_last_change_monthly
dic_tmp = {}

for var in dic:
    df_tmp = pd.DataFrame(np.nan, index=dic[var].index, columns=dic[var].columns)

    for firm in tqdm(dic[var].iloc[:, :], desc="Creating df delta for {}".format(var)):

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

    dic_tmp['delta_{}'.format(var)] = df_tmp
'''

'''
# firm variables
ls_output = []
data_type = ['dep_var_firm', 'ind_var_firm', 'ind_var_firm_trend']
for i in data_type:
    dic_tpm = dic_variables_monthly[i]

    ls_reshaped_data = []
    for variable, df in dic_tpm.items():
        df.index.name = 'DATES'
        df_melt = pd.melt(df.reset_index(), id_vars=['DATES'], var_name='Issuer', value_vars=df.columns, value_name=variable)
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
    dic_tpm = dic_variables_monthly[i]

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
'''

