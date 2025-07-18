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

'''
# Plot repartition of weights in EF for each expected return level
sns.set(context='paper', style='ticks', font_scale=1.0)
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
ax.stackplot(df_sample_EF_plugin['mean'], abs(df_sample_EF_plugin_w))
ax.set_title('Efficient Frontier Weights', size=28)
ax.tick_params(axis='both', labelsize=18)
ax.set_xlabel('Annualised average return', size=20)
ax.set_ylabel('Cumulative of weights absolute value', size=20)
fig.tight_layout()
plt.show()
fig.savefig(Path.joinpath(paths.get('output'), 'Project1', 'P1_Q1.2_sample_EF_weights.png'))
plt.close()
'''

'''
# Plot distribution of carbon intensity
sns.set(context='paper', style='ticks', font_scale=1.5)
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
ax.set_title('Distribution of Carbon Intensity (Y=2021)', size=28)
sns.histplot(df_carb_int_1m.iloc[-1], bins=25, stat='density', kde=False, color='royalblue', edgecolor='white', alpha=0.8)
sns.kdeplot(df_carb_int_1m.iloc[-1], color='red', lw=3)
ax.tick_params(axis='both', labelsize=18)
ax.set_xlim(left=0)
ax.set_xlabel('Carbon Intensity', size=20)
ax.set_ylabel('Density', size=20)
ax.grid(axis='y', alpha=0.4)
fig.tight_layout()
plt.show()
fig.savefig(Path.joinpath(paths.get('output'), 'Project1', 'P1_Q2.1_dist_carbon_intensity.png'))
plt.close()
'''

'''
# Plot DGT test
sns.set(context='paper', style='ticks', font_scale=2.0)
fig, ax = plt.subplots(3, sharex=True, figsize=(12, 15), dpi=300)
i = 0
for F_N_i in dic_F_N:
    ax[i].plot(dic_F_N[F_N_i], '8', color='blue')
    ax[i].plot(df_average_HO, color='black')
    ax[i].plot(df_quantile_HO_5_plus, color='red', label='Critical value @5%')
    ax[i].plot(df_quantile_HO_5_minus, color='red')
    ax[i].plot(df_quantile_HO_1_plus, color='g', label='Critical value @1%')
    ax[i].plot(df_quantile_HO_1_minus, color='g')
    ax[i].legend(loc='best', frameon=False, fontsize=12, ncol=2)
    ax[i].set_title('{} distribution (OS, Long-Only, CF75)'.format(F_N_i), size=20)
    i += 1
fig.tight_layout()
plt.show()
fig.savefig(Path.joinpath(paths.get('output'), 'Project2', 'P2_Q2.1.3_DGT_test.png'.format(i)))
plt.close()

'''

'''
# Plot VaR and ES Skewed t (OS, long-only, CF75)
sns.set(context='paper', style='ticks', font_scale=1.0)
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
ax.set_title('VaR and ES - GARCH - Skewed t (OS, Long-Only, CF75)', size=28)
ax.plot(df_VaR_ES_skewed_t['ES_skewed_t'], label='ES @{}%'.format(round(theta_cond * 100)), color='red', lw=3)
ax.plot(df_VaR_ES_skewed_t['VaR_skewed_t'], label='VaR @{}%'.format(round(theta_cond * 100)), color='black', lw=3)
ax.tick_params(axis='both', labelsize=18)
ax.legend(loc='upper left', fontsize=16)
fig.tight_layout()
plt.show()
fig.savefig(Path.joinpath(paths.get('output'), 'Project2', 'P2_Q2.1.4_VaR_ES_skewed_t.png'))
plt.close()
'''

'''
# Plot distribution of carbon intensity
sns.set(context='paper', style='ticks', font_scale=1.5)
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
ax.set_title('Distribution of Carbon Intensity (Y=2021)', size=28)
sns.histplot(df_carb_int_1m.iloc[-1], bins=25, stat='density', kde=False, color='royalblue', edgecolor='white', alpha=0.8)
sns.kdeplot(df_carb_int_1m.iloc[-1], color='red', lw=3)
ax.tick_params(axis='both', labelsize=18)
ax.set_xlim(left=0)
ax.set_xlabel('Carbon Intensity', size=20)
ax.set_ylabel('Density', size=20)
ax.grid(axis='y', alpha=0.4)
fig.tight_layout()
plt.show()
fig.savefig(Path.joinpath(paths.get('output'), 'Project1', 'P1_Q2.1_dist_carbon_intensity.png'))
plt.close()
'''

'''
df_data['next_12m_downgrade'].value_counts()
df_data['next_12m_upgrade'].value_counts()

df_data.drop(['DATES','Issuer'], axis=1, inplace=True)
zzz = df_data.groupby('next_12m_downgrade').mean()
zzzz = df_data.groupby('next_12m_upgrade').mean()
'''

'''

zzzz = dic_variables_monthly['ind_var_firm']['DD']
zzz = dic_variables_monthly['ind_var_firm_trend']['DD_trend']
www = df_issuer_rating_numeric_daily

xxx = dic_variables_monthly['dep_var_firm']['next_12m_downgrade']
'''

'''
# *** Downgrade ***

df_data_downgrade = df_data.drop(['DATES', 'Issuer', 'next_12m_upgrade'], axis=1)
df_data_downgrade_X = df_data_downgrade.loc[:, df_data_downgrade.columns != 'next_12m_downgrade']
df_data_downgrade_y = df_data_downgrade['next_12m_downgrade']

# Oversampling the data using SMOTE method
sm = SMOTE(random_state=42)
df_data_downgrade_X_res, df_data_downgrade_y_res = sm.fit_resample(df_data_downgrade_X, df_data_downgrade_y)

# Check data after resampling
print('Downgrade - Check data after resampling:')
print('Number of observations after resampling: ',len(df_data_downgrade_X_res))
print('Number of no downgrade after resampling:',len(df_data_downgrade_y_res[df_data_downgrade_y_res == 0]))
print('Number of downgrade after resampling:',len(df_data_downgrade_y_res[df_data_downgrade_y_res == 1]))
print('Proportion of no downgrade in oversampled data is ',len(df_data_downgrade_y_res[df_data_downgrade_y_res == 0])/len(df_data_downgrade_X_res))
print('Proportion of downgrade in oversampled data is ',len(df_data_downgrade_y_res[df_data_downgrade_y_res == 1])/len(df_data_downgrade_X_res))

df_data_downgrade_res = df_data_downgrade_X_res.join(df_data_downgrade_y_res)
# Reorder the columns to put the y column first
cols = [df_data_downgrade_y_res.name] + [col for col in df_data_downgrade_res.columns if col != df_data_downgrade_y_res.name]
df_data_downgrade_res = df_data_downgrade_res[cols]

# Scatter plot df_data_downgrade_res before and after resample
sns.set(context='paper', style='ticks', palette='bright', font_scale=1.0)
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
#ax.set_title('', size=28)
ax.scatter(x=df_data_downgrade[df_data_downgrade['next_12m_downgrade'] == 0]['EBIT/TA_trend'],
           y=df_data_downgrade[df_data_downgrade['next_12m_downgrade'] == 0]['DD_trend'],
           label = 'No Downgrade')
ax.scatter(x=df_data_downgrade[df_data_downgrade['next_12m_downgrade'] == 1]['EBIT/TA_trend'],
           y=df_data_downgrade[df_data_downgrade['next_12m_downgrade'] == 1]['DD_trend'],
           label = 'Downgrade')
ax.tick_params(axis='both', labelsize=18)
ax.set_xlabel('EBIT/TA_trend', size=20)
ax.set_ylabel('DD_trend', size=20)
#ax.grid(axis='y', alpha=0.4)
plt.legend()
fig.tight_layout()
plt.show()
# fig.savefig(Path.joinpath(paths.get('output'), 'XXXXXXXXX'.png'))
plt.close()

sns.set(context='paper', style='ticks', palette='bright', font_scale=1.0)
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
#ax.set_title('', size=28)
ax.scatter(x=df_data_downgrade_res[df_data_downgrade_res['next_12m_downgrade'] == 0]['EBIT/TA_trend'],
           y=df_data_downgrade_res[df_data_downgrade_res['next_12m_downgrade'] == 0]['DD_trend'],
           label = 'No Downgrade')
ax.scatter(x=df_data_downgrade_res[df_data_downgrade_res['next_12m_downgrade'] == 1]['EBIT/TA_trend'],
           y=df_data_downgrade_res[df_data_downgrade_res['next_12m_downgrade'] == 1]['DD_trend'],
           label = 'Downgrade')
ax.tick_params(axis='both', labelsize=18)
ax.set_xlabel('EBIT/TA_trend', size=20)
ax.set_ylabel('DD_trend', size=20)
#ax.grid(axis='y', alpha=0.4)
plt.legend()
fig.tight_layout()
plt.show()
# fig.savefig(Path.joinpath(paths.get('output'), 'XXXXXXXXX'.png'))
plt.close()
'''

'''
# *** Recursive Feature Elimination ***
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Downgrade
df_data_downgrade_res_X = df_data_downgrade_res.loc[:, df_data_downgrade_res.columns != 'next_12m_downgrade']
df_data_downgrade_res_y = df_data_downgrade_res['next_12m_downgrade']


estimator = LogisticRegression()
selector = RFE(estimator, n_features_to_select=10)
selector = selector.fit(df_data_downgrade_res_X, df_data_downgrade_res_y)
print(selector.support_)
print(selector.ranking_)

# Upgrade
'''

'''
df_returns = df_equity_tot_return_daily

# Create a new DataFrame with the same index and columns, filled with nan
df_garch_conditional_volatility_t_1_daily = pd.DataFrame(np.nan, index=df_returns.index, columns=df_returns.columns)

for issuer in df_returns:

    returns = df_returns[issuer]

    if returns.count 

    # Rescaling the returns
    returns = 100 * returns

    # Fit the AR(1)-GARCH(1,1) model
    am = arch_model(y=returns, mean='AR', lags=1, vol='GARCH', p=1, o=0, q=1, dist='normal')
    res = am.fit()
    # res.summary()

    # Get the 1 period ahead conditional volatility (annualised)
    conditional_volatility_t_1_daily = res.conditional_volatility.shift(-1) / 100 * np.sqrt(252)

    df_garch_conditional_volatility_t_1_daily[issuer] = conditional_volatility_t_1_daily

# Resample by month and take the last available value within the month
df_garch_conditional_volatility_t_1_monthly = df_garch_conditional_volatility_t_1_daily.resample('ME').ffill()
'''

'''
returns = df_equity_tot_return_daily['AMGN US Equity']

# Rescaling the returns
returns = 100 * returns

# Fit the AR(1)-GARCH(1,1) model
am = arch_model(y=returns, mean='Constant', vol='GARCH', p=1, o=0, q=1, dist='normal')
res = am.fit()
res.summary()

# Get the 1 period ahead conditional volatility (annualised)
conditional_volatility_t_1_daily = res.conditional_volatility.shift(-1) / 100 * np.sqrt(252)
'''

'''
plt.plot(conditional_volatility_t_1_daily)
plt.plot(df_equity_return_vol_daily['AMGN US Equity'])
plt.show()
'''

'''
# Get the 1 period ahead forcast of the volatility
forecast_volatility = np.sqrt(res.forecast(horizon=1).variance) / 100
print(forecast_volatility)
'''

'''
df_test_2 = dic_issuer_fundamental_quarterly['LT_DEBT']
df_test_3 = dic_issuer_fundamental_quarterly['LT_DEBT'].copy()
df_test_3.index = df_test_3.index + pd.offsets.MonthEnd(0)
'''

'''
df_test_q = dic_issuer_fundamental_quarterly['LT_DEBT']
df_test_d = dic_issuer_fundamental_daily['LT_DEBT']
df_test_m = dic_issuer_fundamental_monthly['LT_DEBT']
'''

'''
# Check the values in the rating dataframe
s_issuer_rating_daily_unique = pd.unique(df_issuer_rating_daily.values.ravel())
'''

'''
def dic_conv_daily_to_monthly(dic):
    dic_copy = copy.deepcopy(dic)
    for i in dic_copy:
        dic_copy[i] = dic_copy[i].resample('ME').ffill()
    return dic_copy
'''''

'''
# Replace all values with NaN except where values are 0
df_issuer_rating_nb_months_next_change_monthly = df_issuer_rating_nb_months_last_change_monthly.where(df_issuer_rating_nb_months_last_change_monthly == 0, np.nan)

for issuer in tqdm(df_issuer_rating_nb_months_next_change_monthly.iloc[:, :].columns, desc='Creation nb of months until next change df'):

    count = 0
    start_counting = False
    col = df_issuer_rating_nb_months_next_change_monthly[issuer]
    # Work backwards through the column
    for i in reversed(range(len(col))):
        if col.iloc[i] == 0:
            start_counting = True

        if start_counting == True:
            if col.iloc[i] == 0:
                count = 0
            elif pd.isna(col.iloc[i]):
                count += 1
                col.iloc[i] = count
'''

'''
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

df_issuer_rating_nb_months_next_change_monthly = df_nb_months_until_next_rating_change(df_issuer_rating_nb_months_last_change_monthly)
'''


'''
##########################
# Test
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression().fit(df_data_downgrade_res_X, df_data_downgrade_res_y)
y_predict = clf.predict(sm.add_constant(df_data_downgrade.loc[:, df_data_downgrade.columns != 'next_12m_downgrade']))
##########################
'''


'''
df=df_data_downgrade_res
y=df_data_downgrade_res.columns[0]
variables_for_interaction=df_data_downgrade_res.columns[1:9].tolist()
binary_var=df_data_downgrade_res.columns[-1]

# Create interaction terms between each variable and the binary variable
interactions = [f'{var}:{binary_var}' for var in variables_for_interaction]

# Construct the formula dynamically
independent_vars = list(df.columns.drop([y]))  # Include all variables except y and binary_var

formula = y + ' ~ ' + ' + '.join(independent_vars + interactions)
'''

'''
for i in df_data_downgrade_res.columns[1:9].tolist():
    print(i)
'''

'''
import statsmodels.formula.api as smf
# Fit the logistic regression model with interaction terms
model = smf.logit(formula=formula_downgrade, data=df_data_downgrade_res).fit()
# Print the summary of the model
print(model.summary())

df_data_downgrade_res.drop(columns=['rating_cat_1.0'], inplace=True)
'''

'''
def specification_construction(df, y, variables_for_interaction, binary_var):
    # Create interaction terms between each variable and the binary variable
    interactions = [f'Q("{var}"):Q("{binary_var}")' for var in variables_for_interaction]

    # Construct the formula dynamically
    independent_vars = list(df.columns.drop([y]))
    independent_vars = [f'Q("{var}")' for var in independent_vars]

    formula = y + ' ~ ' + ' + '.join(independent_vars + interactions)

    return formula
    
formula_downgrade = specification_construction(df=df_data_downgrade_res,
                                               y=df_data_downgrade_res.columns[1],
                                               variables_for_interaction=df_data_downgrade_res.columns[1:9].tolist(),
                                               binary_var=df_data_downgrade_res.columns[-1])     

formula_upgrade = specification_construction(df=df_data_upgrade_res,
                                               y=df_data_upgrade_res.columns[1],
                                               variables_for_interaction=df_data_upgrade_res.columns[1:9].tolist(),
                                               binary_var=df_data_upgrade_res.columns[-1])

'''


'''
# Insert a constant
df_data_downgrade_res = sm.add_constant(df_data_downgrade_res)
df_data_upgrade_res = sm.add_constant(df_data_upgrade_res)
'''

'''
df_data_downgrade_res['rating_1.0']
df_dummies = pd.get_dummies(df_data_downgrade_res['rating_cat_id'], prefix='rating').drop(columns=['rating_2.0']).astype(int)
'''

'''
# Creation of dummy variables, with rating == 2 as the reference
df_data_downgrade_res = pd.get_dummies(df_data_downgrade_res, columns=['rating_cat_id'], prefix='rating', dtype=int).drop(columns=['rating_2.0'])
df_data_upgrade_res = pd.get_dummies(df_data_upgrade_res, columns=['rating_cat_id'], prefix='rating', dtype=int).drop(columns=['rating_2.0'])
'''

# TODO: Run ok until here

# TODO: Cluster Robust std ? (I technically don't care?, I want to measure to quality of predictions)
# TODO: Constant
# TODO: winzorization
# TODO: marginal effect
# TODO: matrice de confusion

'''
www = pd.Series(logit_res.params)
yyy = sm.add_constant(df_data_downgrade.loc[:, df_data_downgrade.columns != 'next_12m_downgrade'])
zzz = (sm.add_constant(df_data_downgrade.loc[:, df_data_downgrade.columns != 'next_12m_downgrade']) * df_downgrade_params)
'''

'''
# *** Take the max proba over the last 12m before a credit event ***
'''
'''
We evaluate the performance of the model based max estimated proba reach within the 12m before the credit 
event
'''
'''
def find_max_default_proba():
    return

# Find the maximum probability of default for each credit event
max_prob_df = df_data_test.groupby('downgrade_id')['estimated_proba_downgrade_next_12m'].transform('max')
# Assign this max probability value to all rows for the respective credit_event_id
df_data_test['max_estimated_proba_downgrade_12m_before_default'] = max_prob_df
'''


'''
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
cm = confusion_matrix(df_data_test_downgrade_y_true, df_data_test_downgrade_y_pred, labels=[1, 0], normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 0])
disp.plot()
plt.gcf().set_dpi(300)
plt.show()
#plt.savefig(Path.joinpath(paths.get('output'), 'XXXXXXXXX'.png'), dpi=300, bbox_inches='tight')
plt.close()

# Classification report (for a given threshold)
classification_report = classification_report(df_data_test_downgrade_y_true, df_data_test_downgrade_y_pred)
print(classification_report)

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
'''

'''
def relative_logloss(y_true, y_pred):
    base_rate = y_true.mean() # base rate
    ll_model = log_loss(y_true, y_pred)

    ll_base = log_loss(y_true, [base_rate] * len(y_true))
    rll = 1 - (ll_model / ll_base)
    return rll
'''

'''
def model_evaluation(df_data_test, credit_event_type, recall_target):

    if credit_event_type == 'downgrade':
        df_data_test_y_true = df_data_test['downgrade-dummy']
        df_data_test_estimated_proba = df_data_test['estimated-downgrade-hazard-rate']
    if credit_event_type == 'upgrade':
        df_data_test_y_true = df_data_test['upgrade-dummy']
        df_data_test_estimated_proba = df_data_test['estimated-upgrade-hazard-rate']

    # *** Tuning the decision threshold ***

    # ROC
    fpr, tpr, thresholds = roc_curve(df_data_test_y_true, df_data_test_estimated_proba)
    logit_roc_auc = roc_auc_score(df_data_test_y_true, df_data_test_estimated_proba)

    sns.set(context='paper', style='ticks', palette='bright', font_scale=1.0)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    ax.set_title('ROC Curve - {}'.format(credit_event_type), size=28)
    ax.plot(fpr, tpr, label='Logit Model (AUC = %0.2f)' % logit_roc_auc, linewidth=2)
    ax.plot([0, 1], [0, 1], 'r--', label='Random Classifier (AUC = 0.50)')
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
    precision, recall, thresholds = precision_recall_curve(df_data_test_y_true, df_data_test_estimated_proba)
    # remove the artificial end‑point (last element)
    precision = precision[:-1]
    recall = recall[:-1]
    logit_average_precision = average_precision_score(df_data_test_y_true, df_data_test_estimated_proba)
    positive_class_frequency = df_data_test_y_true.sum() / len(df_data_test_y_true)

    sns.set(context='paper', style='ticks', palette='bright', font_scale=1.0)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    ax.set_title('Precision-Sensitivity Curve', size=28)
    ax.plot(recall, precision, label='Logit Model (AP = %0.2f)' % logit_average_precision, linewidth=2)
    ax.plot([0, 1], [positive_class_frequency, positive_class_frequency], 'r--', label='Random Classifier (AP = %0.2f)' % positive_class_frequency)
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

    df_precision_recall = pd.DataFrame({
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds
    })

    df_precision_recall['f1_score'] = 2 * (precision * recall) / (precision + recall)

    # Plot F1-score, Precision, and Recall
    sns.set(context='paper', style='ticks', palette='bright', font_scale=1.0)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    # ax.set_title('', size=28)
    ax.plot(df_precision_recall['thresholds'], df_precision_recall['f1_score'], label='F1-score', linewidth=2)
    ax.plot(df_precision_recall['thresholds'], df_precision_recall['precision'], label='Precision', linestyle='-', linewidth=2)
    ax.plot(df_precision_recall['thresholds'], df_precision_recall['recall'], label='Sensitivity (Recall)', linestyle='-', linewidth=2)

    # Add a horizontal line at y=recall_target
    ax.axhline(y=recall_target, linestyle='--', linewidth=1.5, label=f'Sensitivity (Recall) = {recall_target}')

    # Find the threshold where Recall crosses y=0.75
    recall = df_precision_recall['recall']
    precision = df_precision_recall['precision']
    thresholds = df_precision_recall['thresholds']

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
    # ax.set_ylabel('F1-score', size=20)
    ax.grid(True, alpha=0.4)
    plt.legend(fontsize=15)
    fig.tight_layout()
    plt.show()
    # fig.savefig(Path.joinpath(paths.get('output'), 'XXXXXXXXX'.png'))
    plt.close()


    # *** Other mesures of performance ***

    # Evolution of proba of downgrade between 36 months to 1 month to downgrade
    if credit_event_type == 'downgrade':
        df_data_test_stats_monthly = df_data_test.drop(['DATES', 'Issuer', 'time_until_next_upgrade', 'downgrade_id', 'upgrade_id'], axis=1).groupby('time_until_next_downgrade').mean()
        y = df_data_test_stats_monthly['estimated-downgrade-hazard-rate'][(df_data_test_stats_monthly.index >= 1) & (df_data_test_stats_monthly.index <= 36)]

    if credit_event_type == 'upgrade':
        df_data_test_stats_monthly = df_data_test.drop(['DATES', 'Issuer', 'time_until_next_downgrade', 'downgrade_id', 'upgrade_id'], axis=1).groupby('time_until_next_upgrade').mean()
        y = df_data_test_stats_monthly['estimated-upgrade-hazard-rate'][(df_data_test_stats_monthly.index >= 1) & (df_data_test_stats_monthly.index <= 36)]

    sns.set(context='paper', style='ticks', palette='bright', font_scale=1.0)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    # ax.set_title('', size=28)
    ax.plot(y, marker='o', linestyle='-', linewidth=2)
    ax.tick_params(axis='both', labelsize=18)
    ax.set_xticks(np.arange(0, len(y), 6))
    ax.set_xlabel('Nb of months until {}'.format(credit_event_type), size=20)
    ax.set_ylabel('Estimated {} hazard rate'.format(credit_event_type), size=20)
    ax.grid(True, alpha=0.4)
    # plt.legend()
    fig.tight_layout()
    plt.show()
    # fig.savefig(Path.joinpath(paths.get('output'), 'XXXXXXXXX'.png'))
    plt.close()


    # *** Class prediction (for a given threshold) ***

    threshold = threshold_at_y
    df_data_test['y_pred'] = [1 if p > threshold else 0 for p in df_data_test_estimated_proba]
    df_data_test_y_pred = df_data_test['y_pred']

    # Confusion Matrix (for a given threshold)
    cm = confusion_matrix(df_data_test_y_true, df_data_test_y_pred, labels=[1, 0], normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 0])
    disp.plot()
    plt.gcf().set_dpi(300)
    plt.show()
    # plt.savefig(Path.joinpath(paths.get('output'), 'XXXXXXXXX'.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Classification report (for a given threshold)
    classification_report_t = classification_report(df_data_test_y_true, df_data_test_y_pred)
    print(classification_report_t)

    # Sensitivity (Recall) (for a given threshold)
    recall_score_t = recall_score(df_data_test_y_true, df_data_test_y_pred, average='binary')
    # Precision (for a given threshold)
    precision_score_t = precision_score(df_data_test_y_true, df_data_test_y_pred, average='binary')
    # Accuracy (for a given threshold)
    accuracy_score_t = accuracy_score(df_data_test_y_true, df_data_test_y_pred)
    # Balance accuracy (for a given threshold)
    balanced_accuracy_t = balanced_accuracy_score(df_data_test_y_true, df_data_test_y_pred)

    # Coefficient of discriminations
    if credit_event_type == 'downgrade':
        df_data_test_mean = df_data_test.drop(['DATES', 'Issuer'], axis=1).groupby('downgrade-dummy').mean()
        s_data_test_mean_proba = pd.Series(df_data_test_mean['estimated-downgrade-hazard-rate'], name='average_estimated-downgrade-hazard-rate')
    if credit_event_type == 'upgrade':
        df_data_test_mean = df_data_test.drop(['DATES', 'Issuer'], axis=1).groupby('upgrade-dummy').mean()
        s_data_test_mean_proba = pd.Series(df_data_test_mean['estimated-upgrade-hazard-rate'], name='average_estimated-downgrade-hazard-rate')

    s_data_test_mean_proba.loc['Coefficient of discrimination'] = s_data_test_mean_proba.loc[1] - s_data_test_mean_proba.loc[0]
    df_data_test_mean_proba = pd.DataFrame(s_data_test_mean_proba)

    return classification_report_t, df_data_test_mean_proba


# *** Downgrade: Evaluation of the model (in sample) ***
# *** Classification model performance ***
classification_report_t_downgrade, df_data_test_mean_proba_downgrade = model_evaluation(df_data_test=df_data_downgrade_res, credit_event_type='downgrade', recall_target=0.75)

# *** Upgrade: Evaluation of the model (in sample) ***
# *** Classification model performance ***
classification_report_t_upgrade, df_data_test_mean_proba_upgrade = model_evaluation(df_data_test=df_data_upgrade_res, credit_event_type='upgrade', recall_target=0.5)
'''

