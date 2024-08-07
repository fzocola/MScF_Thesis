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


