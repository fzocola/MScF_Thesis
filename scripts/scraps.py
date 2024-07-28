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