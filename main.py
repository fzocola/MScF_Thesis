# Import packages
from pathlib import Path
import pandas as pd

# Project directories paths
paths = {'main': Path.cwd()}
paths.update({'data': Path.joinpath(paths.get('main'), 'data')})
paths.update({'scripts': Path.joinpath(paths.get('main'), 'scripts')})



# %%
# **********************************************************
# *** Section: DATA MANAGEMENT                           ***
# **********************************************************

dic_fm_data = pd.read_excel(Path.joinpath(paths.get('data'), 'Thesis_Data_Importable.xlsx'), sheet_name=None)


for name, df in dic_fm_data.items():
    print(name)

ls_fundamental_sheet_name = ['LT_DEBT', 'ST_DEBT', 'CFO', 'INTEREST_EXP', 'EBITDA', 'NET_DEBT', 'TOT_ASSET',
                             'WORK_CAP','RET_EARN', 'EBIT', 'SALES', 'CUR_ASSET', 'CUR_LIAB', 'BOOK_EQUITY',
                             'CASH', 'NET_INCOME']

dic_issuer_fundamental_quarterly =

ls_issuer_market_sheet_name = ['RATING', 'MKT_CAP', 'TOT_RETURN', 'SHARE_PRICE', 'CDS_SPREAD_5Y']

dic_issuer_market_daily =

ls_market_sheet_name = ['RATES', 'SPX']

dic_market_data_daily =

ls_issuer_description_sheet_name = ['ISSUER', 'CDS_TICKER']

dic_issuer_description =

(ISSUER,LT_DEBT,ST_DEBT, CFO, INTEREST_EXP, EBITDA, NET_DEBT, TOT_ASSET, WORK_CAP, RET_EARN, EBIT, SALES, CUR_ASSET,
 CUR_LIAB, BOOK_EQUITY, CASH, NET_INCOME)
RATING
MKT_CAP
TOT_RETURN
SHARE_PRICE
CDS_TICKER
CDS_SPREAD_5Y
RATES
SPX
