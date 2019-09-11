from pandas_datareader import data as pdr

from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
from data.read_ext.series_data import *
from data.read_ext.banxico_data import *
from data.read_ext.inegi_data import *
import pandas as pd
import numpy as np

# INEGI: https://www.inegi.org.mx/servicios/api_biinegi.html
# BANXICO: https://www.banxico.org.mx/SieAPIRest/service/v1/doc/catalogoSeries

na_values = ['.', 'N/E', 'null', 'None', '#N/A N/A']
start_dt = datetime(1900, 1, 1)
end_dt = datetime.now() - timedelta(1)


def get_series(info_dict):

    # Should store data somewhere to avoid reconsulting
    dfs = []
    for series, values in info_dict.items():
        if values['source'] == 'YAHOO':
            df = get_data_yahoo(series, start_dt, end_dt, values['yahoo_ticker'])
        elif values['source'] == 'FRED':
            df = get_data_fred(series, start_dt, end_dt, values['freq_adj'])
        elif values['source'] == 'BANXICO':
            df = get_data_banxico(series, start_dt, end_dt, values['bmx_serie'], values['freq_adj'])
        elif values['source'] == 'INEGI':
            df = get_data_inegi(series, values['series'])

        df = df.fillna(method='ffill')

        dfs.append(df)

    return dfs


def join_series(dfs):
    join = dfs[0]

    i = 0
    for df in dfs:
        if i == 0:
            i += 1
            continue
        
        join = join.merge(df, left_index=True, right_index=True, how='outer')

        i += 1

    return join

# Tried ETHUSD with: ADDRESS_CT, BLOCK_DIFF, BTCUSD, ETHUSD, FEES, GAS, HASHRATE, MARKETCAP, NOTRANS, SUPPLY; probably there is still not enough hist
# data is in etherscan and yahoo

def build_usdmxn(info_dict):
    # Get dataframes with data of each series
    dfs = get_series(info_dict)

    # Get min_date for data frames
    min_date = datetime(1995, 12, 29)

    # Join data frames into one data frame
    join = join_series(dfs)
    join = join.interpolate("linear")
    join = join.loc[min_date:]

    join['inpc_norm'] = join['INPC'] / join['INPC'].loc[min_date]
    join['cpi_norm'] = join['CPIAUCNS'] / join['CPIAUCNS'].loc[min_date]
    join['usdmxn_inc'] = join['USDMXN'] / join['USDMXN'].loc[min_date]

    join['inf_inc'] = join['inpc_norm'] / join['cpi_norm']

    join['DIF'] = join['inf_inc'] / join['usdmxn_inc']

    join = join.drop(['inpc_norm', 'cpi_norm', 'usdmxn_inc', 'inf_inc'], axis=1)

    # drop missing values
    join = join.dropna(how='any', axis=0)

    info_dict['DIF'] = {} 

    return join


def build_std(info_dict):

    # Get dataframes with data of each series
    dfs = get_series(info_dict)

    # Get min_date for data frames
    min_date = max([df.index.min() for df in dfs])

    # Join data frames into one data frame
    join = join_series(dfs)
    join = join.interpolate("linear")
    join = join.loc[min_date:]

    # drop missing values
    join = join.dropna(how='any', axis=0)

    return join


def move_by_freq(df, freq):
    # freq could also be calculated from series - leave that for later
    if freq == 'M':
        df.index = df.index + MonthEnd(1)
    elif freq == 'Q':
        df.index = df.index + QuarterEnd(1)

    return df


def get_data_yahoo(name, start, end, yahoo_ticker):
    logging.info("Downloading data from YAHOO for " + str(name))
    df = pdr.get_data_yahoo(yahoo_ticker, start, end)
    df.index.names = ['DATE']
    df = df.filter(['Adj Close'])
    df = df.rename(columns={'Adj Close': name})
    return df


def get_data_fred(name, start, end, freq):
    logging.info("Downloading data from FRED for " + str(name))

    FRED_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv?'
    var_ws = FRED_URL + 'cosd=' + start.strftime('%Y-%m-%d') + '&coed=' + end.strftime('%Y-%m-%d') + '&id=' + name
    df = pd.read_csv(var_ws, index_col='DATE', parse_dates=True, na_values=na_values)

    df = move_by_freq(df, freq)

    return df


def get_data_stockpup(name, filename, colname):
    """

    :param name: Name that will be assigned to desired column
    :param filename: Name of the file to be extracted from web page
    :param colname: Original column name in file
    :return:
    """

    STOCKPUP_URL = 'http://www.stockpup.com/data/'
    var_ws = STOCKPUP_URL + filename

    df = pd.read_csv(var_ws, index_col='Quarter end', parse_dates=True, na_values=na_values)

    df = df.rename(columns={colname: name})

    return df


def get_data_banxico(name, start, end, bmx_series, freq):
    logging.info("Downloading data from BANXICO for " + str(name))

    df = dwld_bmx([bmx_series], start, end)[0]

    df = df.rename(columns={bmx_series: name})

    df = move_by_freq(df, freq)

    return df


def get_data_inegi(name, series):
    logging.info("Downloading data from INEGI for " + str(name))

    df = dwld_inegi(series)
    df = df.rename(columns={series:name})

    return df


def get_data(target):
    info_dict = data_predict[target]
    if target == 'USDMXN':
        return build_usdmxn(info_dict)
    else:
        return build_std(info_dict)


def create_ts(data, target, pred_interval, n_steps, ordered=False):

    scalers = {}
    info_dict = data_predict[target]

    # For each column in the data set:
    # 1. Scale the data
    # 2. Form as many as possible examples from the time series, this means create many time series. Each of these
    # time series is formed from the original by spacing by the prediction interval - starting at a shift of plus i % pred_interval.
    # 3. Create dataset
    for col in data:

        # Apply log to the series, since an error (for MSE) of 10% should be the same @ $1 than @ $100
        # So without log mse for 10% @1 = (1.01 - 1)^2 = 1E-4 != (101 - 100)^2 = 1
        # But for log mse for 10% @1 = (log(1.01) - log(1))^2 = 9.9E-5 = (log(101) - log(100))^2

        if 'transform' in info_dict[col]:
            trans_func = info_dict[col]['transform']
            data[col] = trans_func(data[col])


        # Next scale from 0 to 1
        scalers[col] = MinMaxScaler(feature_range=(0, 1))
        data.loc[:, col] = scalers[col].fit_transform(data.loc[:, col].values.reshape(-1, 1))

    values = data.values

    # Create lags
    no_ex = values.shape[0] - n_steps * pred_interval
    start = 0
    end = no_ex
    lags = [values[start:end]]
    for i in range(0, n_steps):
        start += pred_interval
        end += pred_interval
        lags.append(values[start:end])

    # stack lags in middle axis, so that final shape is examples, steps/ lags, columns
    res = np.stack(lags, axis=1)

    X_score = res[-1:, 1:, :]

    if ordered:
        X_data = res[:, :-1, :]
        y_data = res[:, 1:, 0:1]
    else:
        shuffle_index = np.random.permutation(res.shape[0])
        X_data = res[shuffle_index, :-1, :]
        y_data = res[shuffle_index, 1:, 0:1]

    return res, X_data, y_data, X_score, scalers

def split_train_test(data, test_pc = 0.2):
    no_rows = data.shape[0]
    split = round(no_rows * (1 - test_pc))
    train = data.iloc[:split, :]
    test = data.iloc[split:, :]

    return train, test


def inverse_scale(arr, scalers, col_predict):
    arr = scalers[col_predict].inverse_transform(arr.reshape(-1, 1))
    return np.exp(arr)

