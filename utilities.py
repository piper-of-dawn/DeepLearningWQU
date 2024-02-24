import yfinance as yf
import pandas as pd

def get_ohlc_data(tickers, start_date='2020-01-01'):
    ohlc_data = {}
    for ticker in tickers:
        try:
            stock_data = yf.download(ticker, start=start_date)
            ohlc_data[ticker] = stock_data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    return ohlc_data

import numpy as np
import scipy.stats as stats
import pandas as pd

def compute_stats(series: pd.Series, name: str):
    stats_dict = {
        'Name': name,
        'Minimum': np.min(series),
        'Q1': series.quantile(0.25),
        'Median': np.median(series),
        'Q3': series.quantile(0.75),
        'Maximum': np.max(series),
        'Range': np.ptp(series),
        'Mean': np.mean(series),
        'Standard deviation': np.std(series),
        'Sum': np.sum(series),
        'Coefficient of variation': stats.variation(series),
        'Median Absolute Deviation': stats.median_abs_deviation(series),
        'Kurtosis': stats.kurtosis(series),
        'Skewness': stats.skew(series)
    }    
    return stats_dict


def getWeights(d,lags):
    # return the weights from the series expansion of the differencing operator
    # for real orders d and up to lags coefficients
    w=[1]
    for k in range(1,lags):
        w.append(-w[-1]*((d-k+1))/k)
    w=np.array(w).reshape(-1,1) 
    return w




def ts_differencing(series, order, lag_cutoff):
    # return the time series resulting from (fractional) differencing
    # for real orders order up to lag_cutoff coefficients
    
    weights=getWeights(order, lag_cutoff)
    res=0
    for k in range(lag_cutoff):
        res += weights[k]*series.shift(k).fillna(0)
    return res[lag_cutoff:] 
