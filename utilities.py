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




from itertools import product
from visualisation import plot_acf_pacf_side_by_side
from preprocessing import encode_labels, train_test_split
import polars as pl

class TimeSeries:
    def __init__(self, ticker: str):
        self.data = get_ohlc_data([ticker], '2017-01-01')[ticker].reset_index()

    def construct_returns(self):
        self.data['Returns'] = self.data['Close'].pct_change()
        return self
    
    def profitability (self, column, threshold):
        self.data['Profitable'] = (self.data[column] > threshold).astype(int)
        return self
    
    def construct_technical_indicators(self, indicators: list, windows: list):
        for func, window in product(indicators, windows):
            series = func(self.data, window)
            self.data[series.name] = series
        return self
    
    def lag_column(self, column, skip_lags,n_lags):
        self.data = pl.DataFrame(self.data)
        self.data = self.data.with_columns([pl.col(column).shift(lag).alias(f"{column}_lag{lag}") for lag in range(skip_lags, n_lags + 1)]).to_pandas()
        return self

    def acf(self, column, max_lags):
        plot_acf_pacf_side_by_side(self.data[column], lags=max_lags).show()

    def clean_data_and_prepare_for_training(self, target: str):
        self.data = self.data.dropna()
        self.data = encode_labels(self.data, [column for column in self.data.columns if column.startswith('Bollinger')])
        self.data = self.data.drop(columns=['Date', 'Open', 'High', 'Low', 'Volume', 'Adj Close'])
        self.training_data = train_test_split(data=self.data,test_size=0.2,y_name=target)
        return self
    
    def add_index(self):
        self.data = self.data.reset_index()
        return self
    