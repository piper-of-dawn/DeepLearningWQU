import pandas as pd
def calculate_bollinger_bands(data, window, num_std_dev=1.96):
    data = data.copy()
    column_name = f"BollingerBand_{window}"
    data['MA'] = data['Adj Close'].rolling(window=window).mean()
    data['UpperBand'] = data['MA'] + (num_std_dev * data['Adj Close'].rolling(window=window).std())
    data['LowerBand'] = data['MA'] - (num_std_dev * data['Adj Close'].rolling(window=window).std())
    data[column_name] = 'Inside'
    data.loc[data['Adj Close'] > data['UpperBand'], column_name] = 'Above'
    data.loc[data['Adj Close'] < data['LowerBand'], column_name] = 'Below' 
    data.drop(['MA', 'UpperBand', 'LowerBand'], axis=1, inplace=True)   
    return data[column_name]

def calculate_rsi(data, window):
    column_name = f"RSI_{window}"
    delta = data['Adj Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    data[column_name] = rsi
    return data[column_name]

def calculate_adx(data, window):
    column_name = f"ADX_{window}"
    # True Range (TR)
    data['High-Low'] = data['High'] - data['Low']
    data['High-PrevClose'] = abs(data['High'] - data['Close'].shift(1))
    data['Low-PrevClose'] = abs(data['Low'] - data['Close'].shift(1))
    data['TR'] = data[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)

    # Directional Movement (DM)
    data['UpMove'] = data['High'].diff()
    data['DownMove'] = -data['Low'].diff()

    # Positive Directional Movement (PDM) and Negative Directional Movement (NDM)
    data['PDM'] = data['UpMove'].where(data['UpMove'] > data['DownMove'], 0)
    data['NDM'] = data['DownMove'].where(data['DownMove'] > data['UpMove'], 0)

    # Smoothed True Range (ATR)
    data['ATR'] = data['TR'].rolling(window=window).mean()

    # Smoothed Positive Directional Movement (PDI) and Negative Directional Movement (NDI)
    data['PDI'] = (data['PDM'].rolling(window=window).mean() / data['ATR']) * 100
    data['NDI'] = (data['NDM'].rolling(window=window).mean() / data['ATR']) * 100

    # Directional Index (DI)
    data['DI+'] = data['PDI']
    data['DI-'] = data['NDI']

    # Calculate the Directional Index (DX)
    data['DX'] = (abs(data['DI+'] - data['DI-']) / (data['DI+'] + data['DI-'])) * 100

    # Average Directional Index (ADX)
    data[column_name] = data['DX'].rolling(window=window).mean()

    # Drop intermediate columns
    data.drop(['High-Low', 'High-PrevClose', 'Low-PrevClose', 'TR', 'UpMove', 'DownMove',
               'PDM', 'NDM', 'ATR', 'PDI', 'NDI', 'DI+', 'DI-', 'DX'], axis=1, inplace=True)
    return data[column_name]

# calculate_bollinger_bands(data_msft, window=50)
# calculate_bollinger_bands(data_msft, window=80)
# calculate_bollinger_bands(data_msft, window=130)