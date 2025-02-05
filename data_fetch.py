import ccxt
import pandas as pd
import pandas_ta as ta  # Import pandas-ta for technical indicators

def fetch_top_100_symbols():
    """Fetch the top 100 cryptocurrency symbols from Binance."""
    exchange = ccxt.binance()
    markets = exchange.load_markets()
    symbols = [symbol for symbol in markets.keys() if '/USDT' in symbol][:100]  # Get top 100 USDT pairs
    return symbols

def fetch_ohlcv(symbol, timeframe='1h', limit=500):
    """Fetch OHLCV data for a given symbol from Binance."""
    exchange = ccxt.binance({'enableRateLimit': True})
    data = exchange.fetch_ohlcv(symbol, timeframe, limit)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def add_indicators(df):
    """Add RSI, MACD, and Bollinger Bands to the dataframe."""
    df['rsi'] = ta.rsi(df['close'], length=14)  # RSI with a 14-period window
    macd = ta.macd(df['close'])  # MACD
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    bollinger = ta.bbands(df['close'], length=20)  # Bollinger Bands
    df['bb_upper'] = bollinger['BBU_20_2.0']
    df['bb_middle'] = bollinger['BBM_20_2.0']
    df['bb_lower'] = bollinger['BBL_20_2.0']
    
    # Drop rows with NaN values after indicator calculations
    df.dropna(inplace=True)
    return df

if __name__ == '__main__':
    symbols = fetch_top_100_symbols()
    all_data = []
    
    for symbol in symbols:
        print(f"Fetching data for {symbol}")
        df = fetch_ohlcv(symbol)
        df = add_indicators(df)
        df['symbol'] = symbol  # Add symbol column
        all_data.append(df)
    
    combined_df = pd.concat(all_data)
    combined_df.to_csv('data.csv', index=False)
    print("Data with indicators for top 100 cryptocurrencies saved to data.csv")