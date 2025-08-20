import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from pathlib import Path
import os
from dotenv import load_dotenv

def fetch_tsla_data(period="1y", interval="1d"):
    try:
        tsla = yf.Ticker("TSLA")
        df = tsla.history(period=period, interval=interval)
        df.reset_index(inplace=True)
        df['Ticker'] = 'TSLA'
        return df
    except Exception as e:
        print(f"Error fetching TSLA data: {e}")
        return generate_sample_tsla_data()

def generate_sample_tsla_data(days=100, start_price=150):

    # Generate sample TSLA data using random walk pattern.
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    prices = [start_price]
    
    # Random walk pattern
    for i in range(1, days):
        change = np.random.normal(0, 2)
        new_price = prices[-1] + change
        prices.append(max(10, new_price))
    
    # DataFrame structure pattern
    df = pd.DataFrame({
        'Date': dates,
        'Ticker': 'TSLA',
        'Open': [p * 0.99 for p in prices],
        'High': [p * 1.02 for p in prices],
        'Low': [p * 0.98 for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 50000000, size=days)
    })
    
    return df

def calculate_technical_indicators(df, price_column='Close'):
    
    # Calculate technical indicators
    df_tech = df.copy()
    
    # Moving average pattern
    df_tech['SMA_20'] = df_tech[price_column].rolling(window=20).mean()
    df_tech['SMA_50'] = df_tech[price_column].rolling(window=50).mean()
    
    # Volatility pattern
    df_tech['Daily_Return'] = df_tech[price_column].pct_change()
    df_tech['Volatility_10d'] = df_tech['Daily_Return'].rolling(window=10).std()
    
    # Price change pattern
    df_tech['Price_Change'] = df_tech[price_column].diff()
    
    return df_tech

def create_target_variable(df, price_column='Close', days_forward=1):
    
    # Create target variable
    df_target = df.copy()
    df_target['Future_Price'] = df_target[price_column].shift(-days_forward)
    df_target['Price_Change_Future'] = df_target['Future_Price'] - df_target[price_column]
    df_target['Target'] = (df_target['Price_Change_Future'] > 0).astype(int)
    
    return df_target

def get_data_paths():
    # Get data paths from environment variables using your pattern.
    load_dotenv()
    raw_dir = Path(os.getenv('DATA_DIR_RAW', 'data/raw'))
    processed_dir = Path(os.getenv('DATA_DIR_PROCESSED', 'data/processed'))
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, processed_dir

def save_dataframe(df, filename, directory='processed'):
    
    # Save DataFrame
    raw_dir, processed_dir = get_data_paths()
    target_dir = processed_dir if directory == 'processed' else raw_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = target_dir / filename
    df.to_csv(filepath, index=False)
    return filepath