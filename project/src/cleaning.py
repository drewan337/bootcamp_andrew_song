import pandas as pd
import numpy as np

def clean_stock_data(df):
    df_clean = df.copy()
    
    # Handling missing values
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    df_clean = df_clean.dropna()
    
    return df_clean

def calculate_technical_features(df, price_column='Close'):

    # Calculate technical features
    df_features = df.copy()
    
    # Moving average pattern
    df_features['SMA_10'] = df_features[price_column].rolling(window=10).mean()
    df_features['SMA_30'] = df_features[price_column].rolling(window=30).mean()
    
    # Volatility pattern
    df_features['Returns'] = df_features[price_column].pct_change()
    df_features['Volatility'] = df_features['Returns'].rolling(window=20).std()
    
    # Price change pattern
    df_features['Price_Change'] = df_features[price_column].diff()
    
    return df_features

def prepare_model_data(df, price_column='Close', target_days=1):
    
    # Prepare data for modeling using your target creation pattern.
    df_model = df.copy()
    
    # Creating Price Target
    df_model['Future_Price'] = df_model[price_column].shift(-target_days)
    df_model['Target'] = (df_model['Future_Price'] > df_model[price_column]).astype(int)
    
    # Remove rows with missing future prices
    df_model = df_model.dropna(subset=['Future_Price', 'Target'])
    
    return df_model