import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def fill_missing_median(df, columns):
    df_filled = df.copy()
    for col in columns:
        if col in df_filled.columns and pd.api.types.is_numeric_dtype(df_filled[col]):
            df_filled[col] = df_filled[col].fillna(df_filled[col].median())
    return df_filled

def drop_missing(df, threshold=0.5):
    missing_ratio = df.isnull().sum() / len(df)
    columns_to_drop = missing_ratio[missing_ratio > threshold].index
    return df.drop(columns=columns_to_drop)

def normalize_data(df, columns):
    df_normalized = df.copy()
    scaler = StandardScaler()
    
    for col in columns:
        if col in df_normalized.columns and pd.api.types.is_numeric_dtype(df_normalized[col]):
            # Handle missing values temporarily for scaling
            non_missing = df_normalized[col].notna()
            if non_missing.any():
                df_normalized.loc[non_missing, col] = scaler.fit_transform(
                    df_normalized.loc[non_missing, [col]]
                )
    
    return df_normalized