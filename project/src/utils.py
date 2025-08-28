import os
import pathlib
import datetime as dt
import pandas as pd
import numpy as np
from typing import Union, Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

def ts() -> str:
    """Generate timestamp string for consistent filenames."""
    return dt.datetime.now().strftime('%Y%m%d-%H%M%S')

def validate_data(df: pd.DataFrame, required_columns: list) -> Dict[str, Any]:
    """Validate DataFrame structure and data quality."""
    validation = {
        'missing_columns': [col for col in required_columns if col not in df.columns],
        'shape': df.shape,
        'na_total': int(df.isna().sum().sum()),
        'na_by_column': df.isna().sum().to_dict(),
        'dtypes': df.dtypes.to_dict()
    }
    return validation

def detect_format(path: Union[str, pathlib.Path]) -> str:
    """Detect file format from extension."""
    path_str = str(path).lower()
    if path_str.endswith('.csv'):
        return 'csv'
    if any(path_str.endswith(ext) for ext in ['.parquet', '.pq', '.parq']):
        return 'parquet'
    raise ValueError(f'Unsupported format: {path_str}')

def read_df(path: Union[str, pathlib.Path]) -> pd.DataFrame:
    """Read DataFrame from file with automatic format detection."""
    path_obj = pathlib.Path(path)
    fmt = detect_format(path_obj)
    
    if fmt == 'csv':
        try:
            sample = pd.read_csv(path_obj, nrows=5)
            date_cols = [col for col in sample.columns if 'date' in col.lower()]
            if date_cols:
                return pd.read_csv(path_obj, parse_dates=date_cols)
            return pd.read_csv(path_obj)
        except Exception as e:
            raise RuntimeError(f'Failed to read CSV: {e}')
    
    elif fmt == 'parquet':
        try:
            return pd.read_parquet(path_obj)
        except Exception as e:
            raise RuntimeError('Parquet engine not available. Install pyarrow or fastparquet.')

def write_df(df: pd.DataFrame, path: Union[str, pathlib.Path]) -> pathlib.Path:
    """Write DataFrame to file with automatic format detection."""
    path_obj = pathlib.Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    fmt = detect_format(path_obj)
    
    if fmt == 'csv':
        df.to_csv(path_obj, index=False)
    elif fmt == 'parquet':
        try:
            df.to_parquet(path_obj)
        except Exception as e:
            raise RuntimeError('Parquet engine not available. Install pyarrow or fastparquet.')
    
    return path_obj

def calculate_technical_indicators(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
    """Calculate common technical indicators for stock data."""
    df = df.copy()
    
    # Simple Moving Averages
    df['sma_10'] = df[price_col].rolling(window=10).mean()
    df['sma_20'] = df[price_col].rolling(window=20).mean()
    df['sma_50'] = df[price_col].rolling(window=50).mean()
    
    # Price Returns
    df['daily_return'] = df[price_col].pct_change()
    df['weekly_return'] = df[price_col].pct_change(5)
    
    # Volatility
    df['volatility_10'] = df['daily_return'].rolling(window=10).std()
    df['volatility_20'] = df['daily_return'].rolling(window=20).std()
    
    # Price vs SMA ratios
    df['price_sma_10_ratio'] = df[price_col] / df['sma_10']
    df['price_sma_20_ratio'] = df[price_col] / df['sma_20']
    
    return df

# Storage utility functions (for use in notebooks)
def list_data_files(directory: Union[str, pathlib.Path], pattern: str = "*") -> List[pathlib.Path]:
    """List files in a data directory."""
    dir_path = pathlib.Path(directory)
    return list(dir_path.glob(pattern))

def get_latest_file(directory: Union[str, pathlib.Path], pattern: str = "*") -> pathlib.Path:
    """Get the most recent file in a directory."""
    files = list_data_files(directory, pattern)
    if not files:
        raise FileNotFoundError(f"No files found in {directory} matching {pattern}")
    return max(files, key=lambda x: x.stat().st_mtime)

def ensure_data_directories():
    """Ensure all data directories exist."""
    directories = ['data/raw', 'data/processed', 'data/backup']
    for dir_path in directories:
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("Data directories ensured: raw, processed, backup")


def load_data(file_path):
    """Load and prepare the processed TSLA data"""
    df = pd.read_csv(file_path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill()
    df.dropna(inplace=True)
    
    return df

def prepare_features(df):
    """Prepare features and target variable"""
    feature_columns = [
        'open', 'high', 'low', 'close', 'volume', 
        'sma_10', 'sma_20', 'sma_50', 
        'daily_return', 'weekly_return', 
        'volatility_10', 'volatility_20',
        'price_sma_10_ratio', 'price_sma_20_ratio'
    ]
    
    available_features = [col for col in feature_columns if col in df.columns]
    X = df[available_features]
    y = df['target']
    
    return X, y, available_features

def train_model(X_train, y_train):
    """Train and return Random Forest model"""
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=5,
        min_samples_split=10
    )
    model.fit(X_train, y_train)
    return model

def save_model(model, file_path):
    """Save trained model to file"""
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

def load_model(file_path):
    """Load trained model from file"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def generate_plots(df, predictions, output_path):
    """Generate and save analysis plots"""
    plt.figure(figsize=(12, 6))
    cumulative_returns = (1 + df['actual_return']).cumprod()
    plt.plot(cumulative_returns)
    plt.title('Cumulative Strategy Returns')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    plt.savefig(f'{output_path}/cumulative_returns.png')
    plt.close()
    
    # Feature importance plot
    plt.figure(figsize=(10, 6))
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    plt.barh(importance['feature'], importance['importance'])
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(f'{output_path}/feature_importance.png')
    plt.close()