import pandas as pd
import numpy as np
from typing import Dict, Any, List
from .utils import ts, validate_data, write_df, calculate_technical_indicators

class DataCleaner:
    def __init__(self):
        self.required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    
    def drop_missing_columns(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """Drop columns with more than threshold% missing values."""
        missing_ratio = df.isnull().mean()
        columns_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
        df_clean = df.drop(columns=columns_to_drop)
        
        if columns_to_drop:
            print(f"Dropped columns with >{threshold*100}% missing values: {columns_to_drop}")
        
        return df_clean
    
    def fill_missing_values(self, df: pd.DataFrame, numeric_strategy: str = 'median') -> pd.DataFrame:
        """Fill missing values using specified strategies."""
        df_clean = df.copy()
        
        # Fill numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        if numeric_strategy == 'median':
            for col in numeric_cols:
                if df_clean[col].isnull().any():
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        elif numeric_strategy == 'mean':
            for col in numeric_cols:
                if df_clean[col].isnull().any():
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        
        # Forward fill for time series data
        if 'date' in df_clean.columns:
            df_clean = df_clean.sort_values('date')
            df_clean = df_clean.ffill()
        
        return df_clean
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the DataFrame."""
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column for technical indicators")
        
        df_with_indicators = calculate_technical_indicators(df)
        return df_with_indicators
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary target variable for next-day price direction."""
        df_target = df.copy()
        df_target['next_day_close'] = df_target['close'].shift(-1)
        df_target['price_change'] = df_target['next_day_close'] - df_target['close']
        df_target['target'] = (df_target['price_change'] > 0).astype(int)
        
        # Drop the last row which won't have a target
        df_target = df_target.dropna(subset=['target'])
        
        return df_target
    
    def normalize_data(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """Normalize specified columns using StandardScaler."""
        from sklearn.preprocessing import StandardScaler
        
        df_normalized = df.copy()
        
        if columns is None:
            # Default to normalizing all numeric columns except date and target
            numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns
            columns = [col for col in numeric_cols if col not in ['target', 'next_day_close', 'price_change']]
        
        scaler = StandardScaler()
        df_normalized[columns] = scaler.fit_transform(df_normalized[columns])
        
        return df_normalized, scaler
    
    def clean_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Complete data cleaning pipeline."""
        results = {}
        
        # Initial validation
        initial_validation = validate_data(df, self.required_columns)
        results['initial_validation'] = initial_validation
        
        # Step 1: Drop columns with too many missing values
        df_clean = self.drop_missing_columns(df)
        
        # Step 2: Fill remaining missing values
        df_clean = self.fill_missing_values(df_clean)
        
        # Step 3: Add technical indicators
        df_clean = self.add_technical_indicators(df_clean)
        
        # Step 4: Create target variable
        df_clean = self.create_target_variable(df_clean)
        
        # Step 5: Normalize data
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['target', 'next_day_close', 'price_change']]
        df_clean, scaler = self.normalize_data(df_clean, numeric_cols)
        
        # Final validation
        final_validation = validate_data(df_clean, [])
        results['final_validation'] = final_validation
        results['scaler'] = scaler
        
        return df_clean, results

def clean_dataframe(df: pd.DataFrame) -> tuple:
    """Clean a single DataFrame (for use in notebooks)."""
    cleaner = DataCleaner()
    return cleaner.clean_data(df)