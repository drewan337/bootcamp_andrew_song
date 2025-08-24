"""
Outlier detection and handling functions for financial data.
Part of Stage 07: Outlier Analysis for Tomorrow's Stock Trend Predictor project.
"""

import pandas as pd
import numpy as np
from typing import Union, Tuple, Dict, Any

def detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """
    Return boolean mask for IQR-based outliers.
    
    Parameters:
    -----------
    series : pd.Series
        Input data series
    k : float, default=1.5
        Multiplier for IQR (controls outlier strictness)
    
    Returns:
    --------
    pd.Series
        Boolean mask where True indicates outliers
    
    Assumptions:
    ------------
    - Distribution is reasonably summarized by quartiles
    - k=1.5 is standard for moderate outlier detection
    - Financial returns often have heavy tails, making IQR robust
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (series < lower) | (series > upper)

def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Return boolean mask for Z-score outliers where |z| > threshold.
    
    Returns:
    --------
    pd.Series
        Boolean mask where True indicates outliers
    
    Assumptions:
    ------------
    - Data follows approximately normal distribution
    - Sensitive to heavy tails in financial data
    - Standard threshold of 3.0 captures ~99.7% of normal data
    """
    mu = series.mean()
    sigma = series.std(ddof=0)
    z = (series - mu) / (sigma if sigma != 0 else 1.0)
    return z.abs() > threshold

def winsorize_series(series: pd.Series, lower: float = 0.05, upper: float = 0.95) -> pd.Series:
    """
    Winsorize a series by capping extreme values at specified percentiles.
    
    Returns:
    --------
    pd.Series
        Winsorized series with extreme values capped
    
    Assumptions:
    ------------
    - Extreme values contain some useful information but should be constrained
    - Preserves data points while reducing outlier influence
    - 5% winsorizing is common for financial data
    """
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lower=lo, upper=hi)

def analyze_outliers_impact(df: pd.DataFrame, target_col: str = 'daily_return') -> Dict[str, Any]:
    """
    Comprehensive outlier analysis for financial data.
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary with outlier analysis results
    """
    results = {}
    
    # Detect outliers using both methods
    iqr_outliers = detect_outliers_iqr(df[target_col])
    z_outliers = detect_outliers_zscore(df[target_col])
    
    # Calculate summary stats
    summ_all = df[target_col].describe()[['mean', '50%', 'std', 'min', 'max']].rename({'50%': 'median'})
    summ_filtered = df.loc[~iqr_outliers, target_col].describe()[['mean', '50%', 'std', 'min', 'max']].rename({'50%': 'median'})
    
    # Apply winsorizing
    winsorized_data = winsorize_series(df[target_col])
    summ_w = winsorized_data.describe()[['mean', '50%', 'std', 'min', 'max']].rename({'50%': 'median'})
    
    # Compare results
    comp = pd.concat(
        {
            'all': summ_all,
            'filtered_iqr': summ_filtered,
            'winsorized': summ_w
        }, axis=1
    )
    
    # Calculate skewness and kurtosis
    from scipy.stats import skew, kurtosis
    skew_kurt_stats = pd.DataFrame({
        'all': [skew(df[target_col]), kurtosis(df[target_col])],
        'filtered_iqr': [skew(df.loc[~iqr_outliers, target_col]), kurtosis(df.loc[~iqr_outliers, target_col])],
        'winsorized': [skew(winsorized_data), kurtosis(winsorized_data)]
    }, index=['skewness', 'kurtosis'])
    
    # Store results
    results['outlier_counts'] = {
        'iqr': iqr_outliers.sum(),
        'z_score': z_outliers.sum(),
        'total_rows': len(df)
    }
    
    results['summary_stats'] = comp
    results['skew_kurtosis'] = skew_kurt_stats
    results['iqr_outliers_mask'] = iqr_outliers
    results['z_outliers_mask'] = z_outliers
    results['winsorized_data'] = winsorized_data
    
    return results

def add_outlier_flags_to_data(df: pd.DataFrame, target_col: str = 'daily_return') -> pd.DataFrame:
    """
    Add outlier flags to dataframe for downstream processing.
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with added outlier flag columns
    """
    df_out = df.copy()
    df_out['outlier_iqr'] = detect_outliers_iqr(df_out[target_col])
    df_out['outlier_z'] = detect_outliers_zscore(df_out[target_col])
    return df_out