import pandas as pd

def get_summary_stats(dataframe, group_col, value_col):
    """Generate and save summary statistics for a DataFrame."""
    stats = dataframe.groupby(group_col)[value_col].agg(['mean', 'median', 'std', 'count'])
    stats.to_csv(f'data/processed/{group_col}_summary.csv')
    return stats
