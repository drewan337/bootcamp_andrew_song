# src/data_processing.py
import pandas as pd
import numpy as np

def load_sample_data(n_samples=1000):
    """Generate sample data for demonstration"""
    np.random.seed(42)
    X = np.random.rand(n_samples, 5) * 10
    y = 2*X[:,0] + 3*X[:,1] - 1.5*X[:,2] + 0.5*X[:,3] + np.random.randn(n_samples)*0.5
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets"""
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
