# src/utils.py
import matplotlib.pyplot as plt
import pickle
import os

def save_model(model, filepath):
    """Save model to file using pickle"""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath):
    """Load model from file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def create_actual_vs_predicted_plot(y_test, y_pred, save_path=None):
    """Create actual vs predicted values plot"""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    
    if save_path:
        plt.savefig(save_path)
    
    return plt