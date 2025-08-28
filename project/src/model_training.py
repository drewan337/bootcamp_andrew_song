## 6. Create src/model_training.py:
from src.utils import load_data, prepare_features, train_model, save_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

def main():
    # Load and prepare data
    df = load_data('data/tsla_processed.csv')
    X, y, feature_names = prepare_features(df)
    
    # Split data
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = train_model(X_train_scaled, y_train)
    
    # Save model and scaler
    save_model(model, 'model/random_forest_model.pkl')
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Model training complete and saved!")

if __name__ == '__main__':
    main()