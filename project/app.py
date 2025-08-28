from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from src.utils import load_model, prepare_features
import pickle

app = Flask(__name__)

# Load model and scaler
model = load_model('model/random_forest_model.pkl')
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = pd.DataFrame([data])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)
        
        return jsonify({
            'prediction': int(prediction[0]),
            'probability': float(probability[0][1]),
            'message': 'Success'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/run_full_analysis', methods=['POST'])
def run_full_analysis():
    try:
        from src.utils import load_data, generate_plots
        df = load_data('data/tsla_processed.csv')
        X, y, features = prepare_features(df)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Generate plots
        generate_plots(df, predictions, 'reports')
        
        return jsonify({
            'message': 'Analysis complete',
            'plots_generated': ['cumulative_returns.png', 'feature_importance.png']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)