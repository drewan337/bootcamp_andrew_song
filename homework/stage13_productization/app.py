# app.py
from flask import Flask, request, jsonify
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the trained model
model_path = 'model/model.pkl'
if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
else:
    print("Model file not found. Please train the model first.")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({'error': 'No features provided'}), 400
        
        features = data['features']
        
        # Validate input
        if not isinstance(features, list):
            return jsonify({'error': 'Features must be a list'}), 400
        
        if len(features) != 5:
            return jsonify({'error': 'Exactly 5 features required'}), 400
        
        # Convert to numpy array and make prediction
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        
        return jsonify({
            'prediction': float(prediction),
            'features': features,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/<float:input1>', methods=['GET'])
def predict_one(input1):
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Use default values for other features
        features = [float(input1), 5.0, 5.0, 5.0, 5.0]
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        
        return jsonify({
            'prediction': float(prediction),
            'features': features,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/<float:input1>/<float:input2>', methods=['GET'])
def predict_two(input1, input2):
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Use default values for other features
        features = [float(input1), float(input2), 5.0, 5.0, 5.0]
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        
        return jsonify({
            'prediction': float(prediction),
            'features': features,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    if model is not None:
        return jsonify({'status': 'healthy', 'model_loaded': True})
    else:
        return jsonify({'status': 'unhealthy', 'model_loaded': False}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)