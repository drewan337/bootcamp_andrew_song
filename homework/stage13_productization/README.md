# Project Productization Example

## Overview
This project demonstrates how to productize a machine learning model with a Flask API, proper project structure, and documentation for handoff.

## Setup Instructions
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Jupyter notebook to train the model: `jupyter notebook notebooks/stage13_productization_homework-final.ipynb`
4. Start the Flask API: `python app.py`

## API Endpoints
- `POST /predict`: Accepts JSON with features array
- `GET /predict/<input1>`: Single feature input with defaults
- `GET /predict/<input1>/<input2>`: Two feature inputs with defaults
- `GET /plot`: Returns a sample visualization

## Example Usage
```bash
# POST request
curl -X POST -H "Content-Type: application/json" -d '{"features": [1,2,3,4,5]}' http://localhost:5000/predict

# GET request with one parameter
curl http://localhost:5000/predict/1.0

# GET request with two parameters
curl http://localhost:5000/predict/1.0/2.0