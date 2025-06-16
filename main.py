from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os
import traceback

app = Flask(__name__)
CORS(app, supports_credentials=True)

# Global variables for model and scaler
model = None
scaler = None

# Try to load model and scaler with error handling
try:
    print("Loading model...")
    with open("horse_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully")
    
    print("Loading scaler...")
    with open("scaler2.pkl", "rb") as f:
        scaler = pickle.load(f)
    print("Scaler loaded successfully")
    
except Exception as e:
    print(f"Error loading model or scaler: {str(e)}")
    print(f"Traceback: {traceback.format_exc()}")
    model = None
    scaler = None

@app.route("/")
def home():
    status = {
        "status": "live",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }
    return jsonify(status), 200

@app.route("/crash-test")
def crash_test():
    return jsonify({"status": "ok", "message": "Server is running"}), 200

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        response = jsonify({"status": "OK"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response, 200
    
    try:
        # Check if model and scaler are loaded
        if model is None or scaler is None:
            return jsonify({
                "error": "Model or scaler not loaded",
                "model_loaded": model is not None,
                "scaler_loaded": scaler is not None
            }), 500
        
        # Get and validate input data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        print(f"Received data: {data}")
        
        # Preprocess input
        processed = preprocess_input(data)
        print(f"Processed features: {processed}")
        
        # Scale features
        scaled = scaler.transform([processed])
        print(f"Scaled features shape: {scaled.shape}")
        
        # Make prediction
        prediction = model.predict(scaled)[0]
        probabilities = model.predict_proba(scaled)[0]
        
        result = {
            "prediction": int(prediction),
            "confidence": float(max(probabilities)),
            "probabilities": {
                "euthanized": float(probabilities[0]),
                "died": float(probabilities[1]),
                "lived": float(probabilities[2])
            }
        }
        
        print(f"Prediction result: {result}")
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "error": f"Prediction failed: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

def preprocess_input(input_data):
    """
    Preprocess input data to match your model's expected features.
    This is simplified - you may need to adjust based on your actual model training.
    """
    try:
        # Basic vital signs
        pulse = float(input_data.get("pulse", 50))
        rectal_temp = abs(float(input_data.get("rectal_temp", 38.0)) - 37.8)  # Distance from normal
        respiratory_rate = float(input_data.get("respiratory_rate", 18))
        
        # Lab values
        packed_cell_volume = float(input_data.get("packed_cell_volume", 45))
        total_protein = float(input_data.get("total_protein", 7))
        
        # Pain mapping (from your notebook)
        pain_mapping = {
            'severe_pain': 0.0, 'extreme_pain': 1.0, 'depressed': 2.0,
            'moderate': 3.0, 'mild_pain': 4.0, 'alert': 5.0
        }
        pain = pain_mapping.get(input_data.get('pain', 'alert'), 5.0)
        
        # Temperature of extremities mapping
        temp_mapping = {'cool': 0, 'cold': 1, 'warm': 2, 'normal': 3}
        temp_extremities = temp_mapping.get(input_data.get('temp_of_extremities', 'normal'), 3)
        
        # Peripheral pulse mapping
        peripheral_mapping = {'absent': 0, 'reduced': 1, 'increased': 2, 'normal': 3}
        peripheral_pulse = peripheral_mapping.get(input_data.get('peripheral_pulse', 'normal'), 3)
        
        # Capillary refill time mapping
        capillary_mapping = {'more_3_sec': 0, '3': 1, 'less_3_sec': 2}
        capillary_refill = capillary_mapping.get(input_data.get('capillary_refill_time', 'less_3_sec'), 2)
        
        # Simple shock probability calculation
        shock_prob = 0
        if pulse > 80: shock_prob += 0.3
        if input_data.get('mucous_membrane') in ['pale_cyanotic', 'dark_cyanotic']: shock_prob += 0.25
        if input_data.get('capillary_refill_time') == 'more_3_sec': shock_prob += 0.15
        if input_data.get('pain') in ['severe_pain']: shock_prob += 0.2
        if respiratory_rate > 25: shock_prob += 0.15
        shock_prob = min(max(shock_prob, 0), 1)
        
        # Create feature array (adjust based on your actual model's expected features)
        features = [
            pulse, rectal_temp, respiratory_rate, temp_extremities,
            peripheral_pulse, capillary_refill, pain, 3, 2, 7.0, 3, 2, 4,
            packed_cell_volume, total_protein, 2.0, 1, 1, 1, 0, shock_prob
        ]
        
        print(f"Created {len(features)} features: {features}")
        return features
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise
