from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app, supports_credentials=True)

with open("horse_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler2.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/")
def home():
    return "live", 200

@app.route("/crash-test")
def crash_test():
    return "ok", 200

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        response = jsonify({ "status": "OK" })
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response, 200
    data = request.get_json()
    processed = preprocess_input(data)
    scaled = scaler.transform([processed])
    prediction = model.predict(scaled)[0]
    probabilities = model.predict_proba(scaled)[0]

    return jsonify({
        "prediction": int(prediction),
        "confidence": float(max(probabilities)),
        "probabilities": {
            "euthanized": float(probabilities[0]),
            "died": float(probabilities[1]),
            "lived": float(probabilities[2])
        }
    })

def preprocess_input(input_data):
    return [
        float(input_data.get("pulse", 0)),
        float(input_data.get("rectal_temp", 0)),
        float(input_data.get("respiratory_rate", 0)),
        3, 3, 2, 3, 3, 2, 7.0, 3, 2, 4,
        float(input_data.get("packed_cell_volume", 45)),
        float(input_data.get("total_protein", 7)),
        2.0, 1, 1, 1, 0, 0.5
    ]

if __name__ == '__main__':
    from os import getenv
    port = int(getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port)