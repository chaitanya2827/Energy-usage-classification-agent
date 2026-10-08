from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained Decision Tree model
model = joblib.load("energy_usage_model.pkl")

# Load feature names
features = joblib.load("energy_features.pkl")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "Energy Usage Classification API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Check that all required features are present
        missing = [feature for feature in features if feature not in data]

        if missing:
            return jsonify({
                "error": "Missing features",
                "missing": missing
            }), 400

        # Prepare input in the exact order used during training
        input_data = pd.DataFrame([{
            feature: float(data[feature])
            for feature in features
        }])

        # Prediction
        prediction = model.predict(input_data)[0]

        # Prediction probabilities
        probabilities = model.predict_proba(input_data)[0]
        confidence = float(max(probabilities))

        return jsonify({
            "prediction": str(prediction),
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )