from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# load model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model.pkl")
model = joblib.load(MODEL_PATH)

FEATURE_COLS = [
    "delivery_days", "delivery_vs_estimated", "late_delivery_flag",
    "price", "freight_value", "freight_ratio",
    "n_items", "installments_max",
    "product_category", "seller_state", "payment_type",
]

NUMERIC_FEATURES = [
    "delivery_days", "delivery_vs_estimated", "late_delivery_flag",
    "price", "freight_value", "freight_ratio",
    "n_items", "installments_max",
]

CATEGORICAL_FEATURES = ["product_category", "seller_state", "payment_type"]

VALID_PAYMENT_TYPES = ["credit_card", "boleto", "voucher", "debit_card"]


def validate_input(data):
    errors = {}

    # check required fields
    missing = [f for f in FEATURE_COLS if f not in data]
    if missing:
        return {"missing_fields": f"required fields missing: {missing}"}

    # validate numeric fields
    for field in NUMERIC_FEATURES:
        val = data.get(field)
        try:
            float(val)
        except (TypeError, ValueError):
            errors[field] = f"must be a number"

    # validate no negative prices or freight
    for field in ["price", "freight_value"]:
        try:
            if float(data.get(field, 0)) < 0:
                errors[field] = "must be a positive number"
        except (TypeError, ValueError):
            pass

    # validate payment type
    if data.get("payment_type") not in VALID_PAYMENT_TYPES:
        errors["payment_type"] = f"must be one of {VALID_PAYMENT_TYPES}"

    return errors


def build_dataframe(data):
    # compute freight_ratio if not provided
    if "freight_ratio" not in data or data["freight_ratio"] is None:
        price = float(data.get("price", 0))
        freight = float(data.get("freight_value", 0))
        data["freight_ratio"] = freight / price if price > 0 else 0

    row = {col: data.get(col) for col in FEATURE_COLS}
    return pd.DataFrame([row])


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model": "loaded"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "no input data provided"}), 400

    errors = validate_input(data)
    if errors:
        return jsonify({"error": "Invalid input", "details": errors}), 400

    df = build_dataframe(data)
    prediction = int(model.predict(df)[0])
    probability = round(float(model.predict_proba(df)[0][1]), 4)
    label = "positive" if prediction == 1 else "negative"

    return jsonify({
        "prediction": prediction,
        "probability": probability,
        "label": label,
    })


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json()

    if not data or not isinstance(data, list):
        return jsonify({"error": "input must be a JSON array"}), 400

    if len(data) > 100:
        return jsonify({"error": "batch size limit is 100 records"}), 400

    results = []
    for i, record in enumerate(data):
        errors = validate_input(record)
        if errors:
            return jsonify({"error": f"Invalid input at record {i}", "details": errors}), 400

        df = build_dataframe(record)
        prediction = int(model.predict(df)[0])
        probability = round(float(model.predict_proba(df)[0][1]), 4)
        label = "positive" if prediction == 1 else "negative"

        results.append({
            "prediction": prediction,
            "probability": probability,
            "label": label,
        })

    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)