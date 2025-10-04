from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import traceback
import os

# Config - filenames (assumed to sit next to this file)
MODEL_FILENAME = "complication_predictor_pipeline.pkl"
MEDMAP_FILENAME = "medication_map.csv"

app = Flask(__name__)
CORS(app, origins="*", supports_credentials=True)

# Load model
if not os.path.exists(MODEL_FILENAME):
    raise FileNotFoundError(f"Model file not found: {MODEL_FILENAME} - please place it here or retrain locally.")

model = joblib.load(MODEL_FILENAME)

# Load medication map (used as a safe, explainable mapping)
if os.path.exists(MEDMAP_FILENAME):
    med_map_df = pd.read_csv(MEDMAP_FILENAME)
else:
    # If file missing, create a minimal fallback map
    med_map_df = pd.DataFrame([
        {"Complication": "Infection", "Recommended_Medication": "Antibiotics", "Dosage": "500 mg/day", "Duration": "7 days"},
        {"Complication": "Bleeding", "Recommended_Medication": "Blood Transfusion + Hemostatic Agent", "Dosage": "2 units", "Duration": "1 days"},
        {"Complication": "Organ Failure", "Recommended_Medication": "IV Fluids + Vasopressors", "Dosage": "2 L/day", "Duration": "5 days"},
        {"Complication": "None", "Recommended_Medication": "Pain Management", "Dosage": "100 mg/day", "Duration": "1 days"}
    ])

# Helper: predictable features order — infer from model if possible
# Try to read expected features from a bundled variable or fall back to common names
EXPECTED_FEATURES = None
try:
    # many sklearn pipelines keep original column transformer info hidden;
    # user must supply features matching training feature names used previously.
    # We'll infer from the training dataset fields commonly used in this project.
    EXPECTED_FEATURES = [
        'Age','Gender','BMI','ASA_Score','Diabetes','Hypertension','Heart_Disease',
        'Preop_Hb','Preop_WBC','Surgery_Type','Duration_Min','Blood_Loss_ml','Vital_Instability',
        'Postop_Hb','Postop_WBC'
    ]
except Exception:
    EXPECTED_FEATURES = None

def recommend_from_complication(complication):
    """Return the medication recommendation row for a complication using med_map_df.
       If not present, return a sensible default."""
    row = med_map_df[med_map_df['Complication'].str.lower() == str(complication).lower()]
    if not row.empty:
        r = row.iloc[0].to_dict()
        return {
            "Recommended_Medication": r.get("Recommended_Medication", "No recommendation"),
            "Dosage": r.get("Dosage", "N/A"),
            "Duration": r.get("Duration", "N/A"),
            "Source": "medication_map.csv"
        }
    # fallback rule-based
    comp = str(complication).lower()
    if "infect" in comp:
        return {"Recommended_Medication": "Antibiotics", "Dosage": "500 mg/day", "Duration": "7 days", "Source": "rule"}
    if "bleed" in comp:
        return {"Recommended_Medication": "Blood Transfusion + Hemostatic Agent", "Dosage": "2 units", "Duration": "1 days", "Source": "rule"}
    if "organ" in comp or "failure" in comp:
        return {"Recommended_Medication": "IV Fluids + Vasopressors", "Dosage": "2 L/day", "Duration": "5 days", "Source": "rule"}
    # None or unknown
    return {"Recommended_Medication": "Pain Management", "Dosage": "100 mg/day", "Duration": "1 days", "Source": "rule"}

def validate_and_build_df(payload):
    """Takes either a dict (single patient) or list of dicts and returns a pandas DataFrame
       containing EXPECTED_FEATURES columns (filling missing with None)."""
    if isinstance(payload, dict):
        records = [payload]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("Payload must be a JSON object or array of objects.")
    df = pd.DataFrame(records)
    if EXPECTED_FEATURES is not None:
        # ensure columns order and existence
        for col in EXPECTED_FEATURES:
            if col not in df.columns:
                df[col] = None
        df = df[EXPECTED_FEATURES].copy()
    # else use what user provided; model will error if missing required columns.
    return df

@app.route("/predict", methods=["POST"])
def predict():
    """
    POST JSON single patient or list of patients.
    Example body (single):
    {
      "Age":60,"Gender":"Male","BMI":26.5,"ASA_Score":3,"Diabetes":"Yes","Hypertension":"Yes",
      "Heart_Disease":"No","Preop_Hb":12.5,"Preop_WBC":7.8,"Surgery_Type":"Whipple",
      "Duration_Min":360,"Blood_Loss_ml":700,"Vital_Instability":"Yes","Postop_Hb":10.8,"Postop_WBC":12.0
    }
    """
    try:
        payload = request.get_json()
        df = validate_and_build_df(payload)
        preds = model.predict(df)
        probs = None
        try:
            probs = model.predict_proba(df).tolist()
        except Exception:
            probs = None
        results = []
        for p_idx, p in enumerate(preds):
            res = {"Complication": str(p)}
            if probs is not None:
                # map classes to probabilities
                classes = list(model.classes_) if hasattr(model, "classes_") else []
                res["Probabilities"] = dict(zip(classes, probs[p_idx])) if classes else probs[p_idx]
            results.append(res)
        return jsonify({"status": "ok", "predictions": results})
    except Exception as e:
        return jsonify({"status":"error", "error": str(e), "trace": traceback.format_exc()}), 400

@app.route("/recommend", methods=["POST"])
def recommend():
    """
    POST a JSON either:
    - {"Complication": "Infection"}  OR
    - patient features (same as /predict) and backend will predict and recommend
    """
    try:
        payload = request.get_json()
        if not payload:
            raise ValueError("Empty request body")
        # If the payload explicitly contains Complication, use it directly
        if isinstance(payload, dict) and "Complication" in payload:
            comp = payload["Complication"]
            rec = recommend_from_complication(comp)
            return jsonify({"status":"ok","complication": comp, "recommendation": rec})
        # else treat as patient features and predict first
        df = validate_and_build_df(payload)
        preds = model.predict(df)
        results = []
        for p in preds:
            rec = recommend_from_complication(p)
            results.append({"Complication": str(p), "Recommendation": rec})
        return jsonify({"status":"ok","recommendations": results})
    except Exception as e:
        return jsonify({"status":"error", "error": str(e), "trace": traceback.format_exc()}), 400

@app.route("/predict_recommend", methods=["POST"])
def predict_recommend():
    """
    Combined endpoint: accept patient(s), predict complication and return medication recommendation.
    """
    try:
        payload = request.get_json()
        df = validate_and_build_df(payload)
        preds = model.predict(df)
        probs = None
        try:
            probs = model.predict_proba(df).tolist()
        except Exception:
            probs = None
        out = []
        for idx, p in enumerate(preds):
            comp = str(p)
            rec = recommend_from_complication(comp)
            entry = {"Complication": comp, "Recommendation": rec}
            if probs is not None:
                classes = list(model.classes_) if hasattr(model, "classes_") else []
                entry["Probabilities"] = dict(zip(classes, probs[idx])) if classes else probs[idx]
            out.append(entry)
        return jsonify({"status":"ok", "results": out})
    except Exception as e:
        return jsonify({"status":"error", "error": str(e), "trace": traceback.format_exc()}), 400

@app.route("/")
def index():
    return jsonify({"status":"ok", "msg": "Perioperative Risk Backend running. Use /predict, /recommend, /predict_recommend."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

