from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load("stress_predictor.pkl")

@app.route("/")
def home():
    return "Mental Health API is Live."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array([[ 
        data["sleep_hours"], 
        data["screen_time"], 
        data["mood_score"],
        data["workload"],
        data["social_interaction"]
    ]])

    prediction = model.predict(features)[0]
    result = "⚠️ High Stress" if prediction == 1 else "✅ Low Stress"
    
    # Example basic suggestion
    suggestion = "Try meditation and limit screen time." if prediction == 1 else "Keep up the good work!"
    
    return jsonify({
        "stress_level": result,
        "suggestion": suggestion
    })

if __name__ == "__main__":
    app.run(debug=True)
from pymongo import MongoClient
client = MongoClient("mongodb://localhost:27017/")
db = client["mental_health"]
logs = db["checkins"]

# After prediction
logs.insert_one(data) # type: ignore
