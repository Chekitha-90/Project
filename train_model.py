# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Simulated training data
data = {
    "sleep_hours": [6, 8, 5, 4, 7],
    "screen_time": [9, 5, 10, 12, 6],
    "mood_score": [2, 8, 3, 1, 6],
    "workload": [8, 4, 10, 9, 5],
    "social_interaction": [1, 4, 1, 0, 3],
    "stress_level": [1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)
X = df.drop("stress_level", axis=1)
y = df["stress_level"]

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "stress_predictor.pkl")
print("✅ Model trained and saved.")
