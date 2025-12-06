import torch
import numpy as np
import joblib
import pandas as pd
from classifier import LesionClassifier

# Load scaler
scaler = joblib.load("scaler.pkl")

# Load trained model
model = LesionClassifier()
model.load_state_dict(torch.load("lesion_classifier.pt"))
model.eval()

# ---------------------------------------------------------
# Example: Predict from one row in lesion_features.csv
# ---------------------------------------------------------
data = pd.read_csv("lesion_features.csv")

# Pick last extracted lesion
x = data.drop(columns=["Label"]).values[-1]
x_scaled = scaler.transform([x])
x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

with torch.no_grad():
    prob = model(x_tensor).item()
    classification = "Malignant" if prob > 0.5 else "Benign"

print(f"\nPrediction: {classification}")
print(f"Probability of Malignancy: {prob:.4f}")
