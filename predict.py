import torch
import torch.nn.functional as F
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

#Predict from one row in lesion_features.csv
data = pd.read_csv("lesion_features.csv")

# Pick lesion
x = data.drop(columns=["Label"]).values[-1]

# Scale
x_scaled = scaler.transform([x])
x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

# Predict
with torch.no_grad():
    logit = model(x_tensor).item()
    prob  = torch.sigmoid(torch.tensor(logit)).item()

THRESHOLD = 0.45 
classification = "Malignant" if prob > THRESHOLD else "Benign"

print(f"\nPrediction: {classification}")
print(f"Probability of Malignancy: {prob:.4f}")
