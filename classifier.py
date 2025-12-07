import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib


#LOAD DATA
data = pd.read_csv("lesion_features_batch.csv")

# Features and labels
X = data.drop(columns=["Label"]).values
y = data["Label"].values.astype(int)     # 0 = benign, 1 = malignant

X = np.delete(X, [0, 1, 2, 3, 4], axis=1) #used for ablation study removing features


malignant = np.sum(y)
benign = len(y) - malignant
pos_weight_value = benign / malignant   # used later in loss

#Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

joblib.dump(scaler, "scaler.pkl")

#Oversample to balance classes
ros = RandomOverSampler(random_state=42)
X, y = ros.fit_resample(X, y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

#Convert to tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_test  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

#DEFINE THE NEURAL NETWORK
class LesionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

model = LesionClassifier()

#loss and optimizer
pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0012)

#training loop
epochs = 250

for epoch in range(epochs):
    model.train()
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

#results on test set
model.eval()
with torch.no_grad():
    logits = model(X_test)
    probs = torch.sigmoid(logits)    # convert logits → probabilities
    preds_class = (probs > 0.6).float()
    
probs_np = probs.numpy()
y_test_np = y_test.numpy()

accuracy = accuracy_score(y_test, preds_class)
print("\nTest Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, preds_class))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds_class))

#saved model
torch.save(model.state_dict(), "lesion_classifier.pt")
print("\nModel saved as lesion_classifier.pt")
