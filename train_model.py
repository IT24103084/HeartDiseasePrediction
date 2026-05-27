import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier


# Load dataset
df = pd.read_csv("data/heart.csv")

print("Dataset loaded successfully")
print(df.head())

# Most essential selected features
selected_features = [
    "age",
    "cp",
    "thalach",
    "exang",
    "oldpeak",
    "ca",
    "thal"
]

X = df[selected_features]
y = df["target"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Test model
y_pred = model.predict(X_test_scaled)

print("\nModel Evaluation")
print("----------------")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model, scaler, and selected features
joblib.dump(model, "heart_disease_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(selected_features, "selected_features.pkl")

print("\nModel saved successfully")
print("Selected features saved successfully")