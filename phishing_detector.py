import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
data = pd.read_csv("emails.csv")

print("Total Emails Loaded:", len(data))

# Preprocessing function
def preprocess(text):
    text = str(text).lower()

    # Replace URLs
    text = re.sub(r"http\S+|www\S+", " URL ", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    return text

# Apply preprocessing
data["email"] = data["email"].apply(preprocess)

# Features and labels
X = data["email"]
y = data["label"]

# Convert text to numerical features
vectorizer = TfidfVectorizer(stop_words="english")
X_features = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\n=== Model Performance ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Save model
joblib.dump((model, vectorizer), "model.pkl")
print("\nModel saved as model.pkl")

# Single email prediction
print("\n==============================")
print("PHISHING EMAIL DETECTION")
print("==============================")

email_text = input("\nEnter Email Content:\n")

email_text = preprocess(email_text)

email_vector = vectorizer.transform([email_text])

prediction = model.predict(email_vector)[0]
confidence = model.predict_proba(email_vector).max() * 100

print("\n===== RESULT =====")

if prediction == "phishing":
    print("Prediction : PHISHING EMAIL")
else:
    print("Prediction : SAFE EMAIL")

print("Confidence :", round(confidence, 2), "%")