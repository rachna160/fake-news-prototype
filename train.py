import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load dataset
data = pd.read_csv("dataset.csv")

x = data["text"]
y = data["label"]

# Convert text to numerical vectors
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
x = vectorizer.fit_transform(x)

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train model (Passive Aggressive Classifier works well for text)
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)
score = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {round(score * 100, 2)}%")

# Save model & vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Show confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["fake", "real"])
print("\nConfusion Matrix:\n", cm)
