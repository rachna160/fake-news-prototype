import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# 1. LOAD, LABEL, and MERGE the Datasets
#    (Fixes the 'dataset.csv' error)


# Load 'Fake.csv' and 'True.csv'
try:
    df_fake = pd.read_csv('fake-and-real-news-dataset/Fake.csv')
    df_true = pd.read_csv('fake-and-real-news-dataset/True.csv')
except FileNotFoundError as e:
    print(f"Error loading file: {e}. Check the exact folder name")
    exit()

# Add a label column to each DataFrame
# 0 for Fake news, 1 for True news (This is our target variable 'y')
df_fake['label'] = 0
df_true['label'] = 1

# Merge (concatenate) both DataFrames vertically
data = pd.concat([df_fake, df_true], ignore_index=True)

# Shuffle the dataset to ensure a good mix of Fake and True news
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset successfully merged. Total rows: {len(data)}")
print("-" * 50)

# =========================================================
# 2. PREPARE DATA for ML
# =========================================================

# Separate features (X: the text) and target (y: the label)
X = data['text']
y = data['label']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================================================
# 3. TF-IDF VECTORIZER
# =========================================================

# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit the vectorizer ONLY on the training data and transform it
X_train_vectorized = vectorizer.fit_transform(X_train)

# Use the fitted vectorizer to transform the test data
X_test_vectorized = vectorizer.transform(X_test)

# =========================================================
# 4. TRAIN AND PREDICT MODEL
# =========================================================

# Initialize the Passive Aggressive Classifier
model = PassiveAggressiveClassifier(max_iter=50, random_state=42)

# Train the model
model.fit(X_train_vectorized, y_train)

# Predict on the test data
y_pred = model.predict(X_test_vectorized)

# Calculate accuracy score
score = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {round(score * 100, 2)}%")
print("-" * 50)

# =========================================================
# 5. SAVE MODEL AND VECTORIZER
# =========================================================

# Save the trained model and the fitted vectorizer using joblib
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and Vectorizer successfully saved as model.pkl and vectorizer.pkl")
print("-" * 50)

# =========================================================
# 6. SHOW CONFUSION MATRIX
# =========================================================

# Generate the confusion matrix
# Labels are [0, 1] which correspond to [Fake, True]
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print('Confusion Matrix:\n', cm)

# Visualization of the Confusion Matrix (Better for understanding)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fake (0)', 'True (1)'],
            yticklabels=['Fake (0)', 'True (1)'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Passive Aggressive Classifier')
plt.show()