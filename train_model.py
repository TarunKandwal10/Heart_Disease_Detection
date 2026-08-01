# train_model.py
# Heart Disease Prediction - Model Training Script
# This script loads the dataset, preprocesses it, trains a model, and saves it.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# =============================================
# Task 1: Data Understanding and Preprocessing
# =============================================

# Step 1: Load the dataset using Pandas
print("Loading the dataset...")
df = pd.read_csv("heart.csv")

# Step 2: Display first five records
print("\n--- First 5 Records ---")
print(df.head())

# Step 3: Identify features
print("\n--- Dataset Shape ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n--- Column Data Types ---")
print(df.dtypes)

# Numerical features
numerical_features = df.columns[:-1].tolist()  # all columns except target
print("\n--- Numerical Features ---")
print(numerical_features)

# Target variable
print("\n--- Target Variable ---")
print("target")
print(f"Target value counts:\n{df['target'].value_counts()}")

# Step 4: Check for missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())
print(f"\nTotal missing values: {df.isnull().sum().sum()}")

# Step 5: Split dataset into 80% training and 20% testing
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# =============================================
# Task 2: Model Development
# =============================================

# Building a Random Forest Classifier
print("\n--- Training Random Forest Classifier ---")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating model using Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Saving the trained model using Joblib
joblib.dump(model, "model.pkl")
print("\nModel saved as 'model.pkl'")

print("\n--- Model Training Complete ---")
