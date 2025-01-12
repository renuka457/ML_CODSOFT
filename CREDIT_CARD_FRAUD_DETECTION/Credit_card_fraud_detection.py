# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
print("Loading dataset...")
df = pd.read_csv("credit_card_fraud.csv")  # Replace with your dataset name

# Display first few rows of the dataset
print("Dataset preview:")
print(df.head())

# Check for missing values
print("\nChecking for missing values...")
print(df.isnull().sum())

# Drop rows with missing values (if any)
df.dropna(inplace=True)

# Splitting data into features and target
X = df.drop('Class', axis=1)  # Features (all columns except 'Class')
y = df['Class']  # Target variable (fraudulent or not)

# Checking class distribution
print("\nClass distribution before balancing:")
print(y.value_counts())

# Handling class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
print("\nBalancing dataset using SMOTE...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Checking class distribution after balancing
print("\nClass distribution after balancing:")
print(pd.Series(y_resampled).value_counts())

# Splitting data into train and test sets
print("\nSplitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Feature scaling (Standardize features)
print("\nScaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training the model - Logistic Regression
print("\nTraining Logistic Regression model...")
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# Predictions
y_pred = log_reg.predict(X_test)

# Model evaluation
print("\nLogistic Regression Model Evaluation:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Train a Random Forest Classifier for comparison
print("\nTraining Random Forest Classifier...")
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred_rf = rf.predict(X_test)

# Model evaluation for Random Forest
print("\nRandom Forest Model Evaluation:")
print(classification_report(y_test, y_pred_rf))
print("Accuracy:", accuracy_score(y_test, y_pred_rf))

# Save the models
import joblib
print("\nSaving the trained models...")
joblib.dump(log_reg, "logistic_regression_model.pkl")
joblib.dump(rf, "random_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel training and evaluation complete!")
