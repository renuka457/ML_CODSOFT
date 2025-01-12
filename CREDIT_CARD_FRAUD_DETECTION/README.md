# Credit Card Fraud Detection

## Overview
This project aims to build a machine learning model that can detect fraudulent credit card transactions. We use a dataset that contains historical credit card transactions, with features such as transaction amount, time, and user-related information. The goal is to classify each transaction as either **fraudulent** or **legitimate**.

## Dataset
The dataset used in this project is the **Credit Card Fraud Detection** dataset from Kaggle. It contains transactions made by credit cards in September 2013 by European cardholders. The dataset has 31 features, including anonymized features and a target variable indicating whether the transaction was fraudulent or not.

- **Dataset Link**: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

### Features
- `V1`, `V2`, ..., `V28`: Anonymized features.
- `Time`: Time elapsed since the first transaction in the dataset.
- `Amount`: The monetary value of the transaction.
- `Class`: The target variable indicating whether the transaction was fraudulent (`1`) or legitimate (`0`).

## Objectives
- **Data Preprocessing**: Clean the data, handle missing values, and normalize features.
- **Model Training**: Train machine learning models like Logistic Regression, Decision Trees, and Random Forests.
- **Model Evaluation**: Evaluate the performance of each model using appropriate metrics (accuracy, precision, recall, F1 score, confusion matrix).
- **Hyperparameter Tuning**: Tune model hyperparameters for better performance.
- **Save the Best Model**: Save the model for future predictions.

## Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebooks (for easy exploration and visualization)

## Installation

### 1. Clone the Repository
Clone this repository to your local machine:

```bash
git clone https://github.com/your_username/CODSOFT.git
cd CODSOFT/Credit_Card_Fraud_Detection
