# Customer Churn Prediction

This project focuses on predicting customer churn for a subscription-based business using historical customer data, including usage behavior and demographics. The goal is to develop a machine learning model that can identify customers likely to churn, enabling proactive retention strategies.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Approach](#approach)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Workflow](#workflow)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Problem Statement

Customer churn is a critical metric for subscription-based businesses. Churn occurs when customers stop using a company's services, leading to revenue loss. By predicting churn, businesses can identify at-risk customers and take preventive measures to retain them.

---

## Dataset

The dataset contains historical data of customers, including features such as:
- Customer demographics (e.g., gender, age, geography)
- Account information (e.g., tenure, balance, and products held)
- Behavior patterns (e.g., usage statistics, services subscribed)

**Dataset Source**: [Bank Customer Churn Prediction Dataset](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)

---

## Approach

1. **Exploratory Data Analysis (EDA):**
   - Understand data distribution and relationships between features.
   - Visualize trends and patterns using graphs and charts.

2. **Data Preprocessing:**
   - Handle missing values and outliers.
   - Encode categorical variables using techniques like One-Hot Encoding.
   - Normalize numerical features for better model performance.

3. **Feature Engineering:**
   - Select important features based on correlations and feature importance scores.

4. **Model Selection:**
   - Train and evaluate models like:
     - Logistic Regression
     - Random Forest
     - Gradient Boosting (e.g., XGBoost, LightGBM)

5. **Evaluation Metrics:**
   - Use metrics like Accuracy, Precision, Recall, F1-Score, and ROC-AUC for performance evaluation.

---

## Technologies Used

- Python
- Libraries: 
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `scikit-learn`, `xgboost`

---

## Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone <repository_url>
   cd CODSOFT/Customer_Churn_Prediction

