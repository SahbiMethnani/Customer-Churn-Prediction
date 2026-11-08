# Customer Churn Prediction

A Python-based machine learning project designed to predict customer churn using customer account information, service usage, contract details, and billing data.

The project implements and compares two classification algorithms:
- Logistic Regression
- Decision Tree Classifier

## Overview

Customer churn refers to customers discontinuing their subscriptions or services. Predicting churn helps identify customers who are more likely to leave, allowing businesses to take preventive actions.

This project uses historical customer data to classify customers into two categories:
- `Yes`: Customer is likely to churn.
- `No`: Customer is likely to remain.

## Technologies Used

- Python
- Pandas – Data manipulation and analysis
- Scikit-learn – Machine learning algorithms and preprocessing
- Matplotlib – Data visualization
- Seaborn – Statistical visualization
## Dataset

The dataset contains customer information, including:

| Feature | Description |
|---|---|
| Churn | Customer churn status (target) |
| AccountWeeks | Number of weeks the customer has had an account |
| ContractRenewal | Whether the customer renewed the contract |
| DataPlan | Whether the customer has a data plan |
| DataUsage | Customer data usage |
| CustServCalls | Number of customer service calls |
| DayMins | Daily minutes of usage |
| DayCalls | Number of daily calls |
| MonthlyCharge | Monthly subscription charge |
| OverageFee | Additional usage fees |
| RoamMins | Roaming minutes |

## Implemented Features

1. ZIP dataset extraction and data loading.
2. Dataset inspection and preview.
3. Missing-value detection and removal.
4. Target variable distribution analysis.
5. Data preprocessing and encoding.
6. Feature scaling using StandardScaler.
7. Training and testing dataset splitting.
8. Logistic Regression model training.
9. Decision Tree model training.
10. Model evaluation using classification reports and confusion matrices.
11. Graph generation and result storage.

## Model Evaluation

The project evaluates both models using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
