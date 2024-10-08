# Financial Risk Management: Credit Card Default Prediction

## Project Overview
This project focuses on predicting whether a credit card client will default on their payment using machine learning techniques. The dataset used is the [Default of Credit Card Clients Dataset](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset), which consists of 30,000 examples and 24 features. The goal is to build a predictive model to estimate the likelihood of a client defaulting on their credit card payment in the upcoming month.

## Dataset
The dataset contains the following:
- **30,000 rows** representing credit card clients.
- **24 features** including demographic, financial, and behavioral data such as credit limits, payment history, age, marital status, and gender.
- The target variable is `default.payment.next.month`, a binary indicator (1 for default, 0 for non-default).

## Problem Statement
The primary task is to build a binary classification model that predicts whether a credit card client will default on their payments in the following month. This model aims to help financial institutions in identifying high-risk clients, improving credit risk management, and reducing financial losses due to defaults.

## Objectives
- Preprocess the dataset by handling missing values, scaling numerical features, and encoding categorical variables.
- Train various machine learning models to predict client default.
- Evaluate model performance using metrics such as accuracy, precision, recall, and F1-score.
- Optimize the model's performance through hyperparameter tuning.
- Compare the results with baseline models and the findings of the associated research paper.

## Project Steps

1. **Data Exploration and Cleaning**
   - Load the dataset and inspect its structure.
   - Handle missing values, encode categorical variables, and scale numerical features.
   
2. **Data Preprocessing**
   - Split the dataset into training and test sets.
   - Apply feature scaling and one-hot encoding where necessary.
   
3. **Model Training**
   - Train baseline models like Logistic Regression and Decision Tree.
   - Experiment with more advanced models such as Random Forest and Support Vector Machine (SVM).

4. **Model Evaluation**
   - Evaluate models using cross-validation.
   - Use metrics like accuracy, precision, recall, F1-score, and confusion matrix to measure performance.
   
5. **Hyperparameter Optimization**
   - Use GridSearchCV or RandomizedSearchCV to tune hyperparameters for better model performance.

6. **Final Model Testing**
   - Evaluate the final model on the test set to ensure it generalizes well on unseen data.

7. **Reporting and Conclusion**
   - Summarize findings and results.
   - Compare performance with previous studies and discuss potential improvements.

## Installation

To run this project, you will need to install the following libraries:

```bash
pip install numpy pandas scikit-learn matplotlib
