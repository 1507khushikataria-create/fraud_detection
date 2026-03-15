# Credit Card Fraud Detection

## Overview

This project uses machine learning to detect fraudulent credit card transactions. The dataset contains anonymized transaction features obtained through PCA transformation. A Random Forest classifier is used to classify transactions as fraudulent or legitimate.

## Dataset

The dataset used is the **Credit Card Fraud Detection Dataset** from Kaggle.

* Total transactions: 284,807
* Fraudulent transactions: 492
* Features: V1–V28 (PCA transformed), Time, Amount
* Target variable: Class

  * 0 → Normal transaction
  * 1 → Fraudulent transaction

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## Exploratory Data Analysis

The following visualizations were used to understand the dataset:

* Correlation heatmap
* Class distribution graph

These visualizations help identify relationships between features and understand the imbalance between normal and fraudulent transactions.

## Machine Learning Model

A **Random Forest Classifier** was used for classification.

Steps in the workflow:

1. Load dataset
2. Data exploration
3. Data visualization
4. Feature and target selection
5. Train-test split
6. Model training
7. Predictions
8. Model evaluation

## Evaluation Metrics

Because the dataset is highly imbalanced, the following metrics were used:

* Precision
* Recall
* F1 Score

These metrics provide better evaluation than accuracy for fraud detection problems.

## Results

The trained model successfully identifies fraudulent transactions using the Random Forest algorithm.

## Future Improvements

* Handle class imbalance using sampling techniques
* Experiment with additional models
* Add ROC curve and confusion matrix visualization
