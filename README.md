# 🏦 Churn Modelling: Predicting Bank Customer Retention

This project demonstrates how machine learning can be applied to predict whether a customer will stay with or leave a bank. Using a dataset of 10,000 bank records, we built and tested several ML models and developed an app to visualize the results and predictions.

---

## 📌 Problem Statement

The goal is to predict the **likelihood of customer churn**, enabling the bank to proactively take steps to retain at-risk customers.

---

## 📂 Dataset Overview

We used a dataset containing **10,000 bank customer records**. Key features included:

- Credit Score
- Geography
- Gender
- Age
- Tenure (years as a bank customer)
- Balance
- Number of Bank Products Used
- Has Credit Card
- Is Active Member
- Estimated Salary
- Exited (Target: 1 = Left the bank, 0 = Stayed)

---

## 🧹 1. Data Cleaning

We read the data using **Pandas** and removed irrelevant fields such as `CustomerID`, `Surname`, and other identifiers.

This left us with a clean, feature-rich dataset ready for exploration and modeling.

---

## 📊 2. Data Analysis

Using **Pandas**, **Matplotlib**, and **Seaborn**, we performed exploratory data analysis. Key insights:

- **Class Imbalance**:  
  - ~80% of customers stayed (`Exited = 0`)
  - ~20% of customers left (`Exited = 1`)
  
- **Gender-based Churn**:  
  - 25% of female customers left  
  - 16% of male customers left

- **Geography-based Churn**:  
  - Customers from **Germany** showed the highest churn rate  
  - Nearly 1 in 3 German customers exited

To address the class imbalance, we used **SMOTE (Synthetic Minority Over-sampling Technique)** during model training.

---

## 🤖 3. Machine Learning Models

We tested **6 machine learning models** to predict churn:

1. Logistic Regression  
2. Decision Tree  
3. Random Forest  
4. K-Nearest Neighbors  
5. Support Vector Machine (with SMOTE)  
6. XGBoost  

> 🔄 **SMOTE** was used with the `imblearn` package to oversample the minority class in the **training set only** — improving recall and generalization.

---

## ⚙️ Tools & Libraries

- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Scikit-learn
- Imbalanced-learn (`imblearn`)
- XGBoost

---

## ✅ Results & Insights

- Using **SMOTE** helped balance the training data and improve model recall — a key metric when dealing with churn prediction.
- Models like **Random Forest**, **XGBoost**  achieved higher accuracy and recall.
- Geography and gender were strong predictors of churn risk.

---

## 🚀 Next Steps

- Integrate with real-time customer data for proactive intervention
- Use SHAP or LIME for model explainability
- Deploy as a web app or dashboard for stakeholder use

---

## 📁 Project Structure

├── data/ # Raw and processed datasets
├── notebooks/ # Jupyter notebooks for EDA and modeling
├── models/ # Saved models (pickle, h5, etc.)
└── README.md # Project overview

