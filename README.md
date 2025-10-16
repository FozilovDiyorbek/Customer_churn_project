# Customer Churn Prediction Project

This repository contains an end-to-end machine learning project that predicts whether a customer will churn (i.e., leave the service) based on various features. The dataset used in this project comes from a bank and includes demographic, geographic, and account-related information about customers.

---
####  Try using the demo version
You can use this code in Streamlit in a ready-made Demo format by following this link: "https://customerchurnproject-qhitdnc2ayaa5gy2fmevzy.streamlit.app/"

## Project Objective

The primary goal is to build and evaluate various machine learning models to predict customer churn and deploy the best-performing model using Streamlit as an interactive web application.

---

## 📁 Project Structure

```
customer_churn_project/
├── data/
│   ├── customer_churn.csv
│   └── processed/
│       ├── cleaned_customer_churn.csv
|       ├── featured_customer_churn.csv 
│
├── notebooks/
│   ├── 1_data_preparation.ipynb
│   ├── 2_eda.ipynb
│   ├── 3_feature_engineering.ipynb
│   ├── 4_model_training.ipynb
│   ├── 5_model_explainability.ipynb
│   └── 6_final_report_and_model_deployment.ipynb
│
├── models/
│   └── best_model.pkl
│
├── stremlit/
│   └── streamlit_app.py
│
├── requirements.txt
└── README.md
```




---

## Technologies Used

- **Languages:** Python  
- **Data Manipulation:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Modeling:** Scikit-learn, XGBoost, LightGBM, CatBoost  
- **Explainability:** SHAP  
- **Web App Deployment:** Streamlit  
- **Version Control:** Git, GitHub  

---

## Model Performance

Multiple models were evaluated using metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC. The top-performing model was:

- **Model:** CatBoostClassifier  
- **Accuracy:** ~85%  
- **ROC-AUC Score:** ~0.88  

---

## ⚙️ How to Run the Project

## 1. **Clone the repository:**
```bash
git clone https://github.com/FozilovDiyorbek/customer_churn_project.git
cd customer_churn_project
```

## 2.Create and activate a virtual environment:
```
python -m venv .venv
```
# On Windows:
```
.venv\Scripts\activate
```
# On Unix/Mac:
```
source .venv/bin/activate
```

## 3.Install the required dependencies:
```
pip install -r requirements.txt
```

## 4.Run the Streamlit app:
```
streamlit run stremlit/streamlit_app.py
```

## Exploratory Data Analysis (EDA)
The EDA includes:

Distribution of churned vs. retained customers

Correlation heatmaps

Categorical feature analysis (e.g., Gender, Geography)

Numerical feature analysis (e.g., CreditScore, Balance, Age)

## Feature Engineering
Feature engineering steps included:

One-hot encoding for categorical variables

Feature scaling for numerical variables

Handling class imbalance using SMOTE

Creating new features from existing ones

## Model Training & Evaluation
The following models were trained and compared:

Logistic Regression

Random Forest

XGBoost

LightGBM

CatBoost

## Evaluation metrics:

Accuracy

Precision, Recall, F1-Score

ROC-AUC Score

Confusion Matrix

## Model Explainability
SHAP (SHapley Additive exPlanations) was used to interpret model predictions and understand feature importance.

## Web App (Streamlit)
A user-friendly web interface was developed with Streamlit to input customer details and predict churn probability in real-time.

## Author
Diyorbek Fozilov
GitHub: https://github.com/FozilovDiyorbek
Email: diyorbekfozilov011@gmail.com  
