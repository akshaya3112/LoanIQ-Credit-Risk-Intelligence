# 💳 LoanIQ – Credit-Wise Intelligent Loan Approval Prediction System

A Machine Learning-based loan approval prediction system that predicts loan eligibility using applicant financial, demographic, employment, and loan-related information.

---

## 📖 Overview

LoanIQ is an intelligent credit risk assessment system developed using Machine Learning to automate loan approval decisions. The project processes applicant data, performs preprocessing and feature engineering, compares multiple classification models, and predicts loan approval through an interactive Streamlit web application.

---

## ✨ Features

- Intelligent loan approval prediction
- Interactive Streamlit web application
- Complete data preprocessing pipeline
- Missing value handling
- Feature engineering
- Feature scaling
- Real-time prediction
- Model comparison
- User-friendly dashboard

---

## 📊 Dataset

- **Dataset Size:** 1,000 loan applications
- **Features:** 20 applicant attributes
- **Target Variable:** Loan Approved (Yes/No)

### Input Features

- Applicant Income
- Co-applicant Income
- Age
- Dependents
- Credit Score
- Existing Loans
- Debt-to-Income Ratio
- Savings
- Collateral Value
- Loan Amount
- Loan Term
- Employment Status
- Marital Status
- Education Level
- Gender
- Loan Purpose
- Property Area
- Employer Category

---

## 🛠 Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

### Framework

- Streamlit

### Tools

- Jupyter Notebook
- VS Code
- Git
- GitHub

---

## ⚙️ Machine Learning Workflow

1. Data Collection
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Feature Scaling
6. Model Training
7. Model Evaluation
8. Streamlit Deployment

---

## 🤖 Models Implemented

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naïve Bayes

---

## 📈 Model Performance

| Model | Accuracy |
|--------|----------|
| Logistic Regression | **87.0%** |
| Gaussian Naïve Bayes | 86.5% |
| K-Nearest Neighbors | 74.5% |

**Best Performing Model:** Logistic Regression

---

## 🚀 How to Run

### Clone the repository

```bash
git clone https://github.com/akshaya3112/LoanIQ-Credit-Risk-Intelligence.git
```

### Navigate to the project

```bash
cd LoanIQ-Credit-Risk-Intelligence
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
LoanIQ-Credit-Risk-Intelligence/
│
├── app.py
├── LoanIQ_Model.ipynb
├── loan_approval_data.csv
├── requirements.txt
├── README.md
└── assets/
```

---

## 📌 Key Results

- Achieved **87% prediction accuracy** using Logistic Regression.
- Compared three supervised Machine Learning models.
- Implemented complete preprocessing and feature engineering pipeline.
- Built an interactive Streamlit application for real-time loan eligibility prediction.

---

## 🔮 Future Improvements

- Random Forest
- XGBoost
- SMOTE for class balancing
- Explainable AI (SHAP/LIME)
- Hyperparameter Optimization
- Cloud Deployment
- Real-time Credit Bureau Integration

---

## 👨‍💻 Team Members


- Thippireddy Gari Akshaya
- Kondaveeti Preethi
- M. Sanjay Sreekar

**Guide:** Dr. Avnish Thakur

School of Computing & Artificial Intelligence

Lovely Professional University

---

## 📄 License

This project was developed for academic and learning purposes.
