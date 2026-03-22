# 💼 Candidate Hiring Prediction System

## 📌 Project Overview
This project is an end-to-end Machine Learning application that predicts whether a candidate is likely to be hired based on their profile, skills, and experience.

The project includes:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Handling class imbalance using SMOTE
- Model training & hyperparameter tuning
- Deployment using Streamlit

---

## 🎯 Problem Statement
Companies receive thousands of resumes and need an efficient way to shortlist candidates.

This model helps predict:
👉 Whether a candidate will be **Hired (1)** or **Not Hired (0)**

---

## 📊 Dataset Features

### 🔢 Numerical Features
- CGPA
- Skills Score
- Soft Skills Score
- Internships
- Projects
- Certifications
- Experience Years
- Hackathons
- Research Papers
- Programming Languages
- Resume Length (Words)
- Age

### 🧮 Engineered Feature
- `total_experience_score = internships + projects + certifications`

### 🔤 Categorical Features
- Education Level (Bachelors, Masters, PhD)
- University Tier (Tier 1, Tier 2, Tier 3)
- Company Type (Startup, Mid-size, MNC)

---

## ⚙️ Tech Stack

- Python 🐍
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- XGBoost
- Matplotlib & Seaborn
- Streamlit (Deployment)

---

## 🔍 Exploratory Data Analysis (EDA)

- Checked missing values and duplicates
- Analyzed class imbalance (~70% vs 30%)
- Visualized feature distributions
- Identified outliers and skewness
- Correlation analysis

---

## ⚖️ Handling Imbalance

The dataset was imbalanced, so:
- Applied **SMOTE (Synthetic Minority Oversampling Technique)**
- Used **F1-score** instead of accuracy for evaluation

---

## 🤖 Models Used

- Logistic Regression
- Naive Bayes
- Decision Tree
- Random Forest
- XGBoost (Best Performing)

---

## 🏆 Best Model

- **XGBoost Classifier**
- Hyperparameter tuning using GridSearchCV
- Achieved improved F1-score and balanced predictions

---

## 🚀 Deployment

The model is deployed using **Streamlit**.

### ▶️ Run the app:
```bash
streamlit run app.py
