# 🧠 Employee Salary Prediction App

This is a Streamlit-based Machine Learning web application that predicts whether an individual earns more than $50K/year based on various demographic and work-related attributes. The app is built using Python and the UCI Adult Income dataset.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [How It Works](#how-it-works)
- [Setup Instructions](#setup-instructions)
- [Screenshots](#screenshots)
- [Future Scope](#future-scope)
- [References](#references)

---

## 📖 Overview

The goal is to build a binary classification model to predict if a person's income is **<=50K** or **>50K** using attributes like age, occupation, education level, and hours worked per week.

This project helps in:
- HR analytics
- Job profiling
- Resource allocation

---

## ✅ Features

- Upload CSV file (UCI Adult Dataset)
- Data preview before and after cleaning
- Model training using:
  - Decision Tree
  - Random Forest
  - Logistic Regression
- Accuracy evaluation
- Feature importance visualization (if applicable)
- Interactive form for user input predictions
- Simple browser-based UI using Streamlit

---

## 🛠️ Tech Stack

- **Language**: Python
- **Framework**: Streamlit
- **Libraries**:
  - pandas
  - scikit-learn
  - matplotlib
- **ML Models**:
  - Decision Tree
  - Random Forest
  - Logistic Regression

---

## ⚙️ How It Works

1. Upload the CSV file (e.g., `adult.csv`)
2. Data is cleaned:
   - Replace '?' values
   - Drop irrelevant or inconsistent rows
   - Label encode categorical features
3. User selects a model
4. Model is trained and tested
5. Accuracy is displayed
6. User inputs custom data to make predictions
7. App returns predicted salary class

---

## 💻 Setup Instructions

### Prerequisites

Ensure you have Python 3.8+ installed.

### Clone the Repository

```bash
git clone https://github.com/<your-username>/employee-salary-prediction
cd employee-salary-prediction
