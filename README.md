# Credit Card Fraud Detection 💳🔍

## 📋 Project Description
This project implements a credit card fraud detection system using machine learning techniques. The objective is to analyze financial transactions and automatically identify fraudulent transactions using different classification algorithms.
## 🎯 Objectives

Analyze a credit card transaction dataset
Preprocess the data to optimize model performance
Implement and compare different classification algorithms
Evaluate model performance with appropriate metrics
Select the best model for fraud detection

## 📊 Dataset
The project uses the creditcard.csv dataset which contains:

284,807 transactions in total
31 features (characteristics)

Features V1-V28: Results of a Principal Component Analysis (PCA) to anonymize the data
Feature Time: Time elapsed since the first transaction
Feature Amount: Transaction amount
Feature Class: Target variable (0 = normal transaction, 1 = fraud)



## 🔍 Dataset Characteristics

✅ No missing data
⚠️ Highly imbalanced dataset: ~99.83% normal transactions, ~0.17% fraud


## 🛠️ Technologies Used
python# Main libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## Machine Learning
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb

## Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


## 🔄 Methodology

### 1. 📈 Exploratory Data Analysis (EDA)
- Analysis of class distribution
- Study of correlations between features
- Outlier analysis
- Visualization of fraud patterns

### 2. 🔧 Preprocessing
- Separation of features (X) and target (Y)
- Train/test split (80%/20%)
- Data normalization with StandardScaler
- Handling class imbalance

### 3. 🤖 Modeling
Four main approaches were tested:

#### a) **DecisionTree and Random Forest Classifier**
- Hyperparameter tuning with GridSearchCV
- Optimization to maximize recall
- Decision threshold adjustment

#### b) **XGBoost Classifier**
- High-performance gradient boosting algorithm
- Parameters optimized for imbalanced data

#### c) **Deep Neural Network (DNN)**
- Multi-layer architecture with Dropout
- Optimization with Adam optimizer
- Loss function: binary crossentropy

### 4. 📊 Evaluation
Metrics used:
- **Recall** (main priority - detect maximum frauds)
- **Precision** (avoid false positives)
- **AUC-ROC** (overall performance)
- **Precision-Recall Curve**
- **Confusion Matrix**

## 🏆 Results

| Model | Recall | Precision | AUC Score | Performance |
|--------|--------|-----------|-----------|-------------|
| Random Forest | 86% | 74% | Very good | ⭐⭐⭐⭐ |
| XGBoost | 90% | 34% | 0.9475 | ⭐⭐⭐⭐ |
| **DNN** | **90%** | **38%** | **0.947** | ⭐⭐⭐⭐⭐ |

### 🥇 Best Model: Deep Neural Network
- **High recall**: Effectively detects fraud
- **Excellent AUC**: Optimal discrimination capability
- **Generalization**: Robust performance on unseen data
- **Complex patterns**: Captures non-linear relationships

### Prerequisites:
bashpip install pandas numpy matplotlib seaborn
pip install scikit-learn xgboost
pip install tensorflow
pip install jupyter
Execution

Clone the project and navigate to the folder
Download the dataset: Credit Card Fraud Detection
Launch Jupyter Notebook:

bash   jupyter notebook fraud-detection.ipynb

Execute cells sequentially to reproduce the analysis

## 📈 Practical Applications
This system can be used by:

Banks: Real-time detection of suspicious transactions
Fintechs: Securing online payments
E-commerce: Protection against payment fraud
Payment processors: Automatic transaction filtering


## 👥 Team
Project completed as part of the MA513 - Machine Learning for Cybersecurity course
Team Members:

Thibaud RIMBERT
Eliott HANGARD
Arsène GALLIEZ
Hugo DOS ANJOS DOS SANTOS

Institution: IPSA
📄 License
This project is for educational purposes as part of the Machine Learning course.
