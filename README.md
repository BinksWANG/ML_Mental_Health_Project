# 🧠 ML Mental Health Project

## 📌 Problem Statement

Mental health is a growing concern worldwide, influenced by many personal and environmental factors. This project aims to build a machine learning pipeline that predicts whether an individual is likely to report a mental health condition based on their demographics, stress levels, lifestyle choices (e.g., sleep, work, physical activity), and occupation.

### Objectives:
- Identify key indicators of mental health conditions
- Build a predictive ML model using real-world data
- Deploy the model in a web environment with Flask and Docker
- Implement monitoring to track performance and detect drift
- Follow best practices for ML development and MLOps

---

## 🗂️ Dataset Description

**Title:** Comprehensive Mental Health Insights  
**Records:** 1000 individuals  
**Source:** Public domain dataset ([Kaggle Link](https://www.kaggle.com/datasets/bhadramohit/mental-health-dataset?select=mental_health_dataset.csv))  

### Features:
- **Demographics:** Age, Gender, Country, Occupation (IT, Healthcare, Engineering, etc.)
- **Mental Health Status:** Reported Condition (Yes/No), Severity (Low/Medium/High)
- **Consultation History:** Consulted a mental health professional (Yes/No)
- **Stress Level:** Low / Medium / High
- **Lifestyle Metrics:**
  - Sleep Duration (hours per day)
  - Work Hours per Week
  - Physical Activity (hours per week)
  
---

## 🧰 Technologies Used

| Category                 | Tools/Frameworks                |
|--------------------------|-----------------------------------|
| Infrastructure           | Google Cloud Platform           |
| Containerization         | Docker                          |
| Web Framework            | Flask                           |
| Workflow Orchestration   | Mage                            |
| Experiment Tracking      | MLflow                          |
| Monitoring               | Evidently AI                    |
| IaC                      | Terraform                       |
| CI/CD                    | Makefile, pre-commit, pipenv    |
| ML Libraries             | Scikit-learn, Pandas, NumPy     |
| Testing & Quality        | Pytest, flake8, Black           |

---

## ⚙️ Project Workflow
### 🔹 Phase 0: Environmental Setting
This project follows a modular, automated workflow:

### 🔹 Phase 1: Data Exploration and Preprocessing
- Load and inspect the Kaggle dataset.
- Perform Exploratory Data Analysis (EDA).
- Handle missing values, encode categorical features, scale numeric data.
- Engineer features to enhance model performance.

### 🔹 Phase 2: Model Development
- Train ML models (Logistic Regression, Random Forest, XGBoost).
- Evaluate models using Accuracy, F1-score, and ROC-AUC.
- Perform hyperparameter tuning (GridSearchCV or Hyperopt).

### 🔹 Phase 3: Experiment Tracking & Model Registry
- Use MLflow to:
  - Log experiments
  - Track metrics and parameters
  - Register and version models for production.

### 🔹 Phase 4: Deployment
- **Batch Deployment**: Scheduled inference jobs on AWS EC2.
- **Web API Deployment**: REST API using Flask and FastAPI, deployed via AWS Lambda + API Gateway in Docker containers.

### 🔹 Phase 5: Monitoring
- Monitor data drift and model performance using Evidently AI.
- Set alerts for performance degradation and trigger retraining if necessary.

---

## ☁️ Cloud & Infrastructure

Infrastructure is provisioned using **Terraform** on AWS:  

| Service           | Description                           |
|-------------------|---------------------------------------|
| **S3**            | Stores datasets and model artifacts  |
| **EC2**           | Hosts batch inference jobs           |
| **Lambda & API Gateway** | Serves predictions via REST API |
| **Docker**        | Ensures containerized deployments    |

---

## 🚀 Project Setup

### Prerequisites
- AWS account with IAM user credentials
- Docker installed locally
- Python 3.9+
- Pipenv installed

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ml-mental-health.git
   cd ml-mental-health

---

## 🧰 Best Practices Implemented

| Feature               | Status            |
|-----------------------|--------------------|
| Unit Tests            | ✅                 |
| Integration Test      | ✅                 |
| Linter / Formatter    | ✅ (Black, flake8) |
| Makefile              | ✅                 |
| Pre-commit Hooks      | ✅                 |
