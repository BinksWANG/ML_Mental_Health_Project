# 🧠 Mental Health Insights: End-to-End ML Project

## 📌 Problem Statement

Mental health is a growing concern worldwide, influenced by many personal and environmental factors. This project aims to build a machine learning pipeline that predicts whether an individual is likely to report a mental health condition based on their demographics, stress levels, lifestyle choices (e.g., sleep, work, physical activity), and occupation.

### Objectives:
- Identify key indicators of mental health conditions
- Build a predictive ML model using real-world data
- Deploy the model in a web environment with flask and docker
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

## 🧪 Experiment Tracking & Model Registry

- **Tool:** MLflow  
- **Use:** Track experiments, store metrics (accuracy, F1-score), register models

---

## ⚙️ ML Pipeline

1. **Data Ingestion & Cleaning**
2. **Exploratory Data Analysis (EDA)**
3. **Preprocessing & Feature Engineering**
4. **Model Training & Evaluation**
5. **Experiment Tracking (MLflow)**
6. **Model Registration**

---

## 🔄 Workflow Orchestration

- **Tool:** Prefect  
- **Use:** Automate training and evaluation pipeline  
- **Deployment:** Cloud-hosted with Prefect Cloud

---

## ☁️ Cloud & Infrastructure

- **Cloud Provider:** AWS  
- **Services:** S3, EC2, Lambda, API Gateway  
- **IaC Tool:** Terraform (for provisioning infrastructure)

---

## 🚀 Model Deployment

- **Type:** Batch and Web API Deployment  
- **Frameworks:** Docker, FastAPI  
- **Deployment Targets:**
  - **Batch**: Scheduled inference via EC2
  - **Web**: API Gateway + AWS Lambda (FastAPI container)

---

## 📊 Model Monitoring

- **Tool:** Evidently AI  
- **Monitoring Features:**
  - Data drift and prediction drift
  - Model performance metrics
  - Alert triggers (e.g., accuracy drop, input distribution shift)
  - Optional: Auto-retraining or alert notifications

---

## 🔁 Reproducibility

- **Makefile** with targets for `train`, `deploy`, `monitor`
- Environment managed with `pipenv` and `requirements.txt`
- All steps documented and reproducible
- Dependency versions specified

---

## 🧰 Best Practices Implemented

| Feature               | Status            |
|-----------------------|--------------------|
| Unit Tests            | ✅                 |
| Integration Test      | ✅                 |
| Linter / Formatter    | ✅ (Black, flake8) |
| Makefile              | ✅                 |
| Pre-commit Hooks      | ✅                 |
