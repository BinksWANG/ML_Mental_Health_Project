🧠 Mental Health Insights: End-to-End ML Project
📌 Problem Statement
Mental health is a growing concern worldwide, with various factors influencing an individual's well-being. This project aims to build a machine learning pipeline that predicts whether an individual is likely to report a mental health condition based on their demographics, stress level, lifestyle choices (e.g., sleep, work, physical activity), and occupation.

The goal is to:

Identify key lifestyle and demographic indicators of mental health conditions

Develop a predictive model to assist in early detection

Deploy the model in a cloud environment

Monitor performance to ensure reliability and fairness over time

🗂️ Dataset Description
Title: Comprehensive Mental Health Insights
Size: 1000 records
Source: Public domain dataset (Kaggle or similar)

Features:
Demographics: Age, gender, country, occupation

Mental Health Status: Reported condition (Yes/No), Severity (Low/Medium/High)

Consultation History: Whether the individual sought professional help

Stress Level: Low / Medium / High

Lifestyle Metrics:

Sleep duration (hours per day)

Work hours per week

Physical activity (hours per week)

🧪 Experiment Tracking & Model Registry
Tool: MLflow

Purpose: Track hyperparameters, metrics (accuracy, F1-score), and model versions

⚙️ ML Pipeline
Data Ingestion: Read and clean dataset

Exploratory Data Analysis: Visualize correlations and distributions

Preprocessing: Encode categorical variables, scale numerical features

Modeling: Train/test split, multiple ML models evaluated (Random Forest, Logistic Regression, XGBoost)

Experiment Tracking: MLflow used for tracking and comparison

Model Registry: Best model stored in MLflow registry

🔄 Workflow Orchestration
Tool: Prefect

Purpose: Automate the training and evaluation steps in the pipeline

Deployed as: Cloud workflow using Prefect Cloud

☁️ Cloud & Infrastructure
Platform: AWS (S3 for data, EC2 for compute, Lambda for deployment)

IaC Tool: Terraform (used to provision S3 buckets, EC2 instances, and IAM roles)

🚀 Model Deployment
Deployment Type: Batch and Web Service

Tool: Docker + FastAPI

Hosting: AWS Lambda (via API Gateway for web requests) and batch job on EC2

📊 Model Monitoring
Tool: Evidently

Metrics Monitored:

Data drift (categorical and numerical)

Prediction distribution

Accuracy/F1-score changes over time

Trigger: Alert and retrain if performance degrades beyond threshold

🔁 Reproducibility
All steps included in a reproducible Makefile

Environment managed with pipenv and requirements.txt

All experiments tracked in MLflow

Instructions for local and cloud execution provided

🧰 Best Practices
Feature	Implemented
Unit tests	✅
Integration test	✅
Linter/Formatter	✅ (Black + flake8)
Makefile	✅
Pre-commit hooks	✅
CI/CD Pipeline	✅ (GitHub Actions)

🧠 Use Case
This project provides the foundation for building a real-world health risk assessment tool. HR departments, healthcare providers, and researchers can leverage this to:

Understand patterns in employee stress and mental health

Recommend early interventions

Drive policy change for better work-life balance
