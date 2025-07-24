# 🧠 ML Mental Health Project

An end-to-end MLOps project for predicting mental health conditions based on demographics, lifestyle, and stress-related data. This implementation uses modern ML and MLOps tools to ensure reproducibility, scalability, and robust deployment.

---

## 📌 Problem Statement

Mental health is a growing global concern, influenced by various personal and environmental factors. This project builds a machine learning pipeline to predict whether an individual is likely to report a mental health condition using their:

- Demographics
- Stress levels
- Lifestyle choices (sleep, work, physical activity)
- Occupation

### Objectives
- Identify key indicators of mental health conditions
- Build and evaluate predictive ML models
- Deploy models via web APIs and batch jobs using Docker and Flask
- Implement monitoring to track performance and detect drift
- Follow MLOps best practices for production-ready systems

---

## 🗂️ Dataset Description

**Title:** Comprehensive Mental Health Insights  
**Records:** 1,000 individuals  
**Source:** Kaggle Dataset: https://www.kaggle.com/datasets/bhadramohit/mental-health-dataset?select=mental_health_dataset.csv  

### Features
- **Demographics:** Age, Gender, Country, Occupation (IT, Healthcare, Engineering, etc.)
- **Mental Health Status:** Reported Condition (Yes/No), Severity (Low/Medium/High)
- **Consultation History:** Consulted a mental health professional (Yes/No)
- **Stress Level:** Low / Medium / High
- **Lifestyle Metrics:**
  - Sleep Duration (hours/day)
  - Work Hours per Week
  - Physical Activity (hours/week)

---

## 🧰 Technologies Used

| Category                 | Tools/Frameworks                |
|--------------------------|-----------------------------------|
| Infrastructure           | Google Cloud Platform (GCP)     |
| Containerization         | Docker                          |
| Web Framework            | Flask                           |
| Workflow Orchestration   | Mage                             |
| Experiment Tracking      | MLflow                          |
| Infrastructure as Code   | Terraform                       |
| CI/CD                    | Makefile, Pre-commit Hooks, Pipenv |
| ML Libraries             | Scikit-learn, Pandas, NumPy     |
| Testing & Code Quality   | Pytest, Flake8, Black           |

---

## ⚙️ Project Workflow

### Phase 0: Environment Setup

#### Google Cloud Platform
- Create and configure a GCP project
- Set up a VM instance
- Generate SSH keys for secure access

📸 Screenshots:

<img width="1020" src="https://github.com/user-attachments/assets/782856fb-2a89-4a1b-a9d1-15633a086aad" />

#### Install Anaconda
```bash
# Download Anaconda installer
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh

# Run the installer
bash Anaconda3-2022.05-Linux-x86_64.sh
```
📸 Screenshot:

<img width="826" src="https://github.com/user-attachments/assets/270170ab-2416-47f2-acea-3f84d2246972" />

#### Install Docker and Docker Compose
```bash
# 🐳 Install Docker and Docker Compose

# Update package list
sudo apt update

# Install Docker
sudo apt install -y docker.io

# Create a directory for software
mkdir soft
cd soft

# Download Docker Compose binary
wget https://github.com/docker/compose/releases/download/v2.36.1/docker-compose-linux-x86_64

# Rename and make it executable
mv docker-compose-linux-x86_64 docker-compose
chmod +x docker-compose
```
📸 Screenshot:

<img width="866" src="https://github.com/user-attachments/assets/84e63054-ab60-42f1-bce8-04a3a8aa3dd3" />

---

### Phase 1: Data Exploration & Preprocessing

- Load and inspect Kaggle dataset
- Perform EDA with visualizations
- Handle missing values and outliers
- Feature engineering and scaling

📸 Screenshot:
<img width="1153" src="https://github.com/user-attachments/assets/2740f54e-f6f0-4c32-8dde-91fb1ab74944" />

---

### Phase 2: Model Development

- Train ML models (Logistic Regression, Random Forest, XGBoost)
- Evaluate using metrics (Accuracy, F1-score, ROC-AUC)
- Hyperparameter tuning (GridSearchCV, Hyperopt)

```bash
# 🏗️ Set up Python environment with Conda

# Create a new conda environment with Python 3.9
conda create -n ml-env python=3.9

# Activate the environment
conda activate ml-env

# Install project dependencies
pip install -r requirements.txt

# List installed packages
pip list

# Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
📸 Screenshots:

<img width="1421" src="https://github.com/user-attachments/assets/deaac4f6-2efb-4a63-980f-5564ce395a27" />
<img width="1408" src="https://github.com/user-attachments/assets/4ef165d5-2ffc-412d-a944-edde6363c93f" />

---

### Phase 3: Experiment Tracking & Model Registry

- Log experiments and metrics in MLflow
- Register and version models for production

```bash
jupyter nbconvert --to=script mental-prediction-mlflow-new-ochestration.ipynb
```

📸 Screenshots:

<img width="1718" src="https://github.com/user-attachments/assets/af87eeab-b976-4bca-9ddd-8da2e9b9ca43" />
<img width="1892" src="https://github.com/user-attachments/assets/2291e311-458b-42b9-b87f-8166c01afb79" />

---

### Phase 4: Workflow Orchestration

- Use Mage to automate training and evaluation pipelines

```bash
// install mage
git clone https://github.com/mage-ai/mlops.git
mv mlops/ mlops-mage

//start.sh
export PROJECT_NAME=mlops 
export MAGE_CODE_PATH=home/src 
export SMTP_EMAIL=$SMTP_EMAIL 
export SMTP_PASSWORD=$SMTP_PASSWORD 
docker-compose up

//run start.sh file
./scripts/start.sh
```

- Schedule jobs and monitor runs in the Mage UI
- http://localhost:6789

📸 Screenshots:

<img width="1751" height="845" alt="18  mage_ingest" src="https://github.com/user-attachments/assets/ef89f8bb-d560-4148-9763-3309e81ca28d" />

---

### Phase 5: Deployment

- **scikit-learn and pipenv**

```bash
pip freeze | grep scikit-learn
python -c "import sklearn; print(sklearn.__version__)"

//create env
pipenv install scikit-learn==1.0.2 flask --python3.9
```
or
```bash
//create env
pipenv --python3.9
pipenv install scikit-learn==1.0.2 flask
```

- **Web API Deployment**: Flask app deployed using Docker containers

```bash
//enter
pipenv shell

pipenv install gunicorn
gunicorn --bind=0.0.0.0:9696 predict:app

//DOCKER 
docker build -t mental-health-pridiction-service:v1 .
docker run -it --rm -p 9696:9696 mental-health-pridiction-service:v1
```

📸 Screenshots:

<img width="1218" height="806" alt="19  predict" src="https://github.com/user-attachments/assets/41e8dd55-d944-4db5-9621-b4f6f1a2a2fa" />
<img width="879" height="323" alt="20  flask application" src="https://github.com/user-attachments/assets/c85ba619-7211-4d40-84a4-bc69db201ae8" />
<img width="1441" height="321" alt="21  gunicorn" src="https://github.com/user-attachments/assets/9476503f-82ae-4cd0-810e-25b9777b54c5" />
<img width="1295" height="863" alt="23  test and result" src="https://github.com/user-attachments/assets/c5dea1df-ebcb-4bf4-bc42-5b39bbac25df" />

---

## ✅ Best Practices Implemented

| Practice                   | Status            |
|----------------------------|--------------------|
| Unit Tests                 | ✅                 |
| Integration Tests          | ✅                 |
| Linter / Formatter         | ✅ (Black, flake8) |
| Makefile                   | ✅                 |
| Pre-commit Hooks           | ✅                 |
| Experiment Tracking        | ✅ (MLflow)        |
| Workflow Orchestration     | ✅ (Mage)          |
| Containerization           | ✅ (Docker)        |

---

## 📌 Future Improvements

- Add Grafana dashboards for live monitoring
- Enable auto-retraining on drift detection
- Extend deployment to multi-region support on GCP

---
