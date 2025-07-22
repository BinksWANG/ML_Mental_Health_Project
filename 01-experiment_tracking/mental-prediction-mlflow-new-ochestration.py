#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import mlflow

# Set up MLflow tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("mental-health-experiment")

# Data cleaning function
def clean_data(df):
    # Fill missing values for 'Severity' and 'Consultation_History' with 'Unknown'
    df['Severity'] = df['Severity'].fillna('Unknown')
    df['Consultation_History'] = df['Consultation_History'].fillna('Unknown')

    # Handle missing 'Stress_Level' by filling with 'Unknown'
    df['Stress_Level'] = df['Stress_Level'].fillna('Unknown')

    # Convert categorical columns to string types
    categorical_columns = ['Gender', 'Occupation', 'Country', 'Mental_Health_Condition', 'Severity', 'Consultation_History', 'Stress_Level']
    df[categorical_columns] = df[categorical_columns].astype(str)

    # Convert categorical columns to numerical using LabelEncoder
    le = LabelEncoder()
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])

    # Handle missing numerical values by filling with the median
    df['Sleep_Hours'] = df['Sleep_Hours'].fillna(df['Sleep_Hours'].median())
    df['Work_Hours'] = df['Work_Hours'].fillna(df['Work_Hours'].median())
    df['Physical_Activity_Hours'] = df['Physical_Activity_Hours'].fillna(df['Physical_Activity_Hours'].median())

    # Convert numerical columns to appropriate types
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

    return df

# Load and clean data
df = pd.read_csv('../data/mental_health_dataset.csv')
df_cleaned = clean_data(df)

# Select features and target variable
categorical = ['Gender', 'Occupation', 'Country', 'Mental_Health_Condition', 'Severity', 'Consultation_History', 'Stress_Level']
numerical = ['Age', 'Sleep_Hours', 'Work_Hours', 'Physical_Activity_Hours']
target = 'Mental_Health_Condition'

# Split data into train/test BEFORE vectorization (🟢 CHANGED)
train_df, test_df = train_test_split(df_cleaned, test_size=0.2, random_state=42)

# Combine features into a single dictionary for DictVectorizer
train_df['features'] = train_df[categorical + numerical].apply(lambda x: x.to_dict(), axis=1)
test_df['features'] = test_df[categorical + numerical].apply(lambda x: x.to_dict(), axis=1)

# Prepare features and target for modeling
dv = DictVectorizer()

X_train = dv.fit_transform(train_df['features'].tolist())  # 🟢 Fit on training data
y_train = train_df[target].values

X_test = dv.transform(test_df['features'].tolist())        # 🟢 Transform test data
y_test = test_df[target].values

import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

with mlflow.start_run():
    # Prepare DMatrix for XGBoost
    train_dmatrix = xgb.DMatrix(X_train, label=y_train)
    test_dmatrix = xgb.DMatrix(X_test, label=y_test)

    best_params = {
        'objective': 'reg:squarederror',
        'seed': 42,
        'learning_rate': 0.9477136465696124,
        'max_depth': 8,  
        'min_child_weight': 4.94465806978135,
        'reg_alpha': 0.007258751375563831,
        'reg_lambda': 0.13869944407674337,
    }

    mlflow.set_tag("model", "xgboost-best")
    mlflow.log_params(best_params)

    booster = xgb.train(
        params=best_params,
        dtrain=train_dmatrix,
        num_boost_round=10,
        evals=[(test_dmatrix, 'validation')],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    y_pred = booster.predict(test_dmatrix)
    rmse = root_mean_squared_error(y_test, y_pred)
    mlflow.log_metric("rmse", rmse)
    print(f"Final XGBoost RMSE: {rmse}")

    # 🟢 Save preprocessor and booster
    with open("models/preprocessor_xgb.b", "wb") as f_out:
        pickle.dump(dv, f_out)
    mlflow.log_artifact("models/preprocessor_xgb.b", artifact_path="preprocessor")
    mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")






