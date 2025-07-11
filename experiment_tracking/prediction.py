import pandas as pd
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("mental-health-experiment")

import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv('../data/mental_health_dataset.csv')

# Drop User_ID (not a predictor)
df = df.drop('User_ID', axis=1)

# Encode target column (Yes=1, No=0)
le = LabelEncoder()
df['Mental_Health_Condition'] = le.fit_transform(df['Mental_Health_Condition'])

# Identify categorical columns (object type)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print("Categorical columns to encode:", categorical_cols)

# One-hot encode categorical features
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Split features and target
X = df.drop('Mental_Health_Condition', axis=1)
y = df['Mental_Health_Condition']

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Sanity check: all features are numeric
print("\nFeature data types after encoding:\n", X_train.dtypes)

with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"🔗 MLflow run ID: {run_id}")

    # -------------------------------
    # 🔥 Train Logistic Regression model
    # -------------------------------
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)

    # -------------------------------
    # 📊 Evaluate model
    # -------------------------------
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    print(f"\n✅ Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

    # -------------------------------
    # 📌 Log parameters and metrics
    # -------------------------------
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 200)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)

    # -------------------------------
    # 📊 Log confusion matrix as artifact
    # -------------------------------
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # Save plot locally and log as artifact
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.close()

    # -------------------------------
    # 💾 Log trained model
    # -------------------------------
    mlflow.sklearn.log_model(model, "model")

    # ==========================================
    # 📦 Register model in Model Registry
    # ==========================================
    model_uri = f"runs:/{run_id}/model"
    registered_model_name = "MentalHealthPredictionModel"

    result = mlflow.register_model(model_uri, registered_model_name)
    print(f"\n📦 Registered model: {registered_model_name} (version {result.version})")

    # ==========================================
    # 🚦 Promote model to Production
    # ==========================================
    client = MlflowClient()
    client.transition_model_version_stage(
        name=registered_model_name,
        version=result.version,
        stage="Production",
        archive_existing_versions=True  # Archive older production models
    )
    print(f"🚀 Promoted model version {result.version} to Production")