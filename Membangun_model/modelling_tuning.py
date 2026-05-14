import os
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

# DAGSHUB CONFIG

dagshub.init(
    repo_owner="hanafithri",
    repo_name="MSML_Hana-fithri",
    mlflow=True
)

os.environ["MLFLOW_TRACKING_USERNAME"] = "hanafithri"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "a4f97d528e8223223ce9e5ac360427db9150ad3e"

# LOAD DATASET

df = pd.read_csv(
    "Employee-Attrition-Dataset_preprocessing/employee_attrition_processed.csv"
)

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

print(df["Attrition"].value_counts(normalize=True))

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# HYPERPARAMETER TUNING
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    ),
    param_grid,
    cv=3,
    scoring="f1"
)

# MLFLOW RUN
with mlflow.start_run():

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    # METRICS

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # MANUAL LOGGING

    mlflow.log_param("n_estimators", best_model.n_estimators)
    mlflow.log_param("max_depth", best_model.max_depth)
    mlflow.log_param("min_samples_split", best_model.min_samples_split)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # CONFUSION MATRIX

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")

    plt.savefig("confusion_matrix.png")

    mlflow.log_artifact("confusion_matrix.png")

    # FEATURE IMPORTANCE

    importance = best_model.feature_importances_

    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    plt.figure(figsize=(8,5))

    sns.barplot(
        x="Importance",
        y="Feature",
        data=feature_importance.head(10)
    )

    plt.title("Feature Importance")

    plt.savefig("feature_importance.png")

    mlflow.log_artifact("feature_importance.png")

    # CLASSIFICATION REPORT

    report = classification_report(y_test, y_pred)

    with open("classification_report.txt", "w") as f:
        f.write(report)

    mlflow.log_artifact("classification_report.txt")

    # LOG MODEL

    mlflow.sklearn.log_model(best_model, "model")

    print("Accuracy:", accuracy)