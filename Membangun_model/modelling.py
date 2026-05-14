import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# LOAD DATASET
df = pd.read_csv(
    "Employee-Attrition-Dataset_preprocessing/employee_attrition_processed.csv"
)

# FITUR & TARGET
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# ENABLE AUTOLOG
mlflow.sklearn.autolog()

# TRAINING MODEL
with mlflow.start_run():

    model = RandomForestClassifier(
        random_state=42
    )

    model.fit(X_train, y_train)

    # PREDIKSI
    y_pred = model.predict(X_test)

    # ACCURACY
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)