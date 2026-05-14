import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def preprocess_data():

    # Load dataset
    data = pd.read_csv(
        "../Employee-Attrition-Dataset_raw/employee_attrition.csv"
    )

    # Copy dataframe
    df = data.copy()

    # Drop unnecessary column
    df = df.drop(columns=["Employee_ID"])

    # Handle missing values
    df = df.dropna()

    # Remove duplicates
    df = df.drop_duplicates()

    # Encoding categorical columns
    categorical_cols = df.select_dtypes(
        include=["object"]
    ).columns

    label_encoder = LabelEncoder()

    for col in categorical_cols:
        df[col] = label_encoder.fit_transform(df[col])

    # Feature scaling
    numerical_cols = df.select_dtypes(
        include=["number"]
    ).columns

    target_col = "Attrition"

    numerical_cols = numerical_cols.drop(target_col)

    scaler = StandardScaler()

    df[numerical_cols] = scaler.fit_transform(
        df[numerical_cols]
    )

    # Handle outliers
    for feature in numerical_cols:

        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df = df[
            (df[feature] >= lower_bound) &
            (df[feature] <= upper_bound)
        ]

    # Create output folder
    os.makedirs(
        "Employee-Attrition-Dataset_preprocessing",
        exist_ok=True
    )

    # Save processed dataset
    df.to_csv(
        "Employee-Attrition-Dataset_preprocessing/employee_attrition_processed.csv",
        index=False
    )

    print("Preprocessing Berhasil!")


if __name__ == "__main__":
    preprocess_data()