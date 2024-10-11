from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import streamlit as st
import pandas as pd


@st.cache_data
def load_data():
    df = pd.read_csv("../data/heart_2020_cleaned.csv")
    df["HeartDisease"] = df["HeartDisease"].map({"Yes": 1, "No": 0})
    df["Diabetic"] = df["Diabetic"].replace(
        {"No, borderline diabetes": "No", "Yes (during pregnancy)": "Yes"}
    )
    return df


@st.cache_resource
def train_model(df):
    categorical_cols = df.select_dtypes(exclude=["number"]).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="passthrough",
    )

    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return model, accuracy, precision, recall, f1
