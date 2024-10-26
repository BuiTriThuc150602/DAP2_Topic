from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import streamlit as st


@st.cache_data
def load_data():
    df = pd.read_csv("../data/heart_2020_cleaned.csv")
    df["HeartDisease"] = df["HeartDisease"].map({"Yes": 1, "No": 0})
    df["Diabetic"] = df["Diabetic"].replace(
        {"No, borderline diabetes": "No", "Yes (during pregnancy)": "Yes"}
    )
    return df


@st.cache_resource
def build_model(df):
    categorical_cols = df.select_dtypes(exclude=["number"]).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="passthrough",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    return model


@st.cache_data
def train_with_kfold(df, k=5):
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    train_scores = []
    test_scores = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = build_model(df)

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_scores.append(train_accuracy)

        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_scores.append(test_accuracy)

    return train_scores, test_scores


@st.cache_data
def evaluate_model():
    df = load_data()
    model = build_model(df)
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1
