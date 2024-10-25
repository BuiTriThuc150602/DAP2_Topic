from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import streamlit as st
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# @st.cache_data
def load_data():
    df = pd.read_csv("../data/heart_2020_cleaned.csv")
    df["HeartDisease"] = df["HeartDisease"].map({"Yes": 1, "No": 0})
    df["Diabetic"] = df["Diabetic"].replace(
        {"No, borderline diabetes": "No", "Yes (during pregnancy)": "Yes"}
    )
    return df


def build_model(df):
    # Các cột dạng categorical
    categorical_cols = df.select_dtypes(exclude=["number"]).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="passthrough",
    )

    # Xây dựng pipeline gồm xử lý dữ liệu và mô hình Logistic Regression
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    return model


def train_with_kfold(df, k=5):
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    # Tạo đối tượng K-fold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    train_scores = []
    test_scores = []

    # Huấn luyện qua từng fold
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Tạo mô hình mới cho mỗi fold
        model = build_model(df)

        # Huấn luyện mô hình trên tập huấn luyện
        model.fit(X_train, y_train)

        # Tính accuracy cho tập huấn luyện
        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_scores.append(train_accuracy)

        # Tính accuracy cho tập kiểm tra
        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_scores.append(test_accuracy)

    return train_scores, test_scores

