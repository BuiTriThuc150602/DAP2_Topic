from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
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
            ("classifier", KNeighborsClassifier(n_neighbors=5)),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print metrics to terminal
    # print(f"Độ chính xác: {accuracy:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"F1-score: {f1:.4f}")

    return model, accuracy, precision, recall, f1


# Streamlit app
st.title("Dự đoán nguy cơ mắc bệnh tim")
st.write("Tải dữ liệu và huấn luyện mô hình KNN")

df = load_data()
model, accuracy, precision, recall, f1 = train_model(df)

st.write(f"Độ chính xác: {accuracy:.4f}")
st.write(f"Precision: {precision:.4f}")
st.write(f"Recall: {recall:.4f}")
st.write(f"F1-score: {f1:.4f}")