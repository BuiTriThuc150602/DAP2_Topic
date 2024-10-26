from sklearn.discriminant_analysis import StandardScaler
import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from imblearn.combine import SMOTEENN

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


@st.cache_data
def load_data():
    df = pd.read_csv("../data/heart_2020_cleaned.csv")
    df["HeartDisease"] = df["HeartDisease"].map({"Yes": 1, "No": 0})
    df["Diabetic"] = df["Diabetic"].replace(
        {"No, borderline diabetes": "No", "Yes (during pregnancy)": "Yes"}
    )
    return df


@st.cache_resource
def train_model_RF(df):
    ordinal_cols = ["BMI_Category", "AgeCategory", "Race", "GenHealth"]
    boolean_cols = [
        "HeartDisease",
        "Sex",
        "Smoking",
        "AlcoholDrinking",
        "Stroke",
        "DiffWalking",
        "Diabetic",
        "PhysicalActivity",
        "Asthma",
        "KidneyDisease",
        "SkinCancer",
    ]

    ordinal_mappings = {
        "BMI_Category": [
            "Underweight",
            "Normal weight",
            "Overweight",
            "Obesity I",
            "Obesity II",
            "Obesity III",
        ],
        "AgeCategory": [
            "18-24",
            "25-29",
            "30-34",
            "35-39",
            "40-44",
            "45-49",
            "50-54",
            "55-59",
            "60-64",
            "65-69",
            "70-74",
            "75-79",
            "80 or older",
        ],
        "Race": [
            "White",
            "Black",
            "Asian",
            "Hispanic",
            "American Indian/Alaskan Native",
            "Other",
        ],
        "GenHealth": ["Poor", "Fair", "Good", "Very good", "Excellent"],
    }

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "ord",
                OrdinalEncoder(
                    categories=[ordinal_mappings[col] for col in ordinal_cols]
                ),
                ordinal_cols,
            ),  # Ordinal encoding
            (
                "ohe",
                OneHotEncoder(drop="first"),
                boolean_cols,
            ),  # OneHotEncoding for boolean columns
        ],
        remainder="passthrough",
    )

    df_transformed = preprocessor.fit_transform(df)

    # Convert the transformed data back to a DataFrame with appropriate column names
    # Ordinal columns retain original names, while OneHotEncoder generates new columns
    ohe_columns = preprocessor.named_transformers_["ohe"].get_feature_names_out(
        boolean_cols
    )
    final_columns = (
        ordinal_cols
        + list(ohe_columns)
        + [col for col in df.columns if col not in ordinal_cols + boolean_cols]
    )

    df_encoded = pd.DataFrame(df_transformed, columns=final_columns)

    X = df_encoded.drop("HeartDisease_Yes", axis=1)
    y = df_encoded["HeartDisease_Yes"]
    scale = StandardScaler()
    X_scaled = scale.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    smote_enn = SMOTEENN(random_state=42)
    X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)

    model = RandomForestClassifier()
    model.fit(X_train_resampled, y_train_resampled)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    precision = precision_score(y_test, model.predict(X_test))
    recall = recall_score(y_test, model.predict(X_test))
    f1 = f1_score(y_test, model.predict(X_test))

    return model,df_encoded, accuracy, precision, recall, f1
