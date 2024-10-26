import streamlit as st
import pandas as pd

from LogisticRegression.LogisticRegression_model import load_data, build_model, train_with_kfold, evaluate_model
# from RandomForest.Random_model import load_data, train_model_RF


def run_ui(model, df, accuracy, precision, recall, f1):
    st.title("Chỉ số của mô hình")
    st.write("Số lượng mẫu:", df.shape[0])
    st.write("Số lượng biến đầu vào :", df.shape[1] - 1)
    st.write("Số lượng nhóm:", df["HeartDisease"].value_counts())
    st.title("Đánh giá mô hình dự đoán bệnh tim")
    metrics_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Score": [accuracy, precision, recall, f1],
        }
    )

    st.bar_chart(metrics_df.set_index("Metric"))
    st.write("### Kết Luận về mô hình")
    if f1 < 0.75:
        st.warning("Mô hình này chưa đáng tin cậy cho các quyết định quan trọng.")
    else:
        st.success("Mô hình đạt mức độ tin cậy tốt cho quyết định.")

    st.write("### Phân tích và Gợi ý")
    st.write(
        """
    - Với độ chính xác **{:.2f}**, mô hình có khả năng dự đoán đúng ở mức tương đối. Tuy nhiên, bạn nên cân nhắc sử dụng mô hình này cho các quyết định quan trọng.
    - Precision **{:.2f}** cho thấy mô hình phát hiện các trường hợp dương tính với độ chính xác khá cao, thích hợp cho việc dự đoán trong môi trường không cần phải có độ nhạy cao.
    - Nếu bạn quan tâm đến việc phát hiện tất cả các trường hợp nguy cơ bệnh tim, Recall **{:.2f}** có thể chưa đủ cao để đảm bảo độ bao phủ.
    - F1 Score **{:.2f}** cho thấy mô hình có sự cân bằng tốt giữa Precision và Recall.
        """.format(
            accuracy, precision, recall, f1
        )
    )

    st.title("Dự đoán nguy cơ mắc bệnh tim")

    col1, col2 = st.columns(2)

    with col1:
        bmi = st.number_input(
            "Chỉ số BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.1
        )
        smoking = st.selectbox("Hút thuốc", ["Yes", "No"])
        physical_health = st.number_input(
            "Sức khỏe thể chất (số ngày không khỏe trong 30 ngày qua)",
            min_value=0,
            max_value=30,
            value=0,
        )
        diff_walking = st.selectbox("Khó khăn khi đi bộ", ["Yes", "No"])
        age_category = st.selectbox("Nhóm tuổi", df["AgeCategory"].unique())
        diabetic = st.selectbox("Tiểu đường", df["Diabetic"].unique())
        sleep_time = st.number_input(
            "Thời gian ngủ (giờ)", min_value=1, max_value=24, value=7
        )
        kidney_disease = st.selectbox("Bệnh thận", ["Yes", "No"])

    with col2:
        alcohol_drinking = st.selectbox("Uống rượu", ["Yes", "No"])
        stroke = st.selectbox("Đột quỵ", ["Yes", "No"])
        mental_health = st.number_input(
            "Sức khỏe tinh thần (số ngày không khỏe trong 30 ngày qua)",
            min_value=0,
            max_value=30,
            value=0,
        )
        sex = st.selectbox("Giới tính", ["Female", "Male"])
        race = st.selectbox("Chủng tộc", df["Race"].unique())
        physical_activity = st.selectbox("Hoạt động thể chất", ["Yes", "No"])
        gen_health = st.selectbox("Sức khỏe tổng quát", df["GenHealth"].unique())
        asthma = st.selectbox("Hen suyễn", ["Yes", "No"])
        skin_cancer = st.selectbox("Ung thư da", ["Yes", "No"])

    input_data = pd.DataFrame(
        {
            "BMI": [bmi],
            "Smoking": [smoking],
            "AlcoholDrinking": [alcohol_drinking],
            "Stroke": [stroke],
            "PhysicalHealth": [physical_health],
            "MentalHealth": [mental_health],
            "DiffWalking": [diff_walking],
            "Sex": [sex],
            "AgeCategory": [age_category],
            "Race": [race],
            "Diabetic": [diabetic],
            "PhysicalActivity": [physical_activity],
            "GenHealth": [gen_health],
            "SleepTime": [sleep_time],
            "Asthma": [asthma],
            "KidneyDisease": [kidney_disease],
            "SkinCancer": [skin_cancer],
        }
    )

    if st.button("Dự đoán"):
        prediction = model.predict(input_data)
        st.write("### Kết quả dự đoán", prediction)
        if prediction[0] == 1:
            st.error("Nguy cơ mắc bệnh tim cao")
        else:
            st.success("Nguy cơ mắc bệnh tim thấp")


df = load_data()
model = build_model(df)
accuracy, precision, recall, f1 = evaluate_model()

# model,df_encoded, accuracy, precision, recall, f1 = train_model_RF(df)
run_ui(model, df, accuracy, precision, recall, f1)
