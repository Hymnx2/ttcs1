import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="AI Churn Predictor", page_icon="🎯", layout="centered")

# Tải an toàn các tài sản học máy từ Backend
@st.cache_resource
def load_ml_assets():
    model = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    columns = joblib.load('feature_columns.pkl')
    return model, scaler, columns

try:
    model, scaler, feature_columns = load_ml_assets()
except:
    st.warning("⚠️ Đang khởi tạo hoặc chờ file cấu hình từ Backend pipeline...")
    st.stop()

st.title("🎯 PHẦN MỀM DỰ BÁO KHÁCH HÀNG RỜI MẠNG")
st.markdown("---")

st.subheader("📋 Nhập thông tin hành vi của Khách hàng:")
col1, col2 = st.columns(2)

with col1:
    contract = st.selectbox("Loại hợp đồng (Contract):", ["Month-to-month", "One year", "Two year"])
    tenure = st.slider("Số tháng đã gắn bó (Tenure):", 0, 72, 12)
    internet_service = st.selectbox("Dịch vụ Internet:", ["Fiber optic", "DSL", "No"])
    tech_support = st.selectbox("Hỗ trợ kỹ thuật (Tech Support):", ["No", "Yes", "No internet service"])

with col2:
    monthly_charges = st.number_input("Cước phí hàng tháng ($):", min_value=10.0, max_value=150.0, value=65.0)
    total_charges = st.number_input("Tổng cước tích lũy ($):", min_value=0.0, max_value=9000.0, value=500.0)
    payment_method = st.selectbox("Phương thức thanh toán:", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    paperless = st.selectbox("Nhận hóa đơn điện tử:", ["Yes", "No"])

if st.button("🚀 PHÂN TÍCH VÀ XỬ LÝ RỦI RO", use_container_width=True):
    # Khởi tạo ma trận đầu vào khớp 100% với số lượng cột của tập Train
    input_template = {col: 0 for col in feature_columns}
    
    # Chuẩn hóa dữ liệu số bằng scaler đã lưu từ Train
    df_num = pd.DataFrame([[tenure, monthly_charges, total_charges]], columns=['tenure', 'MonthlyCharges', 'TotalCharges'])
    scaled_num = scaler.transform(df_num)
    
    input_template['tenure'] = scaled_num[0][0]
    input_template['MonthlyCharges'] = scaled_num[0][1]
    input_template['TotalCharges'] = scaled_num[0][2]
    
    # Khớp các lựa chọn của người dùng vào cấu trúc One-Hot tương ứng
    if f'Contract_{contract}' in input_template: input_template[f'Contract_{contract}'] = 1
    if f'InternetService_{internet_service}' in input_template: input_template[f'InternetService_{internet_service}'] = 1
    if f'TechSupport_{tech_support}' in input_template: input_template[f'TechSupport_{tech_support}'] = 1
    if f'PaymentMethod_{payment_method}' in input_template: input_template[f'PaymentMethod_{payment_method}'] = 1
    if f'PaperlessBilling_{paperless}' in input_template: input_template[f'PaperlessBilling_{paperless}'] = 1

    # Chuyển đổi thành DataFrame chuẩn hóa float32 đúng thứ tự cột
    final_features = pd.DataFrame([input_template])[feature_columns].astype(np.float32)
    
    # Dự đoán xác suất rủi ro Churn
    probability = model.predict_proba(final_features)[0][1]
    
    st.markdown("---")
    st.subheader("📊 Kết quả phân tích hệ thống:")
    st.progress(int(probability * 100))
    
    if probability >= 0.5:
        st.error(f"🚨 CẢNH BÁO: Khách hàng có nguy cơ rời mạng cao! (Tỷ lệ: {probability*100:.1f}%)")
    else:
        st.success(f"✅ AN TOÀN: Khách hàng thuộc nhóm ổn định. (Tỷ lệ rủi ro thấp: {probability*100:.1f}%)")
