import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="AI Churn Predictor", page_icon="🎯", layout="centered")

# Tải cấu hình tài sản học máy
@st.cache_resource
def load_machine_learning_assets():
    model = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    columns = joblib.load('feature_columns.pkl')
    return model, scaler, columns

try:
    model, scaler, feature_columns = load_machine_learning_assets()
except Exception as e:
    st.error("Lỗi: Không tìm thấy các tệp tin .pkl. Vui lòng chạy file 'main_pipeline.py' trước để sinh dữ liệu!")
    st.stop()

# Tiêu đề phần mềm
st.title("🎯 PHẦN MỀM DỰ BÁO KHÁCH HÀNG RỜI BỎ DỊCH VỤ VIỄN THÔNG")
st.markdown("---")

st.subheader("📋 Nhập thông tin hành vi của Khách hàng:")

# Thiết kế layout dạng 2 cột song song
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

# Xử lý khi nhấn nút dự báo
if st.button("🚀 PHÂN TÍCH VÀ XỬ LÝ RỦI RO", use_container_width=True):
    
    # Tạo cấu trúc dữ liệu trống tương thích với tập huấn luyện
    input_template = {col: 0 for col in feature_columns}
    
    # Đồng bộ hóa và chuẩn hóa các trường số học đầu vào
    df_num = pd.DataFrame([[tenure, monthly_charges, total_charges]], columns=['tenure', 'MonthlyCharges', 'TotalCharges'])
    scaled_num = scaler.transform(df_num)
    
    input_template['tenure'] = scaled_num[0][0]
    input_template['MonthlyCharges'] = scaled_num[0][1]
    input_template['TotalCharges'] = scaled_num[0][2]
    
    # Thiết lập One-Hot mã hóa các trường phân loại người dùng lựa chọn
    if f'Contract_{contract}' in input_template: input_template[f'Contract_{contract}'] = 1
    if f'InternetService_{internet_service}' in input_template: input_template[f'InternetService_{internet_service}'] = 1
    if f'TechSupport_{tech_support}' in input_template: input_template[f'TechSupport_{tech_support}'] = 1
    if f'PaymentMethod_{payment_method}' in input_template: input_template[f'PaymentMethod_{payment_method}'] = 1
    if f'PaperlessBilling_{paperless}' in input_template: input_template[f'PaperlessBilling_{paperless}'] = 1

    # Chuyển đổi sang định dạng DataFrame chuẩn để đưa vào mô hình
    final_features = pd.DataFrame([input_template])[feature_columns]
    
    # Dự đoán xác suất rủi ro Churn (Yes)
    probability = model.predict_proba(final_features)[0][1]
    
    st.markdown("---")
    st.subheader("📊 Kết quả phân tích hệ thống:")
    st.progress(int(probability * 100))
    
    # Đưa ra kết quả cảnh báo trực quan
    if probability >= 0.5:
        st.error(f"🚨 CẢNH BÁO: Khách hàng có nguy cơ rời mạng rất cao! (Tỷ lệ rủi ro: {probability*100:.1f}%)")
        st.markdown("""
        **💡 GỢI Ý CHIẾN LƯỢC GIỮ CHÂN (MARKETING / CSKH):**
        * Khách hàng hiện tại đang dùng gói cước ngắn hạn với mức chi phí cao. 
        * **Hành động tức thì:** Tự động gửi mã coupon giảm giá **15% cước phí** trong 6 tháng tiếp theo nếu khách hàng đồng ý nâng cấp và ký cam kết gói hợp đồng hạn định **1 năm**.
        """)
    else:
        st.success(f"✅ AN TOÀN: Khách hàng thuộc nhóm ổn định và trung thành. (Tỷ lệ rủi ro thấp: {probability*100:.1f}%)")
        st.markdown("""
        **💡 KHUYẾN NGHỊ:**
        * Tiếp tục duy trì chính sách chăm sóc định kỳ và các chương trình ưu đãi tích điểm thành viên hiện có.
        """)