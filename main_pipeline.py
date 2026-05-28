import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

print("=========================================================")
print("🚀 KHỞI CHẠY TIẾN TRÌNH HUẤN LUYỆN MÔ HÌNH CHURN PREDICTION")
print("=========================================================\n")

# ==========================================
# ==========================================
# BƯỚC 1: TẢI VÀ KHÁM PHÁ DỮ LIỆU NHANH (EDA)
# ==========================================
print("[1/5] Đang tải bộ dữ liệu...")
df = pd.read_csv('Telco-Customer-Churn.csv') 
print(f"-> Kích thước dữ liệu thô: {df.shape[0]} dòng, {df.shape[1]} cột.\n")

# Kiểm tra tỷ lệ mất cân bằng nhãn mục tiêu (Sửa đoạn này để tránh KeyError)
churn_rate = df['Churn'].value_counts(normalize=True) * 100
print(f"-> Tỷ lệ phân bổ nhãn mục tiêu Churn:")

# Tự động kiểm tra xem nhãn đang là dạng chữ (Yes/No) hay dạng số (1/0)
if 'No' in churn_rate:
    print(f"   - No (Ở lại): {churn_rate['No']:.1f}%")
    print(f"   - Yes (Rời mạng): {churn_rate['Yes']:.1f}%\n")
elif 0 in churn_rate:
    print(f"   - 0 (Ở lại): {churn_rate[0]:.1f}%")
    print(f"   - 1 (Rời mạng): {churn_rate[1]:.1f}%\n")
else:
    # Trường hợp nhãn mang tên khác, in ra toàn bộ để kiểm tra
    print(churn_rate)


# ==========================================
# BƯỚC 2: LÀM SẠCH DỮ LIỆU (DATA CLEANING)
# ==========================================
print("[2/5] Đang tiến hành làm sạch dữ liệu...")

# Ép kiểu dữ liệu cột TotalCharges về dạng số, chuyển khoảng trắng lỗi thành NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Thống kê số lượng dòng bị khuyết thiếu
missing_count = df['TotalCharges'].isnull().sum()
print(f"-> Phát hiện {missing_count} dòng bị khuyết thiếu tại cột TotalCharges.")

# Loại bỏ các dòng khuyết thiếu (11 dòng tương ứng khách hàng mới tenure = 0)
df.dropna(subset=['TotalCharges'], inplace=True)
print(f"-> Đã loại bỏ các dòng lỗi. Kích thước dữ liệu hiện tại: {df.shape[0]} dòng.\n")


# ==========================================
# BƯỚC 3: KỸ NGHỆ ĐẶC TRƯNG (FEATURE ENGINEERING)
# ==========================================
print("[3/5] Đang thực hiện kỹ nghệ đặc trưng và chuẩn hóa...")

# Tách biến mục tiêu Churn và loại bỏ mã định danh định tính không liên quan
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0) # Chuyển nhãn mục tiêu thành 0 và 1
X_raw = df.drop(['customerID', 'Churn'], axis=1)

# A. Mã hóa các biến danh mục nhị phân (Label Encoding)
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
le = LabelEncoder()
for col in binary_cols:
    X_raw[col] = le.fit_transform(X_raw[col])

# B. Mã hóa các biến danh mục nhiều nhãn (One-Hot Encoding)
categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                    'Contract', 'PaymentMethod']
X_encoded = pd.get_dummies(X_raw, columns=categorical_cols)

# C. Chuẩn hóa các biến số liên tục có giá trị lớn bằng MinMaxScaler
scaler = MinMaxScaler()
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])

feature_columns = X_encoded.columns.tolist()
print(f"-> Hoàn thành mã hóa. Tổng số đặc trưng đầu vào sau One-Hot: {len(feature_columns)} cột.\n")


# ==========================================
# BƯỚC 4: THIẾT LẬP THỰC NGHIỆM VÀ HUẤN LUYỆN
# ==========================================
print("[4/5] Đang phân chia dữ liệu và huấn luyện mô hình...")

# Phân chia dữ liệu theo tỷ lệ nghiêm ngặt 80% Train và 20% Test
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Khởi tạo và huấn luyện mô hình cơ sở (Logistic Regression)
lr_model = LogisticRegression(max_iter=1000, solver='liblinear')
lr_model.fit(X_train, y_train)

# Khởi tạo và huấn luyện mô hình cây cải tiến (Random Forest)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print("-> Huấn luyện thành công các mô hình thực nghiệm.\n")


# ==========================================
# BƯỚC 5: ĐÁNH GIÁ, SO SÁNH VÀ LƯU TRỮ
# ==========================================
print("[5/5] Đang đánh giá hiệu năng hệ thống...")

# Dự đoán kiểm thử trên cả 2 mô hình
y_pred_lr = lr_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Trích xuất các chỉ số chính
acc_lr = accuracy_score(y_test, y_pred_lr) * 100
f1_lr = f1_score(y_test, y_pred_lr)

acc_rf = accuracy_score(y_test, y_pred_rf) * 100
f1_rf = f1_score(y_test, y_pred_rf)

print("\n=========================================================")
print("📊 KẾT QUẢ ĐÁNH GIÁ HIỆU NĂNG MÔ HÌNH TRÊN TẬP KIỂM THỬ")
print("=========================================================")
print(f"1. Mô hình Logistic Regression (Baseline):")
print(f"   - Độ chính xác (Accuracy): {acc_lr:.1f}%")
print(f"   - F1-Score:                {f1_lr:.2f}")

print(f"\n2. Mô hình Random Forest (Cải tiến):")
print(f"   - Độ chính xác (Accuracy): {acc_rf:.1f}%")
print(f"   - F1-Score:                {f1_rf:.2f}")
print("=========================================================\n")

print("-> Chi tiết ma trận nhầm lẫn của Random Forest:")
print(confusion_matrix(y_test, y_pred_rf))

print("\n-> Đang tiến hành đóng gói và lưu trữ mô hình tối ưu...")
# Lưu trữ các tệp tin cấu hình cốt lõi phục vụ cho Demo Web App
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(feature_columns, 'feature_columns.pkl')

print("🎉 THÀNH CÔNG: Đã tạo ra các file rf_model.pkl, scaler.pkl, feature_columns.pkl!")
print("Hệ thống đã sẵn sàng kết nối tới giao diện Demo Streamlit.")
print("=========================================================")
