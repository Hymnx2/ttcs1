import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

print("=== BACKEND PIPELINE: START TRAINING ===")

# 1. Đọc dữ liệu thô
df = pd.read_csv('Telco-Customer-Churn.csv')

# 2. Làm sạch dữ liệu khuyết thiếu nghiêm ngặt
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# 3. Chuẩn hóa nhãn mục tiêu y (0 và 1)
y = df['Churn'].apply(lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true'] else 0).astype(np.int32)

# 4. Tiền xử lý ma trận đặc trưng X
X_raw = df.drop(['Churn'], axis=1, errors='ignore')

# Loại bỏ các cột định danh dạng chữ như customerID
id_cols = [col for col in X_raw.columns if 'id' in col.lower()]
if id_cols:
    X_raw.drop(columns=id_cols, inplace=True)

# Tách riêng biệt cột số và cột chữ
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_cols = [col for col in X_raw.columns if col not in numerical_cols]

# Mã hóa One-Hot đồng bộ cho các biến phân loại
X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=False)

# Chuẩn hóa các biến định lượng bằng MinMaxScaler
scaler = MinMaxScaler()
X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])

# Dọn dẹp các giá trị lỗi phát sinh ngầm
X_encoded.replace([np.inf, -np.inf], np.nan, inplace=True)
X_encoded.fillna(0, inplace=True)

# Ép ma trận X hoàn toàn sang kiểu float32
X_clean = X_encoded.astype(np.float32)
feature_columns = X_clean.columns.tolist()

# 5. Phân chia dữ liệu và huấn luyện mô hình
X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42)

# Huấn luyện Logistic Regression
lr_model = LogisticRegression(max_iter=2000, solver='liblinear')
lr_model.fit(X_train, y_train)

# Huấn luyện Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print(f"-> Huấn luyện hoàn tất! Số lượng cột đặc trưng chuẩn: {len(feature_columns)}")

# 6. LƯU TRỮ ĐỒNG BỘ TẤT CẢ TÀI SẢN HỌC MÁY
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(feature_columns, 'feature_columns.pkl')

print("=== BACKEND: ĐÃ XUẤT ĐỒNG BỘ CÁC FILE .PKL CHUẨN ===")
