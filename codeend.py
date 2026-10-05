import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
# Thêm các thư viện chia tập dữ liệu & đánh giá chỉ số theo yêu cầu
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 1. Đọc file CSV từ Kaggle
df = pd.read_csv('filemaudulieu.csv')  # Thay tên file CSV thực tế của bạn vào đây

# =========================================================================
# 2. CHIA CỘT RÕ RÀNG (KHỚP 100% VỚI 26 CỘT CỦA BẠN)
# =========================================================================

# Nhóm 1: Các cột dạng SỐ (Chạy qua StandardScaler) - 13 cột
features_numeric = [
    'amount_usd',               # Số tiền giao dịch
    'hours_since_last_txn',     # Số giờ từ giao dịch trước
    'txn_count_last_24h',       # Số giao dịch trong 24h qua
    'distance_from_home_km',    # Khoảng cách từ nhà (km)
    'card_age_months',          # Tuổi đời của thẻ (tháng)
    'customer_age',             # Tuổi khách hàng
    'account_balance_usd',      # Số dư tài khoản
    'cvv_retry_count',          # Số lần gõ sai mã CVV
    'velocity_score',           # Điểm tốc độ giao dịch
    'time_of_day_hour',         # Giờ trong ngày (0 - 23)
    'day_of_week',              # Ngày trong tuần (0 - 6)
    'merchant_risk_score',      # Điểm rủi ro của cửa hàng
    'prior_disputes'            # Số lần tranh chấp/khiếu nại trước đó
]

# Nhóm 2: Các cột dạng CHỮ hoặc ĐÚNG/SAI (Chạy qua OneHotEncoder) - 11 cột
features_categorical = [
    'merchant_category',        # Ngành hàng cửa hàng
    'card_type',                # Loại thẻ
    'auth_method',              # Phương thức xác thực
    'channel',                  # Kênh giao dịch
    'device_type',              # Loại thiết bị
    'is_foreign_transaction',   # Giao dịch nước ngoài (True/False)
    'is_new_merchant',          # Cửa hàng mới giao dịch lần đầu
    'used_vpn',                 # Có dùng VPN hay không
    'ip_country_mismatch',      # IP khác quốc gia đăng ký
    'billing_shipping_mismatch',# Địa chỉ thanh toán khác địa chỉ giao
    'is_ai_generated_scam_attempt' # Dấu hiệu lừa đảo bằng AI
]

all_features = features_numeric + features_categorical

# =========================================================================
# 3. TIỀN XỬ LÝ & CHUẨN HÓA DỮ LIỆU
# =========================================================================

# Điền ô trống (Missing Values) để tránh lỗi crash code
for col in features_numeric:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

for col in features_categorical:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

# Dùng ColumnTransformer để biến đổi dữ liệu
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features_numeric),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_categorical)
    ]
)

# Chuyển đổi toàn bộ 24 cột tính năng thành mảng số X_encoded
X_encoded = preprocessor.fit_transform(df[all_features])

# =========================================================================
# 4. HUẤN LUYỆN MÔ HÌNH ISOLATION FOREST & XUẤT FILE CSV (GIỮ NGUYÊN CODE CỦA ĐỒNG CHÍ)
# =========================================================================

model = IsolationForest(
    contamination=0.01,  # Dự đoán 1.5% giao dịch bất thường
    n_estimators=100,     # Khởi tạo 100 cây cô lập
    random_state=42,      # Cố định ngẫu nhiên
    n_jobs=-1             # Tận dụng tối đa nhân CPU
)
model.fit(X_encoded)

# Gán nhãn kết quả dự đoán (-1 là Bất thường -> True, 1 là Bình thường -> False)
preds = model.predict(X_encoded)
df['is_anomaly'] = np.where(preds == -1, True, False)

# In ra và xuất file CSV
anomalies = df[df['is_anomaly'] == True]
output_filename = 'danh_sach_bat_thuong_day_du.csv'
anomalies.to_csv(output_filename, index=False, encoding='utf-8-sig')

print("="*60)
print(f"Tổng số giao dịch phân tích: {len(df)}")
print(f"Phát hiện: {len(anomalies)} giao dịch bất thường.")
print(f"-> Đã xuất ra file thành công: '{output_filename}'!")
print(f"Phát hiện {len(anomalies)} giao dịch nghi ngờ bất thường trên tổng số {len(df)} giao dịch:")
print(anomalies[['amount_usd', 'card_type', 'channel', 'device_type', 'is_anomaly']].head(10))
print("="*60)

# =========================================================================
# 5. THÊM PHẦN ĐÁNH GIÁ MÔ HÌNH (BỔ SUNG CHO ĐÚNG YÊU CẦU ĐỀ BÀI CHẤM ĐIỂM)
# =========================================================================

# Đổi cột nhãn thực tế 'is_fraud' về dạng nhị phân 1/0 nếu đang ở dạng True/False hoặc 1/0
y_true = df['is_fraud'].astype(int)

# Chia dữ liệu theo tỷ lệ 80% Train, 20% Test
X_train, X_test, y_train, y_test = train_test_split(df[all_features], y_true, test_size=0.2, random_state=42)

# Fit preprocessor trên tập Train và transform tập Test
X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc = preprocessor.transform(X_test)

# Fit lại mô hình trên tập Train và dự đoán tập Test
model_eval = IsolationForest(contamination=0.01, n_estimators=100, random_state=42, n_jobs=-1)
model_eval.fit(X_train_enc)

preds_test = model_eval.predict(X_test_enc)
# Quy đổi dự đoán của Isolation Forest (-1: Bất thường -> 1, 1: Bình thường -> 0)
y_pred_test = np.where(preds_test == -1, 1, 0)

# In kết quả đánh giá thực nghiệm lên Terminal
print("\n=== CÁC CHỈ SỐ ĐÁNH GIÁ MÔ HÌNH (EVALUATION METRICS) ===")
print(f"1. Accuracy  (Độ chính xác):  {accuracy_score(y_test, y_pred_test):.4f}")
print(f"2. Precision (Độ chuẩn xác): {precision_score(y_test, y_pred_test, zero_division=0):.4f}")
print(f"3. Recall    (Độ nhạy):       {recall_score(y_test, y_pred_test, zero_division=0):.4f}")
print(f"4. F1-Score  (Điểm F1):       {f1_score(y_test, y_pred_test, zero_division=0):.4f}")
print("\n--- CHI TIẾT BÁO CÁO PHÂN LOẠI (CLASSIFICATION REPORT) ---")
print(classification_report(y_test, y_pred_test, target_names=['Bình thường (0)', 'Bất thường (1)'], zero_division=0))