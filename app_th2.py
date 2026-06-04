import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="Dự đoán kết quả học tập", layout="wide")
st.title("📊 ỨNG DỤNG AI DỰ ĐOÁN & PHÂN TÍCH KẾT QUẢ HỌC TẬP (THANG ĐIỂM 10)")

# --- 1. TỰ TẠO DỮ LIỆU MẪU (KHÔNG CẦN ĐỌC FILE CSV NỮA) ---
import numpy as np
np.random.seed(42)
n_samples = 395
# Tạo dữ liệu giả lập chuẩn thang điểm 10 cho G1, G2, G3
g1_mock = np.random.uniform(3.0, 10.0, n_samples)
g2_mock = g1_mock * 0.9 + np.random.normal(0, 0.5, n_samples)
g3_mock = g2_mock * 0.95 + np.random.normal(0, 0.5, n_samples)

g1_mock = np.clip(g1_mock, 0, 10)
g2_mock = np.clip(g2_mock, 0, 10)
g3_mock = np.clip(g3_mock, 0, 10)

data = pd.DataFrame({
    'G1': g1_mock,
    'G2': g2_mock,
    'G3': g3_mock,
    'studytime': np.random.randint(1, 5, n_samples),
    'failures': np.random.randint(0, 4, n_samples),
    'absences': np.random.randint(0, 20, n_samples)
})

# --- 2. HUẤN LUYỆN MÔ HÌNH AI THEO THANG ĐIỂM MỚI ---
X = data[['G1', 'G2', 'studytime', 'failures']]
y = data['G3']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# --- TẠO 2 TAB TRÊN WEB ---
tab1, tab2 = st.tabs(["📈 Phân tích số liệu tổng quan", "🤖 Dự đoán điểm số bằng AI"])

# ================= TAB 1: PHÂN TÍCH SỐ LIỆU TỔNG QUAN =================
with tab1:
    st.header("🔍 Phân tích các yếu tố ảnh hưởng đến học tập")
    
    if st.checkbox("Hiển thị bảng dữ liệu gốc (Đã quy đổi sang thang điểm 10)"):
        st.dataframe(data[['G1', 'G2', 'G3', 'studytime', 'failures']].head(10))
        
    st.subheader("1. Phân bổ số lượng sinh viên theo thời gian học hàng tuần")
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    sns.countplot(x='studytime', data=data, palette="Set2", ax=ax1)
    ax1.set_xlabel("Mức độ thời gian học (1: Ít -> 4: Rất nhiều)")
    ax1.set_ylabel("Số lượng sinh viên")
    st.pyplot(fig1)

    st.subheader("2. Biểu đồ xu hướng tương quan điểm số (Thang điểm 10)")
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    sns.scatterplot(x='G1', y='G3', data=data, alpha=0.7, ax=ax2)
    ax2.set_xlabel("Điểm số kỳ 1 (G1)")
    ax2.set_ylabel("Điểm số cuối kỳ (G3)")
    st.pyplot(fig2)

# ================= TAB 2: GIAO DIỆN NHẬP LIỆU DỰ ĐOÁN =================
with tab2:
    st.header("🔮 Nhập thông số để AI dự đoán điểm cuối kỳ (G3)")
    
    col1, col2 = st.columns(2)
    with col1:
        # Thay đổi max_value thành 10.0 và bước nhảy (step) là 0.5 điểm cho dễ kéo chọn
        g1 = st.slider("Điểm số kỳ 1 (G1) - Thang điểm 10:", 0.0, 10.0, 5.0, 0.5)
        g2 = st.slider("Điểm số kỳ 2 (G2) - Thang điểm 10:", 0.0, 10.0, 5.0, 0.5)
    with col2:
        studytime = st.selectbox("Thời gian học hàng tuần:", options=[1, 2, 3, 4], 
                                 format_func=lambda x: {1: "< 2 tiếng", 2: "2 - 5 tiếng", 3: "5 - 10 tiếng", 4: "> 10 tiếng"}[x])
        failures = st.number_input("Số lần trượt môn trước đây (0 - 4):", min_value=0, max_value=4, value=0)

    # Nút bấm tính toán dự đoán
    if st.button("🚀 BẮT ĐẦU DỰ ĐOÁN"):
        input_data = np.array([[g1, g2, studytime, failures]])
        prediction = model.predict(input_data)[0]
        
        # Giới hạn điểm số từ 0 đến 10
        prediction = max(0.0, min(10.0, prediction))
        
        # Hiển thị kết quả theo thang điểm 10
        st.success(f"🎯 Điểm số cuối kỳ (G3) dự đoán của sinh viên này là: **{prediction:.2f} / 10**")
        
        # Nếu điểm dự đoán từ 8.0 trở lên (Học lực Giỏi) thì thả bóng bay chúc mừng
        if prediction >= 8.0:
            st.balloons()
