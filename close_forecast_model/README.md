# 📈 Dự Báo Giá Cổ Phiếu với Bi-LSTM (Vinamilk - VNM)

Dự án này sử dụng mạng thần kinh nhân tạo hồi quy **Bidirectional LSTM (Bi-LSTM)** để dự báo giá đóng cửa của cổ phiếu Vinamilk (VNM) và đánh giá khả năng tổng quát hóa của mô hình trên các thị trường chứng khoán khác (VN30, US Stock, Global Indices).



## 📁 Cấu trúc thư mục
```text
.
├── vnm-bilstm-forecast.ipynb      # Notebook chứa toàn bộ pipeline xử lý và huấn luyện
└── result/                        # Thư mục chứa kết quả đầu ra
    ├── __reults___files/          # Tất cả các hình ảnh output
    ├── best_bilstm_vnm.keras      # File lưu trữ trọng số mô hình tốt nhất (Epoch 15)
    ├── daily_error_chart.png      # Biểu đồ RMSE & MAPE theo từng ngày dự báo
    ├── loss_curve.png             # Biểu đồ Training vs Validation Loss
    ├── vnm_forecast_plot.png      # Biểu đồ so sánh giá thực tế vs dự báo trên tập Test
    └── evaluation_results.csv     # (Tùy chọn) Bảng thống kê MAPE cho VN30 và Quốc tế
```

## 🚀 Đặc điểm kỹ thuật
- **Mô hình:** Bidirectional LSTM (Bi-LSTM) với 2 lớp ẩn (64 units mỗi lớp).
- **Kỹ thuật chống Overfitting:** Dropout (0.4) và Early Stopping (Patience = 20).
- **Dữ liệu đầu vào:** Cửa sổ trượt (Sliding Window) nhìn về quá khứ 60 ngày (**TIME_STEPS = 60**).
- **Đầu ra dự báo:** Dự báo chuỗi liên tục cho 14 ngày kế tiếp (**FORECAST_HORIZON = 14**).
- **Optimizer:** Adam với cơ chế lưu lại trọng số tốt nhất (ModelCheckpoint).



## 📊 Kết quả đạt được

### 1. Hiệu năng trên tập Test của VNM (Sân nhà)
Mô hình đạt được độ chính xác rất cao khi dự báo chính cổ phiếu đã được huấn luyện:
- **MAPE trung bình (14 ngày):** ~3.30% - 4.34%
- **Ngày 1 (D1):** Sai số chỉ ~1.78%
- **Ngày 14 (D14):** Sai số duy trì ở mức ~4.34%

### 2. Khả năng tổng quát hóa (Generalization)
Mô hình được dùng để test "mù" trên các thị trường khác mà không qua huấn luyện lại:
- **VN30:** Hoạt động tốt trên các mã ổn định như CTG (2.86%), POW (3.21%). Gặp khó khăn với các mã biến động mạnh như NVL.
- **Thị trường Quốc tế:** MAPE trung bình cho US Stock (S&P 500, Apple...) đạt khoảng 14-16%.

## 🛠 Hướng dẫn sử dụng
1. Cài đặt các thư viện cần thiết:
   ```bash
   pip install tensorflow pandas numpy matplotlib scikit-learn yfinance
   ```
2. Mở file `vnm-bilstm-forecast.ipynb` bằng Jupyter Notebook hoặc Google Colab.
3. Chạy toàn bộ các Cell để thực hiện từ bước tải dữ liệu đến dự báo.
4. Kết quả hình ảnh và model sẽ tự động được lưu vào thư mục `/result`.

## 📝 Kết luận
Mô hình Bi-LSTM cho thấy sức mạnh trong việc nắm bắt xu hướng ngắn và trung hạn (dưới 14 ngày). Với kỹ thuật **Dropout 0.4**, mô hình đã hạn chế tối đa hiện tượng "học vẹt", đảm bảo tính khách quan khi dự báo các thị trường mới.
