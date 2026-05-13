# 📈 Dự Báo Giá Cổ Phiếu với Bi-LSTM — VN30 (VNM · FPT · HPG · VIC)

Dự án sử dụng mạng **Bidirectional LSTM (Bi-LSTM)** để dự báo giá đóng cửa cho 4 mã VN30. Có 2 notebook tương ứng với 2 giai đoạn phát triển — bản gốc tập trung vào VNM (Vinamilk), bản mở rộng huấn luyện riêng cho 4 mã.

---

## 📓 Hai notebook trong project

| File | Mục đích | Khi nào dùng |
|---|---|---|
| [`vnm-bilstm-forecast.ipynb`](vnm-bilstm-forecast.ipynb) | **V1** — Train + dự báo riêng cho VNM, đồng thời test "mù" trên VN30 / US Stock / Global Indices để khảo sát generalization. | Đọc để hiểu pipeline gốc và phần đánh giá khả năng tổng quát hóa. |
| [`multi-stock-bilstm-forecast.ipynb`](multi-stock-bilstm-forecast.ipynb) | **V2** — Train 4 mô hình **riêng biệt** cho VNM, FPT, HPG, VIC từ data trong [`DataDAV/`](DataDAV/), kèm cell forecast 14 ngày tương lai + cell đóng gói zip. | Đọc để thấy pipeline đa-ticker, bảng so sánh 4 mã, và file kết quả mới nhất. |

> **Khác biệt cốt lõi:** V1 = 1 model "đa năng" chạy thử trên nhiều mã. V2 = 4 model "chuyên biệt", mỗi mã 1 scaler + 1 model riêng → kết quả chính xác hơn nhưng không so sánh khả năng generalization được.

---

## 📁 Cấu trúc thư mục

```text
close_forecast_model/
├── vnm-bilstm-forecast.ipynb            # V1 — Notebook gốc (VNM-centric)
├── multi-stock-bilstm-forecast.ipynb    # V2 — Notebook đa-ticker (4 mã)
│
├── DataDAV/                             # Dataset dùng cho V2
│   ├── VNM.csv  FPT.csv  HPG.csv  VIC.csv   # 4 mã × 1286 dòng (2020-11 → 2025)
│   └── dataset-metadata.json                # Metadata để re-upload Kaggle
│
├── results/                             # Output của V1 (VNM-only)
│   ├── best_bilstm_vnm.keras
│   ├── loss_curve.png · forecast_chart.png · daily_error_chart_fixed.png
│   └── __results___files/
│
└── multi_stock_forecast_results/        # Output của V2 (4 ticker)
    ├── VNM/  FPT/  HPG/  VIC/           # Mỗi ticker: model + 3 chart
    ├── summary.csv                      # Bảng tổng kết RMSE/MAPE 4 mã
    ├── future_forecast.csv              # Dự báo 14 ngày tới (4 mã)
    ├── comparison_4tickers.png          # RMSE & MAPE so sánh 4 mã
    └── future_forecast_4tickers.png     # Chart 2×2 forecast tương lai
```

---

## 🚀 Đặc điểm kỹ thuật (chung cho cả 2 notebook)

| Thông số | Giá trị |
|---|---|
| Kiến trúc | Bidirectional LSTM × 2 lớp, 64 units mỗi lớp |
| Chống overfit | Dropout 0.4 + EarlyStopping (patience = 20) |
| Cửa sổ đầu vào | `TIME_STEPS = 60` ngày |
| Horizon dự báo | `FORECAST_HORIZON = 14` ngày |
| Features | `[open, high, low, close, volume]` (5 cột OHLCV) |
| Optimizer | Adam + ModelCheckpoint (lưu best epoch theo `val_loss`) |
| Train/Test split | 80/20 theo thứ tự thời gian (no shuffle) |

---

## 📊 Kết quả V2 — 4 mã VN30

Train 80% / Test 20%, đánh giá trên 14-day multi-step forecast:

| Ticker | Best Epoch | RMSE (VNĐ) | MAPE | Đánh giá |
|---|---|---|---|---|
| **VNM** | 5/25  | 2.49  | **3.56%** | 🟢 Tốt |
| **HPG** | 7/27  | 1.28  | **4.12%** | 🟢 Tốt |
| **FPT** | 15/35 | 9.06  | **6.61%** | 🟡 Chấp nhận được |
| **VIC** | 11/31 | 16.10 | **15.04%** | 🔴 Có vấn đề — xem caveat dưới |

### ⚠️ Caveat về VIC

VIC MAPE 15% không phải lỗi pipeline. Nguyên nhân:

1. **Overfit nặng**: train loss ~0.005 vs val loss ~0.15 (gap 30×) → đặc trưng VIC có rất nhiều spike, model không generalize được.
2. **Giá vượt range training**: giai đoạn test 2024-2025 VIC có spike lên ~160 VNĐ trong khi train chỉ thấy 20-100 VNĐ. Scaler bóp giá vượt range → predict bị lệch.
3. Hệ quả: forecast 14 ngày tới của VIC (-36.56%) **không nên dùng làm investment signal**.

Hướng cải thiện (nếu cần ngoài scope mid-term): dùng log-return thay vì raw price, hoặc per-window normalization thay vì global MinMax.

---

## 📊 Kết quả V1 — VNM "sân nhà" + generalization

### Hiệu năng trên VNM (mã đã train)
- MAPE trung bình 14 ngày: **~3.30 – 4.34%**
- Day 1: ~1.78% | Day 14: ~4.34%

### Test "mù" trên thị trường khác (không train lại)
- **VN30**: tốt cho mã ổn định (CTG 2.86%, POW 3.21%), kém cho mã biến động (NVL).
- **US Stock (S&P 500, Apple, ...)**: MAPE ~14-16%.

---

## 🛠 Hướng dẫn sử dụng

### Cài đặt
```bash
pip install tensorflow pandas numpy matplotlib scikit-learn
```

### Chạy V2 (khuyến nghị cho đề tài 4 mã)
1. **Trên Kaggle**: notebook đã được thiết kế cho Kaggle.
   - Attach dataset `phvngtngtm/vn30-4tickers-datadav` (hoặc tự upload [`DataDAV/`](DataDAV/) làm dataset mới)
   - Sửa `DATASET_DIR` trong CONFIG cell nếu slug khác
   - Run all → output tự lưu vào `results/` rồi đóng gói thành `multi_stock_forecast_results.zip`
2. **Local**: cần GPU (CPU sẽ rất chậm vì train 4 model). Đổi path `TICKERS` dict trỏ vào [`DataDAV/`](DataDAV/) local.

### Chạy V1 (chỉ cho VNM + khảo sát generalization)
Mở [`vnm-bilstm-forecast.ipynb`](vnm-bilstm-forecast.ipynb) → run all. Cần dataset Kaggle `phvngtngtm/vnm-2225`.

---

## 📝 Kết luận

- Bi-LSTM 2 lớp với cửa sổ 60-ngày dự báo 14-ngày hoạt động tốt cho mã có price-stationary tốt (VNM, HPG, FPT — MAPE 3-7%).
- Đối với mã có spike & non-stationary mạnh (VIC), pipeline raw-price + global MinMax không đủ — cần feature engineering bổ sung.
- V2 cung cấp 4 mô hình chuyên biệt + bảng so sánh + dự báo tương lai, đủ scope mid-term project.
