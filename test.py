import numpy as np
from src.xgboost_scratch import calculate_xgboost_split_gain
from src.lightgbm_scratch import create_lightgbm_histogram_bins

if __name__ == "__main__":
    print("="*60)
    print("🚀 BẮT ĐẦU TEST LÕI THUẬT TOÁN: XGBOOST & LIGHTGBM 🚀")
    print("="*60)

    # --- Đóng vai Tấn Lên ---
    print("\n[1] Hàm XGBoost Gain (Newton-Raphson):")
    gain = calculate_xgboost_split_gain(G_left=1.5, H_left=2.0, G_right=0.8, H_right=1.2)
    print(f" -> Độ lợi Gain tính được: {gain:.4f}")

    # --- Đóng vai Nhựt Duy ---
    print("\n[2] Hàm LightGBM Histogram (Binning):")
    dummy_features = np.array([30.5, 45.2, 55.0, 120.4, 150.0, 45.2, 80.5])
    binned_data, edges = create_lightgbm_histogram_bins(dummy_features, max_bins=4)
    print(f" -> Dữ liệu gốc: {dummy_features}")
    print(f" -> Dữ liệu sau khi gom vào 4 thùng: {binned_data}")
    print("="*60)
