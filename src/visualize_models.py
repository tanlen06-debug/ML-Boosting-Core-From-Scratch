import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import các thư viện Machine Learning
from sklearn.datasets import fetch_california_housing, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

def run_and_visualize_comparison():
    print("="*70)
    print("🚀 ĐANG TẢI DỮ LIỆU VÀ HUẤN LUYỆN 5 MÔ HÌNH (Bao gồm CatBoost)...")
    print("="*70)
    
    # 1. Chuẩn bị dữ liệu 
    # (Có cơ chế tự động dùng dữ liệu giả lập nếu mạng bị chặn tải California Housing)
    try:
        data = fetch_california_housing()
        X, y = data.data, data.target
    except Exception:
        print("Mạng tải bị chậm, tự động chuyển sang dữ liệu giả lập (20.000 dòng)...")
        X, y = make_regression(n_samples=20000, n_features=10, noise=0.5, random_state=42)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Khởi tạo 5 mô hình để so sánh theo đúng lịch sử tiến hóa
    models = {
        '1. Decision Tree\n(Dễ Overfit)': DecisionTreeRegressor(random_state=42),
        '2. GBM Gốc\n(Chậm)': GradientBoostingRegressor(random_state=42),
        '3. XGBoost\n(Chính xác cao)': xgb.XGBRegressor(random_state=42),
        '4. LightGBM\n(Tốc độ)': lgb.LGBMRegressor(random_state=42),
        '5. CatBoost\n(Categorical)': CatBoostRegressor(random_state=42, verbose=0)
    }
    
    results = {'Model': [], 'RMSE': [], 'Training Time (s)': []}
    
    # 3. Huấn luyện và đo lường
    for name, model in models.items():
        # Lấy tên ngắn gọn để in ra Terminal cho đẹp
        model_short_name = name.split('\n')[0]
        print(f"⏳ Đang chạy {model_short_name}...")
        
        # Đo thời gian
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        
        # Dự đoán và tính sai số
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        # Lưu kết quả
        results['Model'].append(name)
        results['RMSE'].append(rmse)
        results['Training Time (s)'].append(end_time - start_time)
        
    df_results = pd.DataFrame(results)
    print("\n✅ HOÀN THÀNH HUẤN LUYỆN! ĐANG VẼ BIỂU ĐỒ...")
    
    # 4. Trực quan hóa dữ liệu (Vẽ 2 biểu đồ)
    sns.set_theme(style="whitegrid")
    # Kéo rộng khung hình (16, 7) để đủ chỗ cho 5 cột không bị chen lấn
    fig, axes = plt.subplots(1, 2, figsize=(16, 7)) 
    
    # Bảng màu cho 5 cột
    my_palette = ['#ff9999','#ffcc99','#99ff99','#99ccff', '#c299ff']
    
    # Biểu đồ 1: So sánh Sai số (RMSE)
    sns.barplot(x='Model', y='RMSE', data=df_results, ax=axes[0], palette=my_palette)
    axes[0].set_title('So sánh Sai số RMSE (Càng thấp càng tốt)', fontsize=14, fontweight='bold', pad=15)
    axes[0].set_ylabel('Root Mean Squared Error (RMSE)', fontsize=12)
    axes[0].set_xlabel('')
    # Ghi số liệu lên cột
    for i, v in enumerate(df_results['RMSE']):
        axes[0].text(i, v + (df_results['RMSE'].max()*0.02), f"{v:.4f}", ha='center', fontweight='bold')

    # Biểu đồ 2: So sánh Thời gian chạy
    sns.barplot(x='Model', y='Training Time (s)', data=df_results, ax=axes[1], palette=my_palette)
    axes[1].set_title('So sánh Thời gian Huấn luyện (Càng nhanh càng tốt)', fontsize=14, fontweight='bold', pad=15)
    axes[1].set_ylabel('Thời gian (Giây)', fontsize=12)
    axes[1].set_xlabel('')
    # Ghi số liệu lên cột
    for i, v in enumerate(df_results['Training Time (s)']):
        axes[1].text(i, v + (df_results['Training Time (s)'].max()*0.02), f"{v:.4f}s", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('model_comparison_charts.png', dpi=300, bbox_inches='tight')
    print("🎯 Đã lưu biểu đồ thành file: 'model_comparison_charts.png'")

if __name__ == "__main__":
    run_and_visualize_comparison()