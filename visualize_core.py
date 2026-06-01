import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.xgboost_scratch import calculate_xgboost_split_gain
from src.lightgbm_scratch import create_lightgbm_histogram_bins

def visualize_xgboost_lambda_effect():
    """
    Vẽ biểu đồ chứng minh: Khi tăng Lambda (Phạt L2), 
    Độ lợi (Gain) sẽ giảm, giúp XGBoost chống Overfitting.
    """
    # Tạo danh sách các giá trị lambda từ 0 đến 10
    lambdas = np.linspace(0, 10, 50)
    
    # Tính Gain tương ứng với mỗi lambda (G và H giữ nguyên)
    gains = [calculate_xgboost_split_gain(G_left=1.5, H_left=2.0, 
                                          G_right=0.8, H_right=1.2, 
                                          lambda_reg=l) for l in lambdas]

    plt.figure(figsize=(8, 5))
    sns.lineplot(x=lambdas, y=gains, color='red', linewidth=2.5, marker='o', markevery=5)
    
    plt.title('XGBoost: Tác động của Lambda lên Độ lợi (Gain)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Giá trị phạt L2 (Lambda)', fontsize=12)
    plt.ylabel('Độ lợi cắt nhánh (Split Gain)', fontsize=12)
    plt.axhline(0, color='black', linestyle='--', linewidth=1) # Đường ranh giới Gain = 0
    plt.fill_between(lambdas, gains, 0, where=(np.array(gains) > 0), color='red', alpha=0.1)
    
    plt.tight_layout()
    plt.savefig('xgboost_lambda_effect.png', dpi=300)
    print(" Đã lưu biểu đồ XGBoost: xgboost_lambda_effect.png")

def visualize_lightgbm_histogram():
    """
    Vẽ biểu đồ chứng minh: Thuật toán gom 1000 điểm dữ liệu lẻ tẻ
    vào đúng 15 cái thùng (Bins) để tăng tốc tính toán.
    """
    # Giả lập 1000 căn nhà có diện tích phân bố chuẩn (trung bình 60m2, độ lệch 15)
    np.random.seed(42)
    dien_tich = np.random.normal(loc=60, scale=15, size=1000)
    
    # Gọi hàm của nhóm em để ép 1000 căn nhà này vào 15 thùng
    so_thung = 15
    binned_data, bin_edges = create_lightgbm_histogram_bins(dien_tich, max_bins=so_thung)

    plt.figure(figsize=(9, 5))
    
    # Vẽ biểu đồ Histogram đếm số lượng dữ liệu trong từng thùng
    sns.histplot(dien_tich, bins=bin_edges, color='dodgerblue', edgecolor='black', alpha=0.7)
    
    # Đánh dấu các vách ngăn (bin edges)
    for edge in bin_edges:
        plt.axvline(edge, color='red', linestyle=':', linewidth=1, alpha=0.6)

    plt.title(f'LightGBM: Gom 1000 điểm dữ liệu vào {so_thung} Histogram Bins', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Diện tích nhà (m2)', fontsize=12)
    plt.ylabel('Số lượng nhà (Count)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('lightgbm_histogram_bins.png', dpi=300)
    print(" Đã lưu biểu đồ LightGBM: lightgbm_histogram_bins.png")

if __name__ == "__main__":
    print("="*50)
    print(" ĐANG XUẤT BIỂU ĐỒ TRỰC QUAN HÓA THUẬT TOÁN...")
    print("="*50)
    
    # Đảm bảo seaborn hiển thị đẹp
    sns.set_theme(style="whitegrid")
    
    # Chạy 2 hàm vẽ
    visualize_xgboost_lambda_effect()
    visualize_lightgbm_histogram()
    print(" Xong! Hãy kiểm tra các file ảnh .png vừa được tạo ra.")