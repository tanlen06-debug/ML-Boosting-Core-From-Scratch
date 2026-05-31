import numpy as np

def create_lightgbm_histogram_bins(feature_array, max_bins=256):
    """
    Chia dữ liệu liên tục thành các thùng (bins) - cốt lõi tăng tốc của LightGBM.
    """
    # Tính toán các điểm phân vị để làm vách ngăn
    percentiles = np.linspace(0, 100, max_bins + 1)
    bin_edges = np.percentile(feature_array, percentiles)
    
    # Gom dữ liệu vào thùng (bin index từ 0 đến max_bins - 1)
    binned_feature = np.digitize(feature_array, bin_edges) - 1
    
    return binned_feature, bin_edges
