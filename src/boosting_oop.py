import numpy as np

class XGBoostNode:
    def __init__(self, lambda_reg=1.0, gamma=0.0):
        self.lambda_reg = lambda_reg
        self.gamma = gamma

    def calculate_gain(self, G_left, H_left, G_right, H_right):
        """Tính Split Gain bằng toán học của XGBoost"""
        G_total, H_total = G_left + G_right, H_left + H_right
        
        score_parent = (G_total ** 2) / (H_total + self.lambda_reg)
        score_left = (G_left ** 2) / (H_left + self.lambda_reg)
        score_right = (G_right ** 2) / (H_right + self.lambda_reg)
        
        gain = 0.5 * (score_left + score_right - score_parent) - self.gamma
        return gain

class LightGBMNode:
    def __init__(self, max_bins=256):
        self.max_bins = max_bins
        self.bin_edges = None

    def create_histogram(self, feature_array):
        """Ép dữ liệu vào Histogram Bins của LightGBM"""
        percentiles = np.linspace(0, 100, self.max_bins + 1)
        self.bin_edges = np.percentile(feature_array, percentiles)
        
        binned_feature = np.digitize(feature_array, self.bin_edges) - 1
        return binned_feature, self.bin_edges

# --- ĐOẠN TEST CHẠY TRỰC TIẾP ---
if __name__ == "__main__":
    xgb = XGBoostNode(lambda_reg=1.5, gamma=0.1)
    print("Gain XGBoost (OOP):", xgb.calculate_gain(1.5, 2.0, 0.8, 1.2))

    lgb = LightGBMNode(max_bins=4)
    data = np.array([30.5, 45.2, 55.0, 120.4, 150.0])
    print("LightGBM Bins (OOP):", lgb.create_histogram(data)[0])