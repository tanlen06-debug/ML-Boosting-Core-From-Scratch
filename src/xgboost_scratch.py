import numpy as np

def calculate_xgboost_split_gain(G_left, H_left, G_right, H_right, lambda_reg=1.0, gamma=0.0):
    """
    Tính toán độ lợi (Gain) theo công thức đạo hàm bậc 1 (G) và bậc 2 (H) của XGBoost.
    """
    G_total = G_left + G_right
    H_total = H_left + H_right
    
    # Tính điểm (Score) của node cha và node con
    score_parent = (G_total ** 2) / (H_total + lambda_reg)
    score_left = (G_left ** 2) / (H_left + lambda_reg)
    score_right = (G_right ** 2) / (H_right + lambda_reg)
    
    # Độ lợi Gain (Công thức Newton-Raphson)
    gain = 0.5 * (score_left + score_right - score_parent) - gamma
    
    return gain
