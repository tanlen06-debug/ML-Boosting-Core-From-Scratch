class TreeNode:
    def __init__(self):
        self.left = None
        self.right = None
        self.split_feature = None
        self.split_value = None
        self.leaf_value = None

# Trong thực tế, XGBoost và LightGBM sẽ gọi hàm từ xgboost_scratch và lightgbm_scratch 
# để quyết định cách phân nhánh (split) cho các TreeNode này.
