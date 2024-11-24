import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
    epsilon = 0.01  # 防止除以零
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
