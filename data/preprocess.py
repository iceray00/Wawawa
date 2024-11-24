import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(traffic_data, window_size=12):
    scaler = StandardScaler()
    traffic_data_scaled = scaler.fit_transform(traffic_data)

    X, y = [], []
    for i in range(len(traffic_data) - window_size):
        X.append(traffic_data_scaled[i:i + window_size])  # 过去12步的数据作为输入
        y.append(traffic_data_scaled[i + window_size])  # 预测下一步数据
    X = np.array(X)
    y = np.array(y)

    return X, y, scaler

def preprocess_data_with_time(traffic_data, window_size=12, steps_per_day=288):
    """
    对交通流量数据进行预处理，并添加时间特征（如时间步和星期几）。

    参数：
        traffic_data: ndarray
            原始交通流量数据，形状为 (时间步数, 节点数)。
        window_size: int
            用于预测的历史时间步数。
        steps_per_day: int
            每天的时间步数，默认为288（假设数据按5分钟采样一天24小时）。

    返回：
        X: ndarray
            训练集的特征数据，形状为 (样本数, window_size, 节点数 + 时间特征数)。
        y: ndarray
            训练集的目标数据，形状为 (样本数, 节点数)。
        scaler: StandardScaler
            数据标准化的Scaler实例，用于后续逆变换。
    """
    # 数据标准化
    scaler = StandardScaler()
    traffic_data_scaled = scaler.fit_transform(traffic_data)

    # 生成时间特征
    timestamps = np.arange(len(traffic_data))
    time_of_day = (timestamps % steps_per_day) / steps_per_day  # 每天的时间步归一化
    day_of_week = (timestamps // steps_per_day) % 7 / 7  # 一周的天数归一化

    # 将时间特征保持为全局特征，而不是为每个节点重复
    time_features = np.stack([time_of_day, day_of_week], axis=-1)  # 形状为 (时间步数, 2)

    # 将时间特征添加到交通流量数据
    traffic_data_with_time = np.hstack([traffic_data_scaled, time_features])  # 拼接特征，形状为 (时间步数, 207 + 2 = 209)

    # 构建输入特征和目标
    X, y = [], []
    for i in range(len(traffic_data_with_time) - window_size):
        X.append(traffic_data_with_time[i:i + window_size])  # 过去的窗口作为输入
        y.append(traffic_data_scaled[i + window_size])  # 当前时间步的值作为输出
    X = np.array(X)  # 转为 NumPy 数组，形状为 (样本数, window_size, 特征数)
    y = np.array(y)  # 转为 NumPy 数组，形状为 (样本数, 节点数)

    return X, y, scaler
