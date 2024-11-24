# main.py

import kagglehub
import numpy as np

from data.data_loader import load_adj_matrix, load_traffic_data
from data.preprocess import preprocess_data_with_time
from train import train_and_evaluate, show_hyperparameters

if __name__ == '__main__':
    dataset_path = kagglehub.dataset_download("annnnguyen/metr-la-dataset")
    print("Path to dataset files:", dataset_path)

    adj_matrix_path = f'{dataset_path}/adj_METR-LA.pkl'
    traffic_data_path = f'{dataset_path}/METR-LA.h5'

    adj_matrix = load_adj_matrix(adj_matrix_path)
    traffic_data = load_traffic_data(traffic_data_path)

    # 超参数设置
    window_size = 12      # 使用过去12个时间节点（每个为 5min）的数据进行预测（改动的话需要也一起改动 max_len）
    steps_per_day = 288   # 每天有 288 个时间步（固定的）
    epochs = 2          # 训练轮次（不过设置了早停，一般都到不了100轮）
    d_model = 32          # 特征维度。通常 d_model / num_heads 在 16 至 64 之间，过小会导致注意力不足，过大会导致计算浪费
    num_heads = 2         # 多头注意力机制的头数。d_model 必须能被 num_heads 整除，以确保每个头的维度是一致的
    ff_dim = 32          # Transformer 块中前馈神经网络的隐层维度
    num_layers = 1        # Transformer 编码器中的层数
    max_len = 12          # 模型能够处理的最大序列长度，这个长度应该大于或等于 window_size，即用作输入的时间步数
    save_fig = True      # 是否保存抽取展示的三张图片（第一张、第一百张、最后一张）

    patience_early_stopping = 12   # 早停的严格度
    patience_reduce_lr = 4         # 降低学习率的严格度

    # 数据预处理
    X, y, scaler = preprocess_data_with_time(traffic_data, window_size, steps_per_day)  # 利用带有时间特征信息的数据进行预处理

    # 检查输出形状
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    # 期望 X 的形状为 (34260, 12, 209)
    # 期望 y 的形状为 (34260, 207)

    # 显示超参数
    show_hyperparameters(
        window_size=window_size,
        d_model=d_model,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        max_len=max_len,
        patience_early_stopping=patience_early_stopping,
        patience_reduce_lr=patience_reduce_lr,
        save_fig=save_fig,
    )

    # 切分数据集
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 训练与评估模型
    train_and_evaluate(
        X_train, y_train, X_test, y_test,
        adj_matrix, scaler,
        epochs=epochs,
        d_model=d_model,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        max_len=max_len,
        window_size=window_size,
        patience_early_stopping=patience_early_stopping,
        patience_reduce_lr=patience_reduce_lr,
        save_fig=save_fig
    )
