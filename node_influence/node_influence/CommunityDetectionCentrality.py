"""
社区检测和节点中心性 (Community Detection and Centrality)

@File    :   CommunityDetectionCentrality.py
@Time    :   2024/12/17 14:08:41
@Author  :   xinxin & iceray
@Version :   1.0
"""


import json
import time

import h5py
import pickle
import numpy as np
import kagglehub
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Dropout, MultiHeadAttention, LayerNormalization, Add, Conv1D, \
    MaxPooling1D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Layer, BatchNormalization, MultiHeadAttention
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
import tensorflow_addons as tfa
from tensorflow.keras.initializers import TruncatedNormal
import networkx as nx
from community import community_louvain


# 加载邻接矩阵数据
def load_adj_matrix(file_path):
    with open(file_path, 'rb') as f:
        adj_data = pickle.load(f, encoding='latin1')
    adj_matrix = adj_data[2]  # 邻接矩阵是一个207x207的numpy矩阵
    return adj_matrix


# 加载交通流量数据
def load_traffic_data(file_path):
    with h5py.File(file_path, 'r') as f:
        data = f['df']
        traffic_data = data['block0_values'][:]
        return traffic_data  # 形状为 (34272, 207)


# 数据预处理
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


# 使用Louvain算法进行社区检测，并计算每个节点的影响力（影响力基于社区大小）
def detect_community(adj_matrix):
    # 将邻接矩阵转换为NetworkX图
    G = nx.from_numpy_array(adj_matrix)

    # 使用Louvain算法进行社区检测
    partition = community_louvain.best_partition(G)

    # 计算每个社区的大小
    community_sizes = {}
    for node, community in partition.items():
        if community not in community_sizes:
            community_sizes[community] = 0
        community_sizes[community] += 1

    # 计算每个节点的影响力（假设影响力与社区大小成正比）
    node_influence = {node: community_sizes[community] for node, community in partition.items()}

    return node_influence


def adjust_adjacency_matrix(adj_matrix, node_influence):
    """
    基于节点影响力调整邻接矩阵的权重。
    :param adj_matrix: 原始邻接矩阵，形状为 (节点数, 节点数)
    :param node_influence: 节点影响力（字典类型），形状为 (节点数,)
    :return: 加权后的邻接矩阵
    """
    # 将节点影响力从字典转换为列表
    node_influence_values = np.array([node_influence[node] for node in sorted(node_influence.keys())])

    # 对节点影响力进行归一化
    influence_norm = node_influence_values / np.max(node_influence_values)

    # 创建加权的邻接矩阵，基于节点影响力调整
    weighted_adj_matrix = adj_matrix * influence_norm[:, np.newaxis]  # 对每列进行缩放

    return weighted_adj_matrix


# /* 将这部分逻辑包装为一个自定义 Keras 层（Layer 类），这样可以在模型构建流程中正确处理符号张量 */
class MultiScaleGraphConvolution(Layer):
    def __init__(self, adj_matrix, d_model, dropout_rate=0.1, num_scales=3):
        super(MultiScaleGraphConvolution, self).__init__()
        self.num_scales = num_scales
        self.d_model = d_model
        self.dropout_rate = dropout_rate

        # 将adj_matrix作为初始值创建一个可训练的参数矩阵W_adj
        self.adj_matrix_shape = adj_matrix.shape
        self.W_adj = self.add_weight(
            shape=self.adj_matrix_shape,
            initializer=tf.keras.initializers.Constant(adj_matrix),
            trainable=True,
            name="dynamic_adj_matrix"
        )

        self.dense_list = [Dense(d_model, kernel_regularizer=tf.keras.regularizers.l2(1e-4))
                           for _ in range(num_scales)]
        self.dropout_list = [Dropout(dropout_rate) for _ in range(num_scales)]
        self.concat_dense = Dense(d_model)

    def call(self, inputs):
        traffic_data = inputs[:, :, :207]  # [batch, window_size, 207]
        time_features = inputs[:, :, 207:]  # [batch, window_size, 2]

        # 计算多尺度的图卷积输出
        # scale 1: A
        # scale 2: A^2
        # scale 3: A^3
        # ...
        # 可以根据需要预先计算A^2, A^3...或者在这里动态计算
        # 为了简化，这里动态计算A^k = A^(k-1)*A，但实际中会在build中缓存提高效率。

        # 由于W_adj是可训练的，我们在每次call中需重新计算A^k
        A = self.W_adj
        outputs = []
        A_power = A
        for scale in range(1, self.num_scales + 1):
            # 图卷积输出
            graph_output = tf.matmul(traffic_data, A_power)  # [batch, window, 207]
            graph_output = self.dense_list[scale - 1](graph_output)
            graph_output = self.dropout_list[scale - 1](graph_output)
            outputs.append(graph_output)

            # 计算下一阶的A^k
            A_power = tf.matmul(A_power, A)

        # 将不同尺度的输出在特征维度拼接: [batch, window, num_scales*d_model]
        multi_scale_output = tf.concat(outputs, axis=-1)

        # 拼接时间特征 (输出维度为 [batch, window, num_scales*d_model + 2])
        combined_output = tf.concat([multi_scale_output, time_features], axis=-1)

        # 将拼接后的多尺度特征映射回 d_model 维度
        combined_output = self.concat_dense(combined_output)  # [batch, window, d_model]
        return combined_output


# 可学习的位置编码
class LearnablePositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model):
        super(LearnablePositionalEncoding, self).__init__()
        self.max_len = max_len
        self.d_model = d_model

        # 可学习的位置编码
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(max_len, d_model),
            initializer=TruncatedNormal(mean=0.0, stddev=0.02),
            trainable=True
        )

        # 固定的位置编码（正弦-余弦）
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        sinusoidal_encoding = np.zeros((max_len, d_model))
        sinusoidal_encoding[:, 0::2] = np.sin(position * div_term)
        sinusoidal_encoding[:, 1::2] = np.cos(position * div_term)
        self.fixed_pos_encoding = tf.constant(sinusoidal_encoding, dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_embedding[:seq_len] + self.fixed_pos_encoding[:seq_len]


def get_optimizer_and_schedule():
    initial_learning_rate = 1e-4
    first_decay_steps = 1000

    # 余弦退火学习率
    lr_schedule = CosineDecayRestarts(
        initial_learning_rate,
        first_decay_steps,
        t_mul=2.0,  # 每次重启后周期乘数
        m_mul=0.9,  # 每次重启后学习率乘数
        alpha=0.01  # 最小学习率比例
    )

    # AdamW优化器配置
    optimizer = AdamW(
        learning_rate=lr_schedule,
        weight_decay=0.01,  # L2正则化系数
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )

    return optimizer


# 构建时空 Transformer 模型
def build_transformer_model(input_shape, adj_matrix, d_model=64, num_heads=4, ff_dim=128, num_layers=3, max_len=12,
                            num_scales=3):
    inputs = Input(shape=input_shape)

    # 使用自定义层提取空间信息 (多尺度+动态邻接)
    x = MultiScaleGraphConvolution(adj_matrix, d_model, dropout_rate=0.01, num_scales=num_scales)(inputs)

    # 可学习的位置编码
    position_encoding = LearnablePositionalEncoding(max_len, d_model)(x)

    # 多层 Transformer 编码
    for _ in range(num_layers):
        x = transformer_block(position_encoding, d_model, num_heads, ff_dim)

    outputs = Dense(207)(x[:, -1, :])
    model = Model(inputs, outputs)
    model.compile(optimizer=get_optimizer_and_schedule(), loss=tf.keras.losses.Huber(delta=1.0),
                  metrics=['mae', 'mape'])
    return model


class R2Callback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val, scaler):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.scaler = scaler

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_val, verbose=0)
        y_pred_inverse = self.scaler.inverse_transform(y_pred)
        y_val_inverse = self.scaler.inverse_transform(self.y_val)

        # 计算 R² 分数
        r2 = r2_score(y_val_inverse, y_pred_inverse)
        print(f"Epoch {epoch + 1}: R² = {r2:.4f}")


# /* 3 */ 实现因果掩码模型
# 更新后的因果掩码函数，不需要 num_heads 参数
def causal_mask_def(batch_size, seq_len):
    mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)  # 下三角矩阵 [seq_len, seq_len]
    mask = tf.expand_dims(mask, axis=0)  # [1, seq_len, seq_len]
    mask = tf.tile(mask, [batch_size, 1, 1])  # [batch_size, seq_len, seq_len]
    return mask


class CausalMultiHeadAttention(MultiHeadAttention):
    def __init__(self, num_heads, key_dim, **kwargs):
        super(CausalMultiHeadAttention, self).__init__(num_heads=num_heads, key_dim=key_dim, **kwargs)
        self.num_heads = num_heads

    def call(self, query, value, key=None, attention_mask=None, training=None, **kwargs):
        batch_size = tf.shape(query)[0]
        seq_len = tf.shape(query)[1]

        # 仅在动态图模式下打印
        if tf.executing_eagerly():
            tf.print("Query shape:", tf.shape(query))
            tf.print("Value shape:", tf.shape(value))

        causal_mask = causal_mask_def(batch_size, seq_len)
        if attention_mask is not None:
            attention_mask = tf.minimum(attention_mask, causal_mask)
        else:
            attention_mask = causal_mask

        attention_mask = tf.expand_dims(attention_mask, axis=1)
        attention_mask = tf.tile(attention_mask, [1, self.num_heads, 1, 1])

        return super().call(query=query, value=value, key=key, attention_mask=attention_mask, training=training,
                            **kwargs)


# 构建 Transformer 块
def transformer_block(inputs, d_model, num_heads, ff_dim, dropout_rate=0.1):
    attn_layer = CausalMultiHeadAttention(num_heads=num_heads, key_dim=d_model)
    attn_output = attn_layer(inputs, inputs)
    attn_output = Dropout(dropout_rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(Add()([inputs, attn_output]))  # 残差连接

    # 添加批归一化
    ff_output = Dense(ff_dim, activation='relu')(out1)
    ff_output = BatchNormalization()(ff_output)  # 批归一化
    ff_output = Dense(d_model)(ff_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    out2 = LayerNormalization(epsilon=1e-6)(Add()([out1, ff_output]))  # 残差连接
    return out2


def preprocess_data_with_time(traffic_data, window_size=12, steps_per_day=288):
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


# 计算 MAPE 并避免除零错误
def mean_absolute_percentage_error(y_true, y_pred):
    epsilon = 0.01  # 防止除以零
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100

# 新
def mean_absolute_percentage_error_custom(y_true, y_pred):
    epsilon = 1e-7  # 更小的 epsilon 防止除以零
    mask = np.abs(y_true) > epsilon
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / (np.abs(y_true[mask]) + epsilon))) * 100
    
def plot_training_history(history):
    # 提取训练和验证损失
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.show()

    plt.savefig('loss.pdf')



# 模型训练与评估
def train_and_evaluate_with_node_influence(X_train, y_train, X_test, y_test, adj_matrix, traffic_data, scaler, epochs,
                                           d_model, num_heads, ff_dim, num_layers, max_len, window_size,
                                           patience_early_stopping,
                                           patience_reduce_lr):
    # 计算节点影响力
    node_influence = detect_community(adj_matrix)

    # 调整邻接矩阵
    adjusted_adj_matrix = adjust_adjacency_matrix(adj_matrix, node_influence)

    # 构建模型
    model = build_transformer_model(
        input_shape=(window_size, traffic_data.shape[1] + 2),
        adj_matrix=adjusted_adj_matrix,
        d_model=d_model,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        max_len=max_len,
        num_scales=3  # 新增参数
    )

    # 定义回调函数
    early_stopping = EarlyStopping(
        monitor="val_loss",  # 监控验证集的损失
        patience=patience_early_stopping,  # 容忍验证集损失在 30 个 epoch 内不下降
        restore_best_weights=True  # 恢复验证集损失最优的权重
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",  # 监控验证集的损失
        factor=0.5,  # 学习率缩小因子，每次缩小为原来的 0.5
        patience=patience_reduce_lr,  # 容忍验证集损失在 5 个 epoch 内不下降
        min_lr=1e-6  # 最小学习率
    )

    r2_callback = R2Callback(X_test, y_test, scaler)

    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stopping, reduce_lr, r2_callback]  # 添加回调函数
    )

    # 预测结果
    y_pred = model.predict(X_test)

    # 逆标准化
    y_pred_inverse = scaler.inverse_transform(y_pred)
    y_test_inverse = scaler.inverse_transform(y_test)

    # 计算评估指标
    mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
    rmse = np.sqrt(mean_squared_error(y_test_inverse, y_pred_inverse))
    # mape = mean_absolute_percentage_error(y_test_inverse, y_pred_inverse)
    mape = mean_absolute_percentage_error_custom(y_test_inverse, y_pred_inverse)
    r2 = r2_score(y_test_inverse, y_pred_inverse)

    print(f'Final MAE: {mae:.4f}')
    print(f'Final RMSE: {rmse:.4f}')
    print(f'Final MAPE: {mape:.4f}')
    print(f'Final R²: {r2:.4f}')

    return history


# 主流程
if __name__ == '__main__':
    dataset_path = kagglehub.dataset_download("annnnguyen/metr-la-dataset")
    print("Path to dataset files:", dataset_path)

    adj_matrix_path = f'{dataset_path}/adj_METR-LA.pkl'
    traffic_data_path = f'{dataset_path}/METR-LA.h5'

    # 加载数据
    adj_matrix = load_adj_matrix(adj_matrix_path)
    traffic_data = load_traffic_data(traffic_data_path)

    # 数据预处理
    window_size = 12  # 使用过去12小时的数据进行预测
    steps_per_day = 288  # 每天有 288 个时间步
    epochs = 80
    d_model = 192
    num_heads = 4
    ff_dim = 128
    num_layers = 3
    max_len = 12
    # num_nested_blocks = 4  # 嵌套残差块的数量

    patience_reduce_lr = 5
    patience_early_stopping = 30

    # X, y, scaler = preprocess_data(traffic_data, window_size)
    X, y, scaler = preprocess_data_with_time(traffic_data, window_size, steps_per_day)

    # 检查输出形状
    print(f"X shape: {X.shape}, y shape: {y.shape} \n\n")
    # 期望 X 的形状为 (34260, 12, 209)
    # 期望 y 的形状为 (34260, 207)

    # 切分数据集
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 训练与评估模型
    history = train_and_evaluate_with_node_influence(
        X_train, y_train, X_test, y_test, adj_matrix, traffic_data, scaler, epochs,
        d_model=d_model, num_heads=num_heads, ff_dim=ff_dim, num_layers=num_layers,
        max_len=max_len, window_size=window_size,
        patience_early_stopping=patience_early_stopping, patience_reduce_lr=patience_reduce_lr
    )

    # 提取 loss 和 val_loss
    minimal_history = {
        "loss": history.history['loss'],
        "val_loss": history.history['val_loss']
    }

    # 使用时间戳作为动态文件名，避免覆盖
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    history_filename = f"temp/history_{timestamp}.json"

    with open(history_filename, 'w') as f:
        json.dump(minimal_history, f)

    print(f"History saved to {history_filename} successfully~!")


