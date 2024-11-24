import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, MultiHeadAttention, LayerNormalization, Add, BatchNormalization
from tensorflow.keras import regularizers

class MultiScaleGraphConvolution(Layer):
    def __init__(self, adj_matrix, d_model, dropout_rate=0.1):
        super(MultiScaleGraphConvolution, self).__init__()
        self.adj_matrix = tf.convert_to_tensor(adj_matrix, dtype=tf.float32)  # 转为张量
        self.dense = Dense(d_model, kernel_regularizer=regularizers.l2(1e-4))  # 添加 L2 正则化
        self.dropout = Dropout(dropout_rate)
        self.concat_dense = Dense(d_model)  # 将拼接后的特征投影回 d_model

    def call(self, inputs):
        # 分离交通流量特征和时间特征
        traffic_data = inputs[:, :, :207]  # 提取交通流量特征（与邻接矩阵匹配的部分）
        time_features = inputs[:, :, 207:]  # 提取时间特征

        # 图卷积操作
        graph_output = tf.matmul(traffic_data, self.adj_matrix)  # [batch, window, 207] * [207, 207] = [batch, window, 207]
        graph_output = self.dense(graph_output)  # [batch, window, d_model]
        graph_output = self.dropout(graph_output)

        # 拼接时间特征
        combined_output = tf.concat([graph_output, time_features], axis=-1)  # [batch, window, d_model + 2]

        # 投影回 d_model 维度
        combined_output = self.concat_dense(combined_output)  # [batch, window, d_model]

        return combined_output

class LearnablePositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model):
        super(LearnablePositionalEncoding, self).__init__()
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(max_len, d_model),
            initializer="uniform",
            trainable=True
        )

    def call(self, x):
        return x + self.pos_embedding[:tf.shape(x)[1], :]
