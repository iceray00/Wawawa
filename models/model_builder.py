import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, Add, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from .layers import MultiScaleGraphConvolution, LearnablePositionalEncoding

def causal_mask_def(batch_size, seq_len):
    mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)  # 下三角矩阵 [seq_len, seq_len]
    mask = tf.expand_dims(mask, axis=0)  # [1, seq_len, seq_len]
    mask = tf.tile(mask, [batch_size, 1, 1])  # [batch_size, seq_len, seq_len]
    return mask

class CausalMultiHeadAttention(tf.keras.layers.MultiHeadAttention):
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

        return super().call(query=query, value=value, key=key, attention_mask=attention_mask, training=training, **kwargs)

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

def build_transformer_model(input_shape, adj_matrix, d_model=64, num_heads=4, ff_dim=128, num_layers=3, max_len=12):
    inputs = Input(shape=input_shape)

    # 使用自定义层提取空间信息
    x = MultiScaleGraphConvolution(adj_matrix, d_model)(inputs)

    # 可学习的位置编码
    x = LearnablePositionalEncoding(max_len, d_model)(x)

    # 多层 Transformer 编码
    for _ in range(num_layers):
        x = transformer_block(x, d_model, num_heads, ff_dim)

    # 输出层（忽略时间特征部分，仅预测交通流量特征）
    outputs = Dense(207)(x[:, -1, :])  # 对最后一个时间步的节点特征进行预测
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss='mse', metrics=['mae', 'mape'])
    return model
