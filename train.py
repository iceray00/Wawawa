import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from models.model_builder import build_transformer_model
from utils.metrics import mean_absolute_percentage_error
from utils.show_hyperparameters import show_hyperparameters

def train_and_evaluate(X_train, y_train, X_test, y_test, adj_matrix, scaler, epochs, d_model, num_heads, ff_dim,
                       num_layers, max_len, window_size, patience_early_stopping, patience_reduce_lr, save_fig):
    # 构建模型
    model = build_transformer_model(
        input_shape=(window_size, X_train.shape[2]),
        adj_matrix=adj_matrix,
        d_model=d_model,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        max_len=max_len
    )

    # 定义回调函数
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience_early_stopping,
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=patience_reduce_lr,
        min_lr=1e-7
    )

    # 训练模型
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stopping, reduce_lr]
    )

    # 预测结果
    y_pred = model.predict(X_test)

    # 逆标准化
    y_pred_inverse = scaler.inverse_transform(y_pred)
    y_test_inverse = scaler.inverse_transform(y_test)

    # 计算评估指标
    mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
    rmse = np.sqrt(mean_squared_error(y_test_inverse, y_pred_inverse))
    mape = mean_absolute_percentage_error(y_test_inverse, y_pred_inverse)
    r2 = r2_score(y_test_inverse, y_pred_inverse)

    # 展示超参数
    show_hyperparameters(
        window_size=window_size,
        d_model=d_model,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        max_len=max_len,
        patience_early_stopping=patience_early_stopping,
        patience_reduce_lr=patience_reduce_lr,
        save_fig=save_fig
    )
    print(f'MAE: {mae:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAPE: {mape:.4f}')
    print(f'R²: {r2:.4f}')

    # 可视化预测结果
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_inverse[:, 0], label='True')
    plt.plot(y_pred_inverse[:, 0], label='Predicted')
    plt.legend()
    plt.title("Traffic Flow Prediction with Time Features at First Load")
    if save_fig: plt.savefig("At_1th_Load.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(y_test_inverse[:, 100], label='True')
    plt.plot(y_pred_inverse[:, 100], label='Predicted')
    plt.legend()
    plt.title("Traffic Flow Prediction with Time Features at 100th Load")
    if save_fig: plt.savefig("At_100th_Load.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(y_test_inverse[:, -1], label='True')
    plt.plot(y_pred_inverse[:, -1], label='Predicted')
    plt.legend()
    plt.title("Traffic Flow Prediction with Time Features at Last Load")
    if save_fig: plt.savefig("At_207th_Load.png")
    plt.show()
