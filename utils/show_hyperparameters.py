# utils/show_hyperparameters.py

def show_hyperparameters(window_size, d_model, num_heads, ff_dim, num_layers, max_len, patience_early_stopping, patience_reduce_lr, save_fig):
    print("\n===== Hyperparameters =====")
    print(f"window_size: {window_size}")
    print(f"d_model: {d_model}")
    print(f"num_heads: {num_heads}")
    print(f"ff_dim: {ff_dim}")
    print(f"num_layers: {num_layers}")
    print(f"max_len: {max_len}")
    print(f"patience_reduce_lr: {patience_reduce_lr}")
    print(f"patience_early_stopping: {patience_early_stopping}")
    print(f"save_fig: {save_fig}")
    print("===========================\n")