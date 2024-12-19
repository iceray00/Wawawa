import glob
import json
import subprocess
import re
import numpy as np
import argparse
import csv
import os

from matplotlib import pyplot as plt

ignore_pattern = re.compile(
    r"^\s*\d+/\d+\s+\[.*\]\s+-\s+ETA:.*|^\s*loss:.*mae:.*mape:.*"  # 匹配训练进度条和指标
)

key_pattern = re.compile(r"Epoch\s+\d+.*R²:.*")  # 保留含有 Epoch 和 R² 的行

epoch_pattern = re.compile(r"Epoch\s+\d+/\d+")

# 正则表达式来提取 R² 和 MAE 值
r2_pattern = re.compile(r'Final R²:\s*([0-9.\-]+)')
mae_pattern = re.compile(r'Final MAE:\s*([0-9.\-]+)')



def run_script(num_runs, model, output_path):
    # 结果存储列表
    r2_scores = []
    mae_scores = []
    all_histories = []

    command = [
        "python3", model,
    ]

    # 多次运行 PageRank.py
    for i in range(num_runs):
        print(f"\nRunning iteration {i + 1}/{num_runs}...\n")
        try:
            # 记录运行前的所有 history 文件，避免混淆
            existing_files = set(glob.glob("temp/history_*.json"))
            print(f"Existing history files: {existing_files}")

            # 使用 Popen 启动 PageRank.py 进程，并捕获输出流
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True)

            # 实时逐行读取输出
            r2_value = None
            mae_value = None
            for line in iter(process.stdout.readline, ''):  # 逐行读取标准输出
                if epoch_pattern.search(line):
                    print(line, end='')
                    continue

                line = line.strip()  # 去除行首尾的空格和换行符
                if not line:  # 跳过空行
                    continue

                # 如果该行包含 "Epoch"，直接输出，不过滤
                if "Epoch" in line:
                    print(line, end='')
                    continue

                # 如果当前行匹配到过滤正则，则跳过，不打印
                if ignore_pattern.match(line):
                    continue

                print(line)  # 将每一行实时输出到屏幕
                # 提取 MAE 和 R²
                if mae_match := mae_pattern.search(line):
                    mae_value = float(mae_match.group(1))
                    print("Get mae in script Successfully!")
                if r2_match := r2_pattern.search(line):
                    r2_value = float(r2_match.group(1))
                    print("Get r2 in script Successfully!")

            process.wait()

            # 查找新生成的 history 文件
            new_files = set(glob.glob("temp/history_*.json")) - existing_files
            if new_files:
                latest_file = max(new_files, key=os.path.getctime)  # 按创建时间获取最新文件
                print(f"Found new history file: {latest_file}")
                with open(latest_file, 'r') as f:
                    history = json.load(f)
                    all_histories.append(history)

            if r2_value is not None and mae_value is not None:
                print(f"Iteration {i + 1}: R² = {r2_value}, MAE = {mae_value}")
                r2_scores.append(r2_value)
                mae_scores.append(mae_value)
            else:
                print(f"Could not find R² or MAE in the output for iteration {i + 1}. Skipping...")

        except Exception as e:
            print(f"Error running {model}.py in iteration {i + 1}: {e}")
            continue  # 跳过错误的运行

    model_basename = os.path.basename(model)
    model_purename = os.path.splitext(model_basename)[0]

    # 计算平均值和标准差
    if r2_scores and mae_scores:
        avg_r2 = np.mean(r2_scores)
        std_r2 = np.std(r2_scores)
        avg_mae = np.mean(mae_scores)
        std_mae = np.std(mae_scores)

        print("\n===== Final Results =====")
        print(f"Average R²: {avg_r2:.4f} ± {std_r2:.4f}")
        print(f"Average MAE: {avg_mae:.4f} ± {std_mae:.4f}")

        # 将结果写入CSV文件（追加模式）
        if os.path.exists(f'{output_path}/results.csv'):
            with open(f'{output_path}/results.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([model_purename, avg_mae, std_mae, avg_r2, std_r2])
        else:
            with open(f'{output_path}/results.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['model_name', 'avg_mae', 'std_mae', 'avg_r2', 'std_r2'])
                writer.writerow([model_purename, avg_mae, std_mae, avg_r2, std_r2])

        if all_histories:
            plot_average_history(all_histories, model_purename, output_path) # 平均处理历史数据并绘图保存

    else:
        print("No valid results were obtained.")



def plot_average_history(all_histories, model_name, output_path):
    num_runs = len(all_histories)
    num_epochs = len(all_histories[0]['loss'])

    # 初始化平均值容器
    avg_train_loss = np.zeros(num_epochs)
    avg_val_loss = np.zeros(num_epochs)

    # 累加每次的损失
    for history in all_histories:
        avg_train_loss += np.array(history['loss'])
        avg_val_loss += np.array(history['val_loss'])

    # 取平均
    avg_train_loss /= num_runs
    avg_val_loss /= num_runs

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(avg_train_loss, label='Average Training Loss', color='blue')
    plt.plot(avg_val_loss, label='Average Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Average Training and Validation Loss ({model_name})')
    plt.legend()
    plt.grid()

    # 保存为 PDF
    output_file = f'{output_path}/{model_name}_average_loss.pdf'
    plt.savefig(output_file)
    print(f"Average loss plot saved as {output_file}")
    plt.close()

    # baocun avg_losses
    avg_losses = {
        "average_train_loss": avg_train_loss.tolist(),
        "average_val_loss": avg_val_loss.tolist()
    }
    with open(f"{output_path}/{model_name}_avg_losses.json", "w") as json_file:
        json.dump(avg_losses, json_file, indent=4)

    print(f"Average losses saved to {output_path}/{model_name}_avg_losses.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--num_runs', type=int, default=10, help='Number of runs to perform')
    parser.add_argument('-m','--model', type=str, default='PageRank.py', help='Path to the model script')
    parser.add_argument('-p','--output-path', type=str, default=None, help='Output file path')
    args = parser.parse_args()

    os.makedirs("temp", exist_ok=True)

    print("Cleaning up the temporary history json files...")
    os.system("rm temp/history*.json")

    os.makedirs(args.output_path, exist_ok=True)

    run_script(args.num_runs, args.model, args.output_path)


if __name__ == '__main__':
    main()
