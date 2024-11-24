# data/data_loader.py

import h5py
import pickle


def load_adj_matrix(file_path):
    with open(file_path, 'rb') as f:
        adj_data = pickle.load(f, encoding='latin1')
    adj_matrix = adj_data[2]  # 邻接矩阵是一个207x207的numpy矩阵
    return adj_matrix

def load_traffic_data(file_path):
    with h5py.File(file_path, 'r') as f:
        data = f['df']
        traffic_data = data['block0_values'][:]
        return traffic_data  # 形状为 (34272, 207)
