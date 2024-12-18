# test/test_cuda_available.py

import tensorflow as tf

def test_cuda_available():
    """
    测试 CUDA 是否可用。
    """
    print(tf.__version__)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


if __name__ == '__main__':
    test_cuda_available()
