
## Dependencies

Please execute the following command to install the necessary dependencies:

```bash
pip3 install -r requirements.txt
```


## Test CUDA available

Must test __TensorFlow__ in __CUDA__ is available: 
```python
import tensorflow as tf
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

* if show that `Num GPUs Available:` ≥ 1, that is Ready successfully!


## Quick Start

```bash
python3 main.py
```


### Attention

**Do NOT** OPEN CUDA version **12.x** in AutoDL!! Because the latest TensorFlow version is not yet available, only **11.8** will be available!
