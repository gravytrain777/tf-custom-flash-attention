# Flash Attention - TensorFlow Custom Op

This is an unofficial implementation of Flash Attention (v1 & v2) for TensorFlow with CUDA. Aim is to be lightweighted and relies solely on TensorFlow 2.x and no cudnn nor xla. Performance may differ from the [official PyTorch implementation](https://github.com/HazyResearch/flash-attention).

## Features

- CPU and GPU implementations
- Causal masking support
- Dropout support
- Lightweight with minimal dependencies

## Installation

### Pre-built Binary

For python 3.11 TensorFlow 2.16 with CUDA 12.3 on Linux, use the pre-compiled `.so` file in the `asset` directory.

### Build from Source

See `build.sh` for build instructions. Update the script with your Python environment and compiler paths (g++, nvcc).

## Usage

### 1. Register the Custom Operation

Create a Python script to load and register the gradient operation:

```python
import tensorflow as tf

# Load the custom op
mod = tf.load_op_library("./sdpa.so.2.16")

def sdpa(q: tf.Tensor, k: tf.Tensor, v: tf.Tensor,
         seed: tf.Tensor, offset: tf.Tensor, 
         dropout: float = 0.0, 
         scale: float = 1.0, 
         causal_mask: bool = False):
    o, s = mod.sdpa(q, k, v, seed, offset,
                    dropout=dropout, scale=scale, causal_mask=causal_mask)
    return o, s

@tf.RegisterGradient("Sdpa")
def _sdpa_grad(op: tf.Operation, d_o, *args):
    q, k, v, seed, offset = op.inputs
    o, s = op.outputs
    dropout = op.get_attr("dropout")
    scale = op.get_attr("scale")
    causal_mask = op.get_attr("causal_mask")
    
    dq, dk, dv = mod.sdpa_grad(
        q, k, v, o, s, d_o, seed, offset,
        dropout=dropout, scale=scale, causal_mask=causal_mask
    )
    return dq, dk, dv, None, None
```

### 2. Example

```python
import numpy as np

# Configuration
B = 20          # Batch size
S_q = 400       # Query sequence length
S_kv = 400      # Key/Value sequence length
D_qk = 32       # Query/Key dimension
D_v = 4         # Value dimension
dropout = 0.2
scale = 1.0 / np.sqrt(D_qk)
causal_mask = False

# Random seed for reproducibility
seed = tf.constant(42, dtype=tf.int32)
offset = tf.constant(1000, dtype=tf.int32)

# Input tensors
q = tf.random.normal(shape=(B, S_q, D_qk), dtype=tf.float16)
k = tf.random.normal(shape=(B, S_kv, D_qk), dtype=tf.float16)
v = tf.random.normal(shape=(B, S_kv, D_v), dtype=tf.float16)

# GPU execution with gradient computation
with tf.device('/GPU:0'):
    with tf.GradientTape() as tape:
        tape.watch([q, k, v])
        out, _ = sdpa(q=q, k=k, v=v, 
                     seed=seed, offset=offset,
                     dropout=dropout, 
                     scale=scale, 
                     causal_mask=causal_mask)
        loss = tf.reduce_sum(tf.reduce_max(out, axis=-1))
    
    gradients = tape.gradient(loss, [q, k, v])

print("Output:", out.numpy())
print("Gradients dQ:", gradients[0].numpy())
```

See `test.py` for more comprehensive examples.

## Details

### Input Requirements
- All inputs must be **3D tensors** with shape `[batch, sequence, dimension]`
- Supported dtypes: `float16`, `float32`

### CPU vs GPU
- **CPU**: Uses memory-efficient attention not strictly Flash Attention
- **GPU**: Implements Flash Attention with self-optimized CUDA kernels

### Backward Pass
The GPU backward pass uses two separate kernels:
1. Compute dQ 
2. Compute dK & dV 

This approach is faster than a single kernel with atomic operations from my experiments.

## References

- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [Official PyTorch Implementation](https://github.com/HazyResearch/flash-attention)

