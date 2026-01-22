import tensorflow as tf
import numpy as np
from time import time
np.set_printoptions(precision=4, suppress=True)
from hfs_tf.utils import set_gpu
set_gpu(0)

mod = tf.load_op_library("./sdpa.so.2.16")
tf.random.set_seed(43)


def sdpa(q: tf.Tensor, k: tf.Tensor, v: tf.Tensor,
         seed: tf.Tensor, offset: tf.Tensor, dropout: float = 0.0, scale: float = 1.0, causal_mask: bool = False):
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


def get_random_float(seed, offset, b, i, j, stride_m, stride_n):
    """
    python random generator
    """
    stride_m = int(stride_m)
    stride_n = int(stride_n)
    batch_stride = stride_m * stride_n
    idx = b * batch_stride + i * stride_n + j
    global_idx = offset + idx
    state = int(seed) + global_idx
    state = (state * 6364136223846793005 +
             1442695040888963407) & ((1 << 64) - 1)
    val = (state >> 32) & 0xFFFFFFFF
    return float(val) / 4294967296.0


def build_random_tensor(B, M, N, seed, offset):
    random_tensor_np = np.zeros((B, M, N), dtype=np.float32)
    for b in range(B):
        for i in range(M):
            for j in range(N):
                random_tensor_np[b, i, j] = get_random_float(
                    int(seed), int(offset), b, i, j, M, N)
    return tf.convert_to_tensor(random_tensor_np)


def make_mask(N):
    idx = tf.range(N)
    mask = idx[:, None] < idx[None, :]
    return mask[None, ...]


def manual_attention(q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, scale: float,
                     causal_mask: bool = False, dropout: float = 0.0,
                     seed: int = 1234, offset: int = 0):
    # This should produce the same results
    B, M = list(q.shape)[:2]
    N = k.shape[1]
    score = tf.matmul(q, k, transpose_b=True)
    score = score * scale
    score = tf.cast(score, dtype='float32')
    if causal_mask:
        mask = make_mask(N)
        mask_float = tf.cast(mask, dtype=score.dtype)
        score += mask_float * -1e9
    m = tf.reduce_max(score, axis=-1)
    l = tf.reduce_sum(tf.exp(score - m[..., None]), axis=-1)
    stats = m + tf.math.log(l)
    probs = tf.nn.softmax(score, axis=-1)
    if dropout > 0.0:
        random_dropout = build_random_tensor(B, M, N, seed, offset)
        dropout_mask = tf.cast(random_dropout >= dropout, probs.dtype)
        probs *= dropout_mask
        probs /= (1.0 - dropout)
    output = tf.matmul(tf.cast(probs, v.dtype), v)
    return output, stats


def test_run(q, k, v, seed, offset,
             dropout, causal_mask, scale, f):
    t0 = time()
    with tf.GradientTape() as tape:
        tape.watch([q, k, v])
        out, *_ = f(q=q, k=k, v=v, causal_mask=causal_mask, seed=seed,
                    offset=offset, dropout=dropout, scale=scale)
        l = tf.reduce_sum(tf.reduce_max(out, axis=-1))
    t1 = time()
    g = tape.gradient(l, [q, k, v])
    t2 = time()
    return out, g, t1 - t0, t2 - t1


def test(q, k, v, seed, offset, dropout, causal_mask, scale, f, d, N):
    with tf.device(d):
        out_, g_, *_ = test_run(q, k, v, seed, offset,
                              dropout, causal_mask, scale, f)
    tt0 = 0
    tt1 = 0
    r = []
    for _ in range(N):
        with tf.device(d):
            outs, g, t0, t1 = test_run(q, k, v, seed, offset,
                                       dropout, causal_mask, scale, f)
            r.append([outs.numpy(), [gg.numpy() for gg in g]])
        tt0 += t0
        tt1 += t1
    print(f.__name__, d, ": forward %.4fs, backward %.4fs, total %.4fs" %
          (tt0, tt1, tt1 + tt0))
    return out_.numpy(), [gg.numpy() for gg in g_], r


def compare(res_a, res_b):
    def calc(a, b):
        d = np.abs(a - b)
        c = np.corrcoef(a.ravel(), b.ravel())[0, 1]
        return d, c
    out_a, grad_a, *_ = res_a
    out_b, grad_b, *_ = res_b
    d, c = calc(out_a, out_b)
    print(
        f"out diff: mean {np.mean(d):.4f}, max {np.max(d):.4f}, corr {c:.4f}")

    for i, (ga, gb) in enumerate(zip(grad_a, grad_b)):
        d, c = calc(ga, gb)
        print(
            f"grad {i} diff: mean {np.mean(d):.4f}, max {np.max(d):.4f}, corr {c:.4f}")


if __name__ == '__main__':
    roundN = 2
    B = 20
    S_q = 400
    S_kv = 400
    D_qk = 32
    D_v = 4
    causal_mask = False
    dropout = 0.2
    dtype = tf.float16
    seed = 42
    offset = 1000
    scale = 1 / np.sqrt(D_qk)
    q = tf.random.normal(shape=(B, S_q, D_qk), dtype=dtype, seed=1024)
    k = tf.random.normal(shape=(B, S_kv, D_qk), dtype=dtype, seed=124)
    v = tf.random.normal(shape=(B, S_kv, D_v), dtype=dtype, seed=24)

    manual_cpu = test(q, k, v, seed, offset, dropout,
                      causal_mask, scale, manual_attention, "/CPU:0", roundN)
    manual_gpu = test(q, k, v, seed, offset, dropout,
                      causal_mask, scale, manual_attention, "/GPU:0", roundN)
    flash_cpu = test(q, k, v, seed, offset, dropout,
                     causal_mask, scale, sdpa, "/CPU:0", roundN)
    flash_gpu = test(q, k, v, seed, offset, dropout,
                     causal_mask, scale, sdpa, "/GPU:0", roundN)

    print("\n>>> flash cpu vs flash gpu")
    compare(flash_cpu, flash_gpu) 

    print("\n>>> flash cpu vs manual cpu")
    compare(flash_cpu, manual_cpu) 

    print("\n>>> manual cpu vs manual gpu")
    compare(manual_cpu, manual_gpu) 

    print("K-gpu repeat")
    for r, g2 in enumerate(flash_gpu[2]):
        print("---", r)
        compare(g2, flash_gpu) 

