import tensorflow as tf
import numpy as np

"""
Multi heads attention, transfer tensor dimensions before and after being send into sdpa 
"""

_mod = tf.load_op_library("./sdpa.so.2.16")

@tf.RegisterGradient("Sdpa")
def _sdpa_grad(op: tf.Operation, d_o, *args):
    q, k, v, seed, offset = op.inputs
    o, s = op.outputs
    dropout = op.get_attr("dropout")
    scale = op.get_attr("scale")
    causal_mask = op.get_attr("causal_mask")
    dq, dk, dv = _mod.sdpa_grad(
        q, k, v, o, s, d_o, seed, offset,
        dropout=dropout, scale=scale, causal_mask=causal_mask
    )
    return dq, dk, dv, None, None


def sdpa(q: tf.Tensor, k: tf.Tensor, v: tf.Tensor,
         seed: tf.Tensor, offset: tf.Tensor, dropout: float = 0.0,
         scale: float = 1.0, causal_mask: bool = False):
    o, s = _mod.sdpa(q, k, v, seed, offset,
                     dropout=dropout, scale=scale, causal_mask=causal_mask)
    return o


class MHA(tf.keras.layers.Layer):
    def __init__(self, num_heads: int = 1, seq_axis: int = -2,
                 feature_axis: int = -1, dropout: float = 0.0,
                 seed: int = 42, causal_mask: bool = False, **kwargs):
        """
        Args:
            num_heads (int): Number of heads. Defaults to 1.
            seq_axis (int): Sequence axis, where attention is looking. Defaults to -2.
            feature_axis (int): Feature axis, where heads are splitted. Defaults to -1.
            dropout (float): Dropout rate. Defaults to 0.0.
            seed (int): Dropout seed. Defaults to 42.
            causal_mask (bool): Causal mask on or off. Defaults to False.
        """
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.seq_axis = seq_axis
        self.feature_axis = feature_axis
        self.dropout = dropout
        self.seed = seed
        self.causal_mask = causal_mask

    def build(self, input_shape):
        # 1. Parse Static Shapes
        q_shape = input_shape[0]
        v_shape = input_shape[2]
        rank = len(q_shape)

        # 2. Canonicalize Axes
        # Use Python logic, not TF ops
        s_ax = self.seq_axis if self.seq_axis >= 0 else rank + self.seq_axis
        f_ax = self.feature_axis if self.feature_axis >= 0 else rank + self.feature_axis

        # 3. Calculate Head Dimensions
        d_model = q_shape[f_ax]
        dv_model = v_shape[f_ax]
        assert d_model % self.num_heads == 0, "num_heads cannot be splitted evenly for Q"
        assert dv_model % self.num_heads == 0, "num_heads cannot be splitted evenly for V"

        self.head_dim = d_model // self.num_heads
        self.head_dim_v = dv_model // self.num_heads
        self.scale = 1.0 / (float(self.head_dim) ** 0.5)

        # 4. Pre-calculate Permutation Indices
        s_ax_split = s_ax if s_ax < f_ax else s_ax + 1
        h_ax_split = f_ax
        d_ax_split = f_ax + 1

        # Target: (Batch_Dims..., Head, Seq, Head_Dim)
        all_axes = list(range(rank + 1))  # +1 because of the split
        batch_axes = [i for i in all_axes if i not in (
            h_ax_split, s_ax_split, d_ax_split)]

        # This perm moves Head to front (after batch), Seq to middle, Head_Dim to end
        self.perm = batch_axes + [h_ax_split, s_ax_split, d_ax_split]

        # 5. Pre-calculate Inverse Permutation for Output
        self.inv_perm = np.argsort(self.perm).astype(np.int32).tolist()

        # Final shape after inverse transpose
        final_shape = []
        for i in range(rank):
            if i == f_ax:
                final_shape.append(dv_model)
            elif q_shape[i] is None:
                final_shape.append(-1)
            else:
                final_shape.append(q_shape[i])
        self.final_shape = final_shape
        self.offset = tf.Variable(0, trainable=False, dtype=tf.uint64)
        super().build(input_shape)

    def _split_and_transpose(self, x: tf.Tensor, head_dim: int):
        # 1. Reshape: Split Feature Axis -> Head, Head_Dim
        x_shape = x.shape
        rank = len(x_shape)
        f_ax = self.feature_axis if self.feature_axis >= 0 else rank + self.feature_axis

        # Construct new shape pattern: [...prefix, heads, head_dim, ...suffix]
        shape_pattern = []
        for i in range(rank):
            if i == f_ax:
                shape_pattern.append(self.num_heads)
                shape_pattern.append(head_dim)
            else:
                if x_shape[i] is None:
                    # Only fetch dynamic shape if static is None
                    shape_pattern.append(-1)
                else:
                    shape_pattern.append(x_shape[i])

        x = tf.reshape(x, shape_pattern)

        # 2. Transpose to align (B..., H, S, D)
        x = tf.transpose(x, perm=self.perm)

        # 3. Flatten [Batch..., Head] to single dim for 3D Attention
        s = tf.shape(x)
        return tf.reshape(x, [-1, s[-2], s[-1]]), s

    def call(self, inputs: list[tf.Tensor], training=None):
        """

        Args:
            inputs (list[tf.Tensor]): q, k, v. 
                - q, k, v should have same rank. 
            training (bool, optional): dropout or not. Defaults to None.

        Returns:
            tf.Tensor: output
        """
        q, k, v = inputs[:3]

        # --- Step 1: Prepare Inputs (Split & Permute) ---
        q_3d, q_shape_after_perm = self._split_and_transpose(q, self.head_dim)
        k_3d, _ = self._split_and_transpose(k, self.head_dim)
        v_3d, _ = self._split_and_transpose(v, self.head_dim_v)

        # --- Step 2: Attention ---
        if training:
            dropout = self.dropout
            offset = self.offset
        else:
            dropout = 0.0
            offset = tf.constant(0, dtype=self.offset.dtype)

        # Output: (B_total, Seq, Head_Dim_V)
        out_3d = sdpa(q_3d, k_3d, v_3d, seed=tf.constant(self.seed, dtype=tf.uint64),
                      causal_mask=self.causal_mask, dropout=dropout,
                      scale=self.scale, offset=offset)  # type: ignore

        if training:
            delta = tf.reduce_prod(
                tf.shape(q_3d)[:-1]) * tf.shape(k_3d)[1]  # B*N_q*N_k
            self.offset.assign_add(tf.cast(delta, self.offset.dtype))

        # --- Step 3: Restore Output ---
        # 1. Un-flatten: (B_total, Seq, Dv) -> (Batch..., Head, Seq, Dv)
        # Take all dims from q_shape_after_perm except the last one
        unflatten_shape = tf.concat(
            [q_shape_after_perm[:-1], [self.head_dim_v]], axis=0)
        out = tf.reshape(out_3d, unflatten_shape)

        # 2. Inverse Transpose: -> (Batch..., Seq, ..., Head, Head_Dim_V)
        out = tf.transpose(out, perm=self.inv_perm)

        # 3. Merge Heads: -> (Batch..., Seq, ..., Dv_total)
        return tf.reshape(out, self.final_shape)
