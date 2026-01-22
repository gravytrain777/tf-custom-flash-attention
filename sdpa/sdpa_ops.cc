#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow
{

REGISTER_OP("Sdpa")
    .Input("q: T") // [B, Sq, D]
    .Input("k: T") // [B, Skv, D]
    .Input("v: T") // [B, Skv, Dv]
    .Input("seed: uint64")
    .Input("offset: uint64")
    .Attr("scale: float = 1.0")
    .Attr("dropout: float = 0.0")
    .Attr("causal_mask: bool = false")
    .Attr("T: {half, float}")
    .Output("output: T")    // [B, Sq, Dv]
    .Output("stats: float") // [B, Sq]
    .SetShapeFn(
        [](shape_inference::InferenceContext *c)
        {
            shape_inference::ShapeHandle q, k, v;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &q));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &k));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 3, &v));
            shape_inference::DimensionHandle b = c->Dim(q, 0);
            shape_inference::DimensionHandle s_q = c->Dim(q, 1);
            shape_inference::DimensionHandle d_v = c->Dim(v, 2);
            c->set_output(0, c->MakeShape({b, s_q, d_v}));
            c->set_output(1, c->MakeShape({b, s_q}));
            return OkStatus();
        });

REGISTER_OP("SdpaGrad")
    .Input("q: T")         // [B, Sq, D]
    .Input("k: T")         // [B, Skv, D]
    .Input("v: T")         // [B, Skv, Dv]
    .Input("out: T")       // [B, Sq, Dv]
    .Input("stats: float") // [B, Sq]
    .Input("do: T")        // [B, Sq, Dv]
    .Input("seed: uint64")
    .Input("offset: uint64")
    .Output("dq: T") // [B, Sq, D]
    .Output("dk: T") // [B, Skv, D]
    .Output("dv: T") // [B, Skv, Dv]
    .Attr("dropout: float = 0.0")
    .Attr("scale: float = 1.0")
    .Attr("causal_mask: bool = false")
    .Attr("T: {half, float}")
    .SetShapeFn(
        [](shape_inference::InferenceContext *c)
        {
            shape_inference::ShapeHandle q_shape;
            shape_inference::ShapeHandle k_shape;
            shape_inference::ShapeHandle v_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &q_shape)); // Q
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &k_shape)); // K
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 3, &v_shape)); // V
            c->set_output(0, q_shape);
            c->set_output(1, k_shape);
            c->set_output(2, v_shape);
            return OkStatus();
        });

} // namespace tensorflow