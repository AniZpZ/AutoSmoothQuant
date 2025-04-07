import gc
import re
from typing import Optional, Tuple
import copy

import torch
import threading
from ..functional.quantization import (
    quantize_per_tensor_absmax,
    per_tensor_quantize_fp8,
    per_token_quantize_fp8,
    static_per_tensor_quantize_fp8
)
from autosmoothquant._CUDA import I8CUGEMM


class Int8GEMM(object):
    _instance_lock = threading.Lock()

    def __init__(self):
        if not hasattr(self, "i8cugemm"):
            self.i8cugemm = I8CUGEMM()

    def __new__(cls, *args, **kwargs):
        if not hasattr(Int8GEMM, "_instance"):
            with Int8GEMM._instance_lock:
                if not hasattr(Int8GEMM, "_instance"):
                    Int8GEMM._instance = object.__new__(cls)
        return Int8GEMM._instance

    def get_i8cugemm(self):
        return self.i8cugemm


class W8A8BFP32OFP32Linear(torch.nn.Module):
    # For fc2 and out_proj
    def __init__(
        self, in_features, out_features, use_bias=False, act_quant="per-tensor"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        assert act_quant in ["per-token", "per-tensor"], '"act_quant must be "per-token" or "per-tensor"'
        self.act_quant = act_quant
        GEMM = Int8GEMM()
        self.i8cugemm = GEMM.get_i8cugemm()

        self.register_buffer(
            "weight",
            torch.empty(
                self.out_features,
                self.in_features,
                dtype=torch.int8,
                requires_grad=False,
            ),
        )
        if self.use_bias:
            self.register_buffer(
                "bias",
                torch.zeros(self.out_features, dtype=torch.float32, requires_grad=False
                ),
            )
        self.register_buffer(
            "dequant_scale", torch.tensor(1.0, dtype=torch.float32, requires_grad=False)
        )

    def _apply(self, fn):
        # prevent the bias from being converted to half
        super()._apply(fn)
        self.dequant_scale = self.dequant_scale.cpu()
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.dequant_scale = self.dequant_scale.to(*args, **kwargs)
        self.dequant_scale = self.dequant_scale.to(torch.float32)
        if self.use_bias:
            self.bias = self.bias.to(*args, **kwargs)
            self.bias = self.bias.to(torch.float32)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        dtype = x.dtype
        x = x.view(-1, x_shape[-1])
        if self.act_quant == "per-token":
            quant_scale = (
                x.abs().max(dim=-1, keepdim=True)[0].div(127.0).to(torch.float32)
            )
            x = (x / quant_scale).round().clamp(-128, 127).to(torch.int8)
            dequant_scale = self.dequant_scale.item() * quant_scale
        else:
            x = x.round().clamp(-128, 127).to(torch.int8)
            dequant_scale = self.dequant_scale.item()
        out = torch.empty(
            x.shape[0],
            self.out_features,
            dtype=torch.int32,
            device=torch.cuda.current_device(),
        )
        self.i8cugemm.linear_a8_w8_o32_(x, self.weight, out)
        out = dequant_scale * out + self.bias if self.use_bias else dequant_scale * out
        out = out.view(*x_shape[:-1], -1).to(dtype)
        return out

    @staticmethod
    def from_float(
        module: torch.nn.Linear,
        input_scale=1.0,
        save_device=torch.device("cpu"),
        act_quant="per-tensor",
    ):
        assert act_quant in [
            "per-token",
            "per-tensor",
        ], '"act_quant must be "per-token" or "per-tensor"'
        use_bias = False if module.bias is None else True
        int8_module = W8A8BFP32OFP32Linear(
            module.in_features, module.out_features, use_bias, act_quant
        )
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        alpha = weight_scale if act_quant == "per-token" else input_scale * weight_scale
        int8_module.dequant_scale = alpha.to(torch.float32).to(save_device)
        int8_module.weight = int8_weight.to(save_device)
        if int8_module.use_bias:
            int8_module.bias = module.bias.to(torch.float32).to(save_device)
        return int8_module


class W8A8BFP32OFP32QKVLinear(W8A8BFP32OFP32Linear):
    # for fused qkv weight
    def __init__(self, qkv_size, *args, **kwargs):
        self.qkv_size = qkv_size
        super().__init__(*args, **kwargs)
        self._buffers.pop("dequant_scale")
        self.register_buffer(
            "q_dequant_scale",
            torch.tensor(1.0, dtype=torch.float32, requires_grad=False),
        )
        self.register_buffer(
            "k_dequant_scale",
            torch.tensor(1.0, dtype=torch.float32, requires_grad=False),
        )
        self.register_buffer(
            "v_dequant_scale",
            torch.tensor(1.0, dtype=torch.float32, requires_grad=False),
        )

    def _apply(self, fn):
        # prevent the bias from being converted to half
        super(W8A8BFP32OFP32Linear, self)._apply(fn)
        self.q_dequant_scale = self.q_dequant_scale.cpu()
        self.k_dequant_scale = self.k_dequant_scale.cpu()
        self.v_dequant_scale = self.v_dequant_scale.cpu()
        return self

    def to(self, *args, **kwargs):
        super(W8A8BFP32OFP32Linear, self).to(*args, **kwargs)
        if self.use_bias:
            self.bias = self.bias.to(*args, **kwargs)
            self.bias = self.bias.to(torch.float32)
        self.q_dequant_scale = self.q_dequant_scale.to(*args, **kwargs)
        self.q_dequant_scale = self.q_dequant_scale.to(torch.float32)
        self.k_dequant_scale = self.k_dequant_scale.to(*args, **kwargs)
        self.k_dequant_scale = self.k_dequant_scale.to(torch.float32)
        self.v_dequant_scale = self.v_dequant_scale.to(*args, **kwargs)
        self.v_dequant_scale = self.v_dequant_scale.to(torch.float32)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        dtype = x.dtype
        x = x.view(-1, x_shape[-1])
        if self.act_quant == "per-token":
            quant_scale = (
                x.abs().max(dim=-1, keepdim=True)[0].div(127.0).to(torch.float32)
            )
            x = (x / quant_scale).round().clamp(-128, 127).to(torch.int8)
            q_dequant_scale = self.q_dequant_scale.item() * quant_scale
            k_dequant_scale = self.k_dequant_scale.item() * quant_scale
            v_dequant_scale = self.v_dequant_scale.item() * quant_scale
        else:
            x = x.round().clamp(-128, 127).to(torch.int8)
            q_dequant_scale = self.q_dequant_scale.item()
            k_dequant_scale = self.k_dequant_scale.item()
            v_dequant_scale = self.v_dequant_scale.item()
        out = torch.empty(
            x.shape[0],
            self.out_features,
            dtype=torch.int32,
            device=torch.cuda.current_device(),
        )
        self.i8cugemm.linear_a8_w8_o32_(x, self.weight, out)
        q, k, v = out.split(self.qkv_size, dim=-1)
        q_dq = q_dequant_scale * q
        k_dq = k_dequant_scale * k
        v_dq = v_dequant_scale * v
        if self.use_bias:
            q_bias, k_bias, v_bias = self.bias.split(self.qkv_size, dim=-1)
            q_dq += q_bias
            k_dq += k_bias
            v_dq += v_bias
        out = torch.cat([q_dq, k_dq, v_dq], dim=-1)
        out = out.view(*x_shape[:-1], -1).to(dtype)
        return out

    @staticmethod
    def from_float(
        module: torch.nn.Linear,
        input_scale,
        qkv_size,
        save_device=torch.device("cpu"),
        act_quant="per-tensor",
    ):
        assert act_quant in [
            "per-token",
            "per-tensor",
        ], '"act_quant must be "per-token" or "per-tensor"'
        use_bias = False if module.bias is None else True
        int8_module = W8A8BFP32OFP32QKVLinear(
            qkv_size, module.in_features, module.out_features, use_bias, act_quant
        )
        q_weight, k_weight, v_weight = module.weight.data.split(qkv_size, dim=0)
        q_int8_weight, q_weight_scale = quantize_per_tensor_absmax(q_weight)
        k_int8_weight, k_weight_scale = quantize_per_tensor_absmax(k_weight)
        v_int8_weight, v_weight_scale = quantize_per_tensor_absmax(v_weight)
        int8_module.weight = torch.cat(
            [q_int8_weight, k_int8_weight, v_int8_weight], dim=0
        ).to(save_device)
        alpha = q_weight_scale
        beta = k_weight_scale
        gamma = v_weight_scale
        if act_quant == "per-tensor":
            alpha = q_weight_scale * input_scale
            beta = k_weight_scale * input_scale
            gamma = v_weight_scale * input_scale
        int8_module.q_dequant_scale = alpha.to(torch.float32).to(save_device)
        int8_module.k_dequant_scale = beta.to(torch.float32).to(save_device)
        int8_module.v_dequant_scale = gamma.to(torch.float32).to(save_device)
        if int8_module.use_bias:
            int8_module.bias = module.bias.to(torch.float32).to(save_device)
        return int8_module


class W8A8BFP32OFP32LinearWithQuantScale(W8A8BFP32OFP32Linear):
    # For fc2 and out_proj
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.act_quant == "per-tensor":
            self.register_buffer(
                "quant_scale",
                torch.tensor(1.0, dtype=torch.float32, requires_grad=False),
            )

    def _apply(self, fn):
        # prevent the bias from being converted to half
        super(W8A8BFP32OFP32Linear, self)._apply(fn)
        self.dequant_scale = self.dequant_scale.cpu()
        if self.act_quant == "per-tensor":
            self.quant_scale = self.quant_scale.cpu()
        return self

    def to(self, *args, **kwargs):
        super(W8A8BFP32OFP32Linear, self).to(*args, **kwargs)
        if self.use_bias:
            self.bias = self.bias.to(*args, **kwargs)
            self.bias = self.bias.to(torch.float32)
        self.dequant_scale = self.dequant_scale.to(*args, **kwargs)
        self.dequant_scale = self.dequant_scale.to(torch.float32)
        if self.act_quant == "per-tensor":
            self.quant_scale = self.quant_scale.to(*args, **kwargs)
            self.quant_scale = self.quant_scale.to(torch.float32)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        dtype = x.dtype
        x = x.view(-1, x_shape[-1])
        if self.act_quant == "per-token":
            quant_scale = (
                x.abs().max(dim=-1, keepdim=True)[0].div(127.0).to(torch.float32)
            )
            dequant_scale = self.dequant_scale.item() * quant_scale
        else:
            dequant_scale = self.dequant_scale.item()
            quant_scale = self.quant_scale.item()
        # quant here
        x = (x / quant_scale).round().clamp(-128, 127).to(torch.int8)
        out = torch.empty(
            x.shape[0],
            self.out_features,
            dtype=torch.int32,
            device=torch.cuda.current_device(),
        )
        self.i8cugemm.linear_a8_w8_o32_(x, self.weight, out)
        out = dequant_scale * out + self.bias if self.use_bias else dequant_scale * out
        out = out.view(*x_shape[:-1], -1).to(dtype)
        return out

    @staticmethod
    def from_float(
        module: torch.nn.Linear,
        input_scale,
        save_device=torch.device("cpu"),
        act_quant="per-token",
    ):
        assert act_quant in [
            "per-token",
            "per-tensor",
        ], '"act_quant must be "per-token" or "per-tensor"'
        use_bias = False if module.bias is None else True
        int8_module = W8A8BFP32OFP32LinearWithQuantScale(
            module.in_features, module.out_features, use_bias, act_quant
        )
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        if act_quant == "per-token":
            alpha = weight_scale
        else:
            alpha = input_scale * weight_scale
            int8_module.quant_scale = torch.tensor(input_scale, dtype=torch.float32).to(save_device)
        int8_module.dequant_scale = alpha.to(torch.float32).to(save_device)
        int8_module.weight = int8_weight.to(save_device)
        if int8_module.use_bias:
            int8_module.bias = module.bias.to(torch.float32).to(save_device)
        return int8_module


##### fp8 linear #####
# adapt from https://github.com/neuralmagic/AutoFP8/blob/main/auto_fp8/quantize.py
# TODO(huangtingwei): replace gemm with cutlass fp8 gemm

def easy_fp8_gemm(A, A_scale, B, B_scale, bias, out_dtype):
    if A.numel() == 0:
        # Deal with empty tensors (triggeted by empty MoE experts)
        return torch.empty(size=(0, B.shape[0]), dtype=out_dtype, device=A.device)

    # Note: testing with L20/L40
    native_fp8_support = False
    if native_fp8_support:
        need_reshape = A.dim() == 3
        if need_reshape:
            batch_size = A.shape[0]
            A_input = A.reshape(-1, A.shape[-1])
        else:
            batch_size = None
            A_input = A
        output, _ = torch._scaled_mm(
            A_input,
            B.t(),
            out_dtype=out_dtype,
            scale_a=A_scale.to(A.device),
            scale_b=B_scale.to(B.device),
            bias=bias,
        )
        if need_reshape:
            output = output.reshape(
                batch_size, output.shape[0] // batch_size, output.shape[1]
            )
    else:
        output = torch.nn.functional.linear(
            A.to(out_dtype) * A_scale,
            B.to(out_dtype) * B_scale,
            bias=bias,
        )
    return output.to(out_dtype)


# per-token quant for activation
class FP8LinearDynamic(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            act_quant,
            use_bias=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act_quant = act_quant
        self.use_bias = use_bias

        self.register_buffer(
            "weight",
            torch.empty(
                self.out_features,
                self.in_features,
                dtype=torch.float8_e4m3fn,
                requires_grad=False,
            ),
        )
        if self.use_bias:
            self.register_buffer(
                "bias",
                torch.empty(self.out_features, dtype=torch.float32, requires_grad=False
                            ),
            )
        # currently only per-tensor
        self.register_buffer(
            "weight_scale", torch.tensor(1.0, dtype=torch.float32, requires_grad=False)
        )

    def _apply(self, fn):
        # prevent the bias from being converted to half
        super()._apply(fn)
        self.weight_scale = self.weight_scale.cpu()
        return self

    def forward(self, x):
        if self.act_quant == "per-token":
            qinput, x_scale = per_token_quantize_fp8(x)
        else:
            qinput, x_scale = per_tensor_quantize_fp8(x)
        # dequant have been fused in easy_fp8_gemm
        output = easy_fp8_gemm(
            A=qinput,
            A_scale=x_scale,
            B=self.weight,
            B_scale=self.weight_scale,
            bias=self.bias if self.use_bias else None,
            out_dtype=x.dtype,
        )
        return output

    @staticmethod
    def from_float(
            module: torch.nn.Linear,
            input_scale=1.0,
            save_device=torch.device("cpu"),
            act_quant="per-token",
    ):
        # dynamic scale only support per token activation quant
        assert act_quant == "per-token"
        quant_weight, weight_scale = per_tensor_quantize_fp8(module.weight)
        # assert torch.allclose(dequant_weight, module.weight, atol=tolerance)
        bias = copy.deepcopy(module.bias) if module.bias is not None else None
        alpha = input_scale * weight_scale
        # only support dynamic per token quant here
        use_bias = False if module.bias is None else True
        fp8_module = FP8LinearDynamic(
            module.in_features, module.out_features, use_bias
        )
        fp8_module.weight = quant_weight
        fp8_module.weight_scale = alpha
        if use_bias:
            fp8_module.bias = bias

        return fp8_module


class FP8StaticLinearQuantizer(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            weight: torch.Tensor,
            weight_scale: torch.Tensor,
            bias: torch.nn.Parameter,
            quantize_output: bool = False,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        self.bias = bias
        self.input_scale = None
        self.output_scale = None
        self.quantize_output = quantize_output

        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        qinput, x_input_scale = per_tensor_quantize_fp8(x)
        if self.input_scale is None:
            self.input_scale = torch.nn.Parameter(x_input_scale, requires_grad=False)
        elif x_input_scale > self.input_scale:
            self.input_scale = torch.nn.Parameter(x_input_scale, requires_grad=False)
        output = easy_fp8_gemm(
            A=qinput,
            A_scale=self.input_scale,
            B=self.weight,
            B_scale=self.weight_scale,
            bias=self.bias,
            out_dtype=x.dtype,
        )

        # Optionally, quantize output and record scale
        if self.quantize_output:
            qoutput, output_scale = per_tensor_quantize_fp8(output)
            if self.output_scale is None:
                self.output_scale = torch.nn.Parameter(output_scale, requires_grad=False)
            elif output_scale > self.output_scale:
                self.output_scale = torch.nn.Parameter(output_scale, requires_grad=False)
            output = qoutput.to(output.dtype) * output_scale

        return output


class FP8LinearStatic(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            use_bias=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        self.register_buffer(
            "weight",
            torch.empty(
                self.out_features,
                self.in_features,
                dtype=torch.float8_e4m3fn,
                requires_grad=False,
            ),
        )
        if self.use_bias:
            self.register_buffer(
                "bias",
                torch.empty(self.out_features, dtype=torch.float32, requires_grad=False
                            ),
            )
        # currently only per-tensor
        self.register_buffer(
            "weight_scale", torch.tensor(1.0, dtype=torch.float32, requires_grad=False)
        )
        # currently only per-tensor
        self.register_buffer(
            "input_scale", torch.tensor(1.0, dtype=torch.float32, requires_grad=False)
        )
        # currently only per-tensor
        self.register_buffer(
            "output_scale", torch.tensor(1.0, dtype=torch.float32, requires_grad=False)
        )

    def _apply(self, fn):
        # prevent the bias from being converted to half
        super()._apply(fn)
        self.weight_scale = self.weight_scale.cpu()
        self.input_scale = self.input_scale.cpu()
        self.output_scale = self.output_scale.cpu()
        return self

    def forward(self, x):
        qinput = static_per_tensor_quantize_fp8(x, self.input_scale)
        output = easy_fp8_gemm(
            A=qinput,
            A_scale=self.input_scale,
            B=self.weight,
            B_scale=self.weight_scale,
            bias=self.bias if self.use_bias else None,
            out_dtype=x.dtype,
        )

        if self.output_scale:
            qoutput = static_per_tensor_quantize_fp8(output, self.output_scale)
            output = qoutput.to(output.dtype) * self.output_scale

        return output

    @staticmethod
    def from_float(
            quantizer: torch.nn.Linear,
    ):
        use_bias = False if quantizer.bias is None else True
        fp8_module = FP8LinearStatic(quantizer.in_features, quantizer.out_features, use_bias)
        fp8_module.weight = quantizer.weight
        if use_bias:
            fp8_module.bias = quantizer.bias
        fp8_module.weight_scale = quantizer.weight_scale
        fp8_module.input_scale = quantizer.input_scale
        fp8_module.output_scale = quantizer.output_scale
        return fp8_module


class FP8E5M2Linear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            use_bias=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        self.register_buffer(
            "weight",
            torch.empty(
                self.out_features,
                self.in_features,
                dtype=torch.float8_e5m2,
                requires_grad=False,
            ),
        )
        if self.use_bias:
            self.register_buffer(
                "bias",
                torch.empty(self.out_features, dtype=torch.float32, requires_grad=False
                            ),
            )

    def _apply(self, fn):
        # prevent the bias from being converted to half
        super()._apply(fn)
        return self

    def forward(self, x):
        out_dtype = x.dtype
        qinput = x.to(torch.float8_e5m2)
        output, _ = torch._scaled_mm(qinput,
                                     self.weight,
                                     out_dtype=out_dtype,
                                     scale_a=None,
                                     scale_b=None,
                                     bias=self.bias)

        return output

    @staticmethod
    def from_float(
            module: torch.nn.Linear,
    ):
        quant_weight = module.weight.to(torch.float8_e5m2)
        bias = copy.deepcopy(module.bias) if module.bias is not None else None
        use_bias = False if module.bias is None else True

        fp8_module = FP8E5M2Linear(
            module.in_features, module.out_features, use_bias=use_bias
        )

        fp8_module.weight = quant_weight
        if use_bias:
            fp8_module.bias = bias

        return fp8_module