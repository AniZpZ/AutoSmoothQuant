import torch
import numpy as np
import transformers
import gc
import re
from typing import Optional, Tuple

##### Int8 quantization #####
@torch.no_grad()
def quantize_per_tensor_absmax(t):
    scale = t.abs().max() / 127
    if not t.is_cuda:
        # half rounding is not supported on CPU
        t = t.float()
    # use inplace operation to save memory
    t.div_(scale).round_()
    t_q = t.to(torch.int8)
    return t_q, scale

# for weights with same shape to share a common scale
@torch.no_grad()
def quantize_fused_tensor_absmax(t):
    scale = 0
    for linear in t:
        scale = max(linear.weight.abs().max() / 127, scale)
    weight_list = []
    for linear in t:
        cweight = linear.weight
        if not cweight.is_cuda:
            # half rounding is not supported on CPU
            cweight = cweight.float()
        # use inplace operation to save memory
        cweight.div_(scale).round_()
        t_q = cweight.to(torch.int8)
        weight_list.append(t_q)

    return weight_list, scale

@torch.no_grad()
def quantize_weight_per_channel_absmax(w):
    # w: [out_channel, in_channel]
    scales = w.abs().max(dim=1)[0] / 127
    scales = scales.view(-1, 1)
    if not w.is_cuda:
        # half rounding is not supported on CPU
        w = w.float()
    # use inplace operation to save memory
    w.div_(scales).round_().clamp_(-128, 127)
    w_q = w.to(torch.int8)
    return w_q, scales


@torch.no_grad()
def dynamic_quantize_activation_per_tensor_zeropoint(t):
    max_val = t.max()[0]
    min_val = t.min()[0]
    quant_min = -127
    quant_max = 127
    nudged_scale = (max_val - min_val) / (quant_max - quant_min)
    zp = (max_val + min_val) / 2
    zp = (zp / nudged_scale).round() * nudged_scale
    t -= zp
    max_val = (max_val - min_val) / 2

    max_val = torch.clamp(max_val, min=1e-8) / 127
    q_act = (t / max_val).round().clamp(-128, 127).to(torch.int8)
    return q_act, max_val, zp


@torch.no_grad()
def dynamic_quantize_activation_per_tensor_absmax(t):
    max_val = t.abs().max()
    max_val = torch.clamp(max_val, min=1e-8) / 127
    q_act = (t / max_val).round().clamp(-128, 127).to(torch.int8)
    return q_act, max_val


@torch.no_grad()
def dynamic_quantize_activation_per_token_absmax(t):
    max_val = t.abs().max(dim=-1, keepdim=True)[0]
    max_val = torch.clamp(max_val, min=1e-8) / 127
    t.div_(max_val).round_().clamp_(-128, 127)
    q_act = t.to(torch.int8)
    return q_act, max_val

@torch.no_grad()
def fake_quantize_activation_per_tensor_absmax(t):
    max_val = t.abs().max()
    max_val = torch.clamp(max_val, min=1e-8) / 127
    t.div_(max_val).round_().clamp_(-128, 127).mul_(max_val)
    return t


@torch.no_grad()
def fake_quantize_activation_per_token_absmax(t):
    max_val = t.abs().max(dim=-1, keepdim=True)[0]
    max_val = torch.clamp(max_val, min=1e-8) / 127
    t.div_(max_val).round_().clamp_(-128, 127).mul_(max_val)
    return t


@torch.no_grad()
def dequantize_activation_w_per_channel_a_per_token(q_act, w_scales, a_scales):
    # q_act: [B, dim]
    # w_scales: [dim]
    # a_scales: [B 1]
    dtype = a_scales.dtype
    q_act = q_act.to(torch.float32)
    q_act.mul_(w_scales.reshape(1, -1)).mul_(a_scales.reshape(-1, 1))
    return q_act.to(dtype)

@torch.no_grad()
def dequantize_activation_w_per_channel_a_per_tensor(q_act, w_scales, a_scales):
    # q_act: [..., dim]
    # w_scales: [dim]
    # a_scales: [1]
    dtype = a_scales.dtype
    q_act = q_act.to(torch.float32)
    q_act = q_act * w_scales.reshape(1, -1) * a_scales
    return q_act.to(dtype)


##### FP8 quantization #####
# adapt from https://github.com/neuralmagic/AutoFP8/blob/main/auto_fp8/quantize.py
# TODO(huangtingwei): support fine grained quantization
def new_dtype_byte_size(dtype):
    if dtype == torch.bool:
        return 1 / 8
    bit_search = re.search(r"[^\d](\d+)_?", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8


transformers.modeling_utils.dtype_byte_size = new_dtype_byte_size


def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()


def per_tensor_quantize_fp8(tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Quantize a tensor using per-tensor static scaling factor.
    Args:
        tensor: The input tensor.
    """
    finfo = torch.finfo(torch.float8_e4m3fn)
    # Calculate the scale as dtype max divided by absmax.
    # Since .abs() creates a new tensor, we use aminmax to get
    # the min and max first and then calculate the absmax.
    if tensor.numel() == 0:
        # Deal with empty tensors (triggered by empty MoE experts)
        min_val, max_val = (
            torch.tensor(-16.0, dtype=tensor.dtype),
            torch.tensor(16.0, dtype=tensor.dtype),
        )
    else:
        min_val, max_val = tensor.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs())
    scale = amax / finfo.max
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    qweight = (tensor / scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    qweight = qweight.to(torch.float8_e4m3fn)
    return qweight, scale


def per_token_quantize_fp8(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor using per-token scaling factor.
    Args:
        tensor: The input tensor.
    """
    finfo = torch.finfo(torch.float8_e4m3fn)
    # Calculate the scale as dtype max divided by absmax.
    # Since .abs() creates a new tensor, we use aminmax to get
    # the min and max first and then calculate the absmax.
    assert tensor.numel() > 0
    scale = tensor.abs().max(dim=-1, keepdim=True)[0].div(finfo.max).to(torch.float32)
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    qweight = (tensor / scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    qweight = qweight.to(torch.float8_e4m3fn)
    return qweight, scale


def fake_per_tensor_quantize_fp8(tensor: torch.Tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    assert tensor.numel() > 0
    scale = tensor.abs().max().div(finfo.max).to(torch.float32)
    qweight = (tensor / scale).clamp(min=finfo.min, max=finfo.max).mul(scale)
    return qweight

def fake_per_token_quantize_fp8(tensor: torch.Tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    assert tensor.numel() > 0
    scale = tensor.abs().max(dim=-1, keepdim=True)[0].div(finfo.max).to(torch.float32)
    qweight = (tensor / scale).clamp(min=finfo.min, max=finfo.max).mul(scale)
    return qweight

def static_per_tensor_quantize_fp8(tensor: torch.Tensor, inv_scale: float) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    qweight = (tensor / inv_scale).clamp(min=finfo.min, max=finfo.max)
    return qweight.to(torch.float8_e4m3fn)
