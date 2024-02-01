import torch
import threading
from ..functional.quantization import quantize_per_tensor_absmax
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
        input_scale,
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
