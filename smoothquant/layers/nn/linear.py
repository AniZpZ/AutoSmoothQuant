import torch
import threading
from ..functional.quantization import quantize_per_tensor_absmax
from auto_smoothquant._CUDA import I8CUGEMM
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
    def __init__(self,
                 in_features,
                 out_features,
                 use_bias=True,
                 act_per_token_quant=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.act_per_token_quant = act_per_token_quant
        GEMM = Int8GEMM()
        self.i8cugemm = GEMM.get_i8cugemm()

        self.register_buffer(
            'weight',
            torch.empty(self.out_features,
                        self.in_features,
                        dtype=torch.int8,
                        requires_grad=False))
        if self.use_bias:
            self.register_buffer(
                'bias',
                torch.zeros((1, self.out_features),
                            dtype=torch.float32,
                            requires_grad=False))
        self.register_buffer(
            'dequant_scale',
            torch.tensor(1.0, dtype=torch.float32, requires_grad=False))

    def _apply(self, fn):
        # prevent the bias from being converted to half
        super()._apply(fn)
        self.dequant_scale = self.dequant_scale.cpu()
        # if self.use_bias:
        #     self.bias = self.bias.to(torch.float32)
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # self.weight = self.weight.to(*args, **kwargs)
        self.dequant_scale = self.dequant_scale.to(*args, **kwargs)
        self.dequant_scale = self.dequant_scale.to(torch.float32)
        if self.use_bias:
            self.bias = self.bias.to(*args, **kwargs)
            self.bias = self.bias.to(torch.float32)
        return self

    @torch.no_grad()
    def forward(self, x, act_scale=None):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        if self.act_per_token_quant and act_scale is not None:
            dequant_scale = self.dequant_scale.item() * act_scale
            dequant_scale = dequant_scale.view(x_shape[0], -1)
        else:
            dequant_scale = self.dequant_scale.item()
        x = x.round().clamp(-128, 127).to(torch.int8)
        out = torch.empty(x_shape[0],
                          self.out_features,
                          dtype=torch.int32,
                          device=torch.cuda.current_device())
        y = self.i8cugemm.linear_a8_w8_o32_(x, self.weight, out)
        y = dequant_scale * y + self.bias if self.use_bias else dequant_scale * y
        y = y.view(*x_shape[:-1], -1)
        return y

    @staticmethod
    def from_float(module: torch.nn.Linear, input_scale, act_per_token_quant=False):
        use_bias = False if module.bias is None else True
        int8_module = W8A8BFP32OFP32Linear(module.in_features,
                                           module.out_features, use_bias,
                                           act_per_token_quant)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        alpha = weight_scale if act_per_token_quant else input_scale * weight_scale
        int8_module.dequant_scale = torch.tensor(alpha,
                                                 dtype=torch.float32,
                                                 requires_grad=False)
        int8_module.weight = int8_weight
        if int8_module.use_bias:
            int8_module.bias = module.bias.to(torch.float32)
            int8_module.bias.requires_grad = False
        return int8_module


class W8A8BFP32OFP32QKVLinear(W8A8BFP32OFP32Linear):
    # for fused qkv weight
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._buffers.pop('dequant_scale')
        self.register_buffer(
            'q_dequant_scale',
            torch.tensor(1.0, dtype=torch.float32, requires_grad=False))
        self.register_buffer(
            'k_dequant_scale',
            torch.tensor(1.0, dtype=torch.float32, requires_grad=False))
        self.register_buffer(
            'v_dequant_scale',
            torch.tensor(1.0, dtype=torch.float32, requires_grad=False))

    def _apply(self, fn):
        # prevent the bias from being converted to half
        super(W8A8BFP32OFP32Linear, self)._apply(fn)
        self.q_dequant_scale = self.q_dequant_scale.cpu()
        self.k_dequant_scale = self.k_dequant_scale.cpu()
        self.v_dequant_scale = self.v_dequant_scale.cpu()
        # self.bias = self.bias.to(torch.float32)
        return self

    def to(self, *args, **kwargs):
        super(W8A8BFP32OFP32Linear, self).to(*args, **kwargs)
        # self.weight = self.weight.to(*args, **kwargs)
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
    def forward(self, x, qkv_size, act_scale=None):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        if self.act_per_token_quant and act_scale is not None:
            q_dequant_scale = self.q_dequant_scale.item() * act_scale
            q_dequant_scale = q_dequant_scale.view(x_shape[0], -1)
            k_dequant_scale = self.k_dequant_scale.item() * act_scale
            k_dequant_scale = k_dequant_scale.view(x_shape[0], -1)
            v_dequant_scale = self.v_dequant_scale.item() * act_scale
            v_dequant_scale = v_dequant_scale.view(x_shape[0], -1)
        else:
            q_dequant_scale = self.q_dequant_scale.item()
            k_dequant_scale = self.k_dequant_scale.item()
            v_dequant_scale = self.v_dequant_scale.item()
        x = x.round().clamp(-128, 127).to(torch.int8)
        out = torch.empty(x_shape[0],
                          self.out_features,
                          dtype=torch.int32,
                          device=torch.cuda.current_device())
        y = self.i8cugemm.linear_a8_w8_o32_(x, self.weight, out)
        q, k, v = y.split(qkv_size, dim=-1)
        q_dq = q_dequant_scale * q
        k_dq = k_dequant_scale * k
        v_dq = v_dequant_scale * v
        if self.use_bias:
            q_bias, k_bias, v_bias = self.bias.split(qkv_size, dim=-1)
            q_dq += q_bias
            k_dq += k_bias
            v_dq += v_bias
        y = torch.cat([q_dq, k_dq, v_dq], dim=-1)
        y = y.view(*x_shape[:-1], -1)
        return y

    @staticmethod
    def from_float(module: torch.nn.Linear,
                   input_scale,
                   qkv_size,
                   act_per_token_quant=False):
        use_bias = False if module.bias is None else True
        int8_module = W8A8BFP32OFP32QKVLinear(module.in_features,
                                              module.out_features, use_bias,
                                              act_per_token_quant)
        q_weight, k_weight, v_weight = module.weight.data.split(qkv_size,
                                                                dim=0)
        q_int8_weight, q_weight_scale = quantize_per_tensor_absmax(q_weight)
        k_int8_weight, k_weight_scale = quantize_per_tensor_absmax(k_weight)
        v_int8_weight, v_weight_scale = quantize_per_tensor_absmax(v_weight)
        int8_module.weight.data = torch.cat(
            [q_int8_weight, k_int8_weight, v_int8_weight], dim=0)
        alpha = q_weight_scale
        beta = k_weight_scale
        gamma = v_weight_scale
        if not act_per_token_quant:
            alpha = q_weight_scale * input_scale
            beta = k_weight_scale * input_scale
            gamma = v_weight_scale * input_scale
        int8_module.q_dequant_scale = torch.tensor(alpha,
                                                   dtype=torch.float32,
                                                   requires_grad=False)
        int8_module.k_dequant_scale = torch.tensor(beta,
                                                   dtype=torch.float32,
                                                   requires_grad=False)
        int8_module.v_dequant_scale = torch.tensor(gamma,
                                                   dtype=torch.float32,
                                                   requires_grad=False)
        if int8_module.use_bias:
            int8_module.bias = module.bias.to(torch.float32)
            int8_module.bias.requires_grad = False
        return int8_module

class W8A8BFP32OFP32LinearWithQuantScale(W8A8BFP32OFP32Linear):
    # For fc2 and out_proj
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.act_per_token_quant:
            self.register_buffer(
                'quant_scale',
                torch.tensor(1.0, dtype=torch.float32, requires_grad=False))

    def _apply(self, fn):
        # prevent the bias from being converted to half
        super(W8A8BFP32OFP32Linear, self)._apply(fn)
        self.dequant_scale = self.dequant_scale.cpu()
        if not self.act_per_token_quant:
            self.quant_scale = self.quant_scale.cpu()
        # self.bias = self.bias.to(torch.float32)
        return self

    def to(self, *args, **kwargs):
        super(W8A8BFP32OFP32Linear, self).to(*args, **kwargs)
        # self.weight = self.weight.to(*args, **kwargs)
        if self.use_bias:
            self.bias = self.bias.to(*args, **kwargs)
            self.bias = self.bias.to(torch.float32)
        self.dequant_scale = self.dequant_scale.to(*args, **kwargs)
        self.dequant_scale = self.dequant_scale.to(torch.float32)
        if not self.act_per_token_quant:
            self.quant_scale = self.quant_scale.to(*args, **kwargs)
            self.quant_scale = self.quant_scale.to(torch.float32)
        return self

    @torch.no_grad()
    def forward(self, x, act_scale=None):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        if self.act_per_token_quant and act_scale is not None:
            dequant_scale = self.dequant_scale.item() * act_scale
            dequant_scale = dequant_scale.view(x_shape[0], -1)
            quant_scale = x.abs().max(dim=-1, keep_dim=True)[0].div(127.0)
        else:
            dequant_scale = self.dequant_scale.item()
            quant_scale = self.quant_scale.item()
        # quant here
        x = (x / quant_scale).round().clamp(-128, 127).to(torch.int8)
        out = torch.empty(x_shape[0],
                          self.out_features,
                          dtype=torch.int32,
                          device=torch.cuda.current_device())
        y = self.i8cugemm.linear_a8_w8_o32_(x, self.weight, out)
        y = dequant_scale * y + self.bias if self.use_bias else dequant_scale * y
        y = y.view(*x_shape[:-1], -1)
        return y

    @staticmethod
    def from_float(module: torch.nn.Linear, input_scale, act_per_token_quant=False):
        use_bias = False if module.bias is None else True
        int8_module = W8A8BFP32OFP32LinearWithQuantScale(
            module.in_features, module.out_features, use_bias, act_per_token_quant)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        if act_per_token_quant:
            alpha = weight_scale
        else:
            alpha = input_scale * weight_scale
            int8_module.quant_scale = torch.tensor(input_scale,
                                                   dtype=torch.float32,
                                                   requires_grad=False)
        int8_module.dequant_scale = torch.tensor(alpha,
                                                 dtype=torch.float32,
                                                 requires_grad=False)
        int8_module.weight = int8_weight
        if int8_module.use_bias:
            int8_module.bias = module.bias.to(torch.float32)
            int8_module.bias.requires_grad = False
        return int8_module
