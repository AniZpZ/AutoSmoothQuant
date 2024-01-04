import torch
import math
from torch import nn
from transformers.activations import ACT2FN
from transformers.utils import logging
from typing import Optional, Tuple, List

from autosmoothquant.layers.nn.linear import W8A8BFP32OFP32LinearWithQuantScale, W8A8BFP32OFP32Linear
from autosmoothquant.thirdparty.baichuan.modeling_baichuan import (
    RMSNorm,
    MLP,
    BaichuanAttention,
    BaichuanLayer,
    BaichuanPreTrainedModel,
    BaichuanModel,
    BaichuanForCausalLM,
    NormHead
)
from autosmoothquant.thirdparty.baichuan.configuration_baichuan import BaichuanConfig

logger = logging.get_logger(__name__)

try:
    from xformers import ops as xops
except ImportError:
    xops = None
    logger.warning(
        "Xformers is not installed correctly. If you want to use memory_efficient_attention to accelerate training use the following command to install Xformers\npip install xformers."
    )

class Int8BaichuanRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.register_buffer('weight', torch.ones(hidden_size, dtype=torch.float32, requires_grad=False))
        self.epsilon = eps
    
    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.epsilon)

        # convert into half-precision
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        out = self.weight * hidden_states
        int8_out = out.round().clamp(-128, 127).to(torch.int8)
        return int8_out
    
    @staticmethod
    def from_float(module: RMSNorm,
                   output_scale: float):
        int8_norm = Int8BaichuanRMSNorm(module.weight.numel(), module.epsilon)

        int8_norm.weight.to(module.weight.dtype)
        int8_norm.weight = module.weight / output_scale

        return int8_norm


# attention is the same as opt
class Int8BaichuanAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(
        self,
        config: BaichuanConfig
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.model_max_length

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size {self.hidden_size} is not divisible by num_heads {self.num_heads}"
            )
        self.W_pack = W8A8BFP32OFP32Linear(self.hidden_size, 3 * self.num_heads * self.head_dim)
        self.o_proj = W8A8BFP32OFP32LinearWithQuantScale(self.num_heads * self.head_dim, self.hidden_size)
    
    _shape = BaichuanAttention._shape
    
    @staticmethod
    @torch.no_grad()
    def from_float(module: BaichuanAttention,
                   config: BaichuanConfig,
                   attn_input_scale: float,
                   attn_output_scale: float,
                   out_input_scale: float):
        int8_module = Int8BaichuanAttention(config)
        
        # we do not impelement attn for now bacuase we want to use paged attention
        int8_module.W_pack = W8A8BFP32OFP32Linear.from_float(
          module.W_pack, attn_input_scale)
        int8_module.o_proj = W8A8BFP32OFP32LinearWithQuantScale.from_float(
            module.o_proj, out_input_scale)
        return int8_module

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        proj = self.W_pack(hidden_states).to(torch.float16)
        proj = (
            proj.unflatten(-1, (3, self.hidden_size))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
        )
        query_states = (
            proj[0].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        )
        key_states = (
            proj[1].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        )
        value_states = (
            proj[2].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None
        if xops is not None and self.training:
            attn_weights = None
            # query_states = query_states.transpose(1, 2)
            # key_states = key_states.transpose(1, 2)
            # value_states = value_states.transpose(1, 2)
            # attn_output = xops.memory_efficient_attention(
            #     query_states, key_states, value_states, attn_bias=attention_mask
            # )
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
                attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask = attention_mask)
            attn_output = attn_output.transpose(1, 2)
        else:
            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                if q_len == 1:  # inference with cache
                    if len(attention_mask.size()) == 4:
                        attention_mask = attention_mask[:, :, -1:, :]
                    else:
                        attention_mask = attention_mask[:, -1:, :]
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )

            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, value_states)

            attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class Int8BaichuanMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.intermediate_size)
        self.down_proj = W8A8BFP32OFP32LinearWithQuantScale(self.intermediate_size, self.hidden_size)
        self.up_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.intermediate_size)
        self.act_fn = ACT2FN[hidden_act]
    
    @staticmethod
    @torch.no_grad()
    def from_float(module: MLP,
                   config: BaichuanConfig,
                   gate_input_scale: float,
                   down_input_scale: float):
        int8Mlp = Int8BaichuanMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )

        # FIXME: Fuse the scaling into the q_proj output scale
        linearList = [module.gate_proj, module.up_proj]
        gateup_list = W8A8BFP32OFP32Linear.from_float_fuse(
            linearList, 
            gate_input_scale)

        if len(gateup_list) != 2:
            raise ValueError(
                f"invalid qkv gateup len, must return 2 linears but get {len(qkv_list)}")

        int8Mlp.gate_proj = gateup_list[0]
        int8Mlp.up_proj = gateup_list[1]
        int8Mlp.down_proj = W8A8BFP32OFP32LinearWithQuantScale.from_float(
            module.down_proj, 
            down_input_scale)

        return int8Mlp
        
    def forward(self, x):
        # TODO: supprot self.config.pretraining_tp > 1 condition, adapt from transformer.modeling_llama
        hidden = self.act_fn(self.gate_proj(x).to(torch.float16))
        hidden = hidden * self.up_proj(x)
        return self.down_proj(hidden)


class Int8BaichuanLayer(nn.Module):
    def __init__(self, config: BaichuanConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Int8BaichuanAttention(config=config)
        self.mlp = Int8BaichuanMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = Int8BaichuanRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Int8BaichuanRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
    
    @staticmethod
    def from_float(module: BaichuanLayer,
                   config: BaichuanConfig,
                   attn_input_scale: float,
                   attn_output_scale: float,
                   out_input_scale: float,
                   gate_input_scale: float,
                   down_input_scale: float
                   ):
        int8_module = Int8BaichuanLayer(
            config
        )

        int8_module.self_attn = Int8BaichuanAttention.from_float(
            module.self_attn, 
            config,
            attn_input_scale,
            attn_output_scale,
            out_input_scale
        )
        
        int8_module.mlp = Int8BaichuanMLP.from_float(
            module.mlp, 
            config,
            gate_input_scale,
            down_input_scale
        )
        int8_module.input_layernorm = Int8BaichuanRMSNorm.from_float(
            module.input_layernorm,
            attn_input_scale
        )
        int8_module.post_attention_layernorm = Int8BaichuanRMSNorm.from_float(
            module.post_attention_layernorm,
            gate_input_scale
        )
        return int8_module
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        residual.add_(hidden_states.to(residual.dtype))
        # Fully Connected
        hidden_states = self.post_attention_layernorm(residual)
        hidden_states = self.mlp(hidden_states)
        residual.add_(hidden_states.to(residual.dtype))
        outputs = (residual,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class Int8BaichuanModel(BaichuanPreTrainedModel):
    def __init__(self, config: BaichuanConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.n_head = config.num_attention_heads
        self.embed_tokens = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = torch.nn.ModuleList(
            [BaichuanLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)

        self.gradient_checkpointing = config.gradient_checkpointing
        self.post_init()
        self.max_cache_pos = config.model_max_length
        self.first_run = True
        self.alibi_mask = None
    
    @staticmethod
    def from_float(module, decoder_layer_scales):
        int8_module = Int8BaichuanModel(module.config)
        
        int8_module.embed_tokens = module.embed_tokens
        int8_module.norm = module.norm
        
        for i, layer in enumerate(module.layers):
            int8_module.layers[i] = Int8BaichuanLayer.from_float(
                layer, module.config, **decoder_layer_scales[i])
        return int8_module
    
    get_input_embeddings = BaichuanModel.get_input_embeddings
    set_input_embeddings = BaichuanModel.set_input_embeddings
    get_alibi_mask = BaichuanModel.get_alibi_mask
    forward = BaichuanModel.forward

class Int8BaichuanForCausalLM(BaichuanPreTrainedModel):
    def __init__(self, config, *model_args, **model_kwargs):
        super().__init__(config, *model_args, **model_kwargs)
        self.model = BaichuanModel(config)
        self.lm_head = NormHead(config.hidden_size, config.vocab_size, bias=False)
        #if hasattr(config, "quantization_config") and config.quantization_config['load_in_4bit']:
        if hasattr(config, "quantization_config") and isinstance(config.quantization_config, dict) and config.quantization_config.get('load_in_4bit', False):
            try:
                from .quantizer import quantize_offline, init_model_weight_int4
            except ImportError:
                raise ImportError(f"Needs quantize_offline to run quantize.")
            quantize_offline(self, 4)
        # Initialize weights and apply final processing
        self.post_init()
    
    @staticmethod
    def from_float(module, decoder_layer_scales):
        int8_module = Int8BaichuanForCausalLM(module.config)
        print("start trans into int8, this might take a while")
        int8_module.model = Int8BaichuanModel.from_float(
            module.model, decoder_layer_scales)
        int8_module.lm_head = module.lm_head
        return int8_module