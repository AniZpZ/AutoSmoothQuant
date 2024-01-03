import torch
import math
from torch import nn
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaMLP,
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaPreTrainedModel,
    LlamaModel,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    BaseModelOutputWithPast
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.activations import SiLUActivation
from typing import Optional, Tuple, List
from autosmoothquant.layers.nn.linear import W8A8BFP32OFP32LinearWithQuantScale, W8A8BFP32OFP32Linear
from transformers.utils import logging
logger = logging.get_logger(__name__)

# we keep scale in LlamaRMSNorm layer for kernel fusion
class Int8LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.register_buffer('weight', torch.ones(hidden_size, dtype=torch.float32, requires_grad=False))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        out = self.weight * hidden_states
        int8_out = out.round().clamp(-128, 127).to(torch.int8)
        return int8_out
    
    @staticmethod
    def from_float(module: LlamaRMSNorm,
                   output_scale: float):
        int8_module = Int8LlamaRMSNorm(module.weight.numel(), module.variance_epsilon)

        int8_module.weight.to(module.weight.dtype)
        int8_module.weight = module.weight / output_scale

        return int8_module

_RMSNorm = {
    "per-tensor": Int8LlamaRMSNorm,
    "per-token": LlamaRMSNorm
}
class Int8LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(
        self,
        config: LlamaConfig,
        act_quant_map: dict[str, str]
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.out_input_scale = 0.
        # hidden_size is embed_dim in OptAttetion
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.qkv_quant_type = act_quant_map["qkv_proj"]
        self.o_quant_type = act_quant_map["o_proj"]
        self.k_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.num_heads * self.head_dim, act_quant=self.qkv_quant_type)
        self.v_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.num_heads * self.head_dim, act_quant=self.qkv_quant_type)
        self.q_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.num_heads * self.head_dim, act_quant=self.qkv_quant_type)
        # out is fp32
        self.o_proj = W8A8BFP32OFP32LinearWithQuantScale(self.num_heads * self.head_dim, self.hidden_size, act_quant=self.o_quant_type)

        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
    
    _shape = LlamaAttention._shape
    
    @staticmethod
    @torch.no_grad()
    def from_float(module: LlamaAttention,
                   config: LlamaConfig,
                   act_quant_map: dict[str, str],
                   attn_input_scale: float,
                   q_output_scale: float,
                   k_output_scale: float,
                   v_output_scale: float,
                   out_input_scale: float):
        int8_module = Int8LlamaAttention(config, act_quant_map)
        # we do not impelement attn for now bacuase we want use paged attention
        # act_quant
        int8_module.q_proj = W8A8BFP32OFP32Linear.from_float(module.q_proj, attn_input_scale, act_quant=int8_module.qkv_quant_type)
        int8_module.k_proj = W8A8BFP32OFP32Linear.from_float(module.k_proj, attn_input_scale, act_quant=int8_module.qkv_quant_type)
        int8_module.v_proj = W8A8BFP32OFP32Linear.from_float(module.v_proj, attn_input_scale, act_quant=int8_module.qkv_quant_type)
        int8_module.o_proj = W8A8BFP32OFP32LinearWithQuantScale.from_float(
            module.o_proj, out_input_scale, act_quant=int8_module.o_quant_type)
        return int8_module
    
    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        # already quant before attention
        query_states = self.q_proj(hidden_states).to(torch.float16)
        key_states = self.k_proj(hidden_states).to(torch.float16)
        value_states = self.v_proj(hidden_states).to(torch.float16)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if use_cache else None
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        # quant method from torch-int
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value
    

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Int8LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig, act_quant_map: dict[str, str]):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.down_input_scale = 0.
        self.gate_up_quant_type = act_quant_map["gate_up_proj"]
        self.down_quant_type = act_quant_map["down_proj"]
        # need fp32 out bcause silu
        self.gate_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.intermediate_size, act_quant=self.gate_up_quant_type)

        self.up_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.intermediate_size, act_quant=self.gate_up_quant_type)
        self.down_proj = W8A8BFP32OFP32LinearWithQuantScale(self.intermediate_size, self.hidden_size, act_quant=self.down_quant_type)
        # silu_and_mul_kernel in vLLM can be a reference of SwiGLU
        self.act_fn = SiLUActivation()
    
    @staticmethod
    @torch.no_grad()
    def from_float(module: LlamaMLP,
                   config: LlamaConfig,
                   act_quant_map: dict[str, str],
                   gate_input_scale: float,
                   gate_output_scale: float,
                   up_input_scale: float,
                   up_output_scale: float,
                   down_input_scale: float,
                   down_output_scale: float):
        int8_module = Int8LlamaMLP(config, act_quant_map)
        # act_quant
        int8_module.gate_proj = W8A8BFP32OFP32Linear.from_float(module.gate_proj, gate_input_scale, act_quant=int8_module.gate_up_quant_type)
        int8_module.up_proj = W8A8BFP32OFP32Linear.from_float(module.up_proj, up_input_scale, act_quant=int8_module.gate_up_quant_type)
        int8_module.down_proj = W8A8BFP32OFP32LinearWithQuantScale.from_float(
            module.down_proj, 
            down_input_scale,
            act_quant=int8_module.down_quant_type)
        return int8_module
        
    def forward(self, x):
        # TODO: supprot self.config.pretraining_tp > 1 condition, adapt from transformer.modeling_llama
        hidden = self.act_fn(self.gate_proj(x).to(torch.float16))
        hidden = hidden * self.up_proj(x)
        return self.down_proj(hidden)

class Int8LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, act_quant_map: dict[str, str]):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Int8LlamaAttention(config, act_quant_map)
        self.mlp = Int8LlamaMLP(config, act_quant_map)
        #FIXME: use int8 rmsnorm
        input_layernorm_cls = _RMSNorm[act_quant_map["qkv_proj"]]
        post_attention_layernorm_cls = _RMSNorm[act_quant_map["gate_up_proj"]]
        self.input_layernorm = input_layernorm_cls(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = post_attention_layernorm_cls(self.hidden_size, eps=config.rms_norm_eps)

    @staticmethod
    def from_float(module: LlamaDecoderLayer,
                   config: LlamaConfig,
                   act_quant_map: dict[str, str],
                   attn_input_scale: float,
                   q_output_scale: float,
                   k_output_scale: float,
                   v_output_scale: float,
                   out_input_scale: float,
                   gate_input_scale: float,
                   up_input_scale: float,
                   down_input_scale: float,
                   gate_output_scale: float,
                   up_output_scale: float,
                   down_output_scale: float
                   ):
        int8_module = Int8LlamaDecoderLayer(
            config,
            act_quant_map
        )

        int8_module.self_attn = Int8LlamaAttention.from_float(
            module.self_attn, 
            config,
            act_quant_map,
            attn_input_scale,
            q_output_scale,
            k_output_scale,
            v_output_scale,
            out_input_scale
        )
        
        int8_module.mlp = Int8LlamaMLP.from_float(
            module.mlp, 
            config,
            act_quant_map,
            gate_input_scale,
            gate_output_scale,
            up_input_scale,
            up_output_scale,
            down_input_scale,
            down_output_scale
        )
        if act_quant_map["qkv_proj"] == "per-tensor":
            int8_module.input_layernorm = Int8LlamaRMSNorm.from_float(
                module.input_layernorm,
                attn_input_scale
            )
        else:
            int8_module.input_layernorm = module.input_layernorm
        if act_quant_map["gate_up_proj"] == "per-tensor":
            int8_module.post_attention_layernorm = Int8LlamaRMSNorm.from_float(
                module.post_attention_layernorm,
                gate_input_scale
            )
        else:
            int8_module.post_attention_layernorm = module.post_attention_layernorm
        return int8_module
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        *args,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        residual.add_(hidden_states.to(residual.dtype))
        
        # mlp
        hidden_states = self.post_attention_layernorm(residual)
        hidden_states = self.mlp(hidden_states)
        residual.add_(hidden_states.to(residual.dtype))
        outputs = (residual,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs

class Int8LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig, act_quant_map: dict[str, str]):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Int8LlamaDecoderLayer(config, act_quant_map) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
    
    get_input_embeddings = LlamaModel.get_input_embeddings
    set_input_embeddings = LlamaModel.set_input_embeddings
    _prepare_decoder_attention_mask = LlamaModel._prepare_decoder_attention_mask
    # iter self.layers and calcu forward
    forward = LlamaModel.forward
    
    @staticmethod
    def from_float(module, decoder_layer_scales, act_quant_map):
        int8_module = Int8LlamaModel(module.config, act_quant_map)
        
        int8_module.embed_tokens = module.embed_tokens
        int8_module.norm = module.norm
        
        for i, layer in enumerate(module.layers):
            int8_module.layers[i] = Int8LlamaDecoderLayer.from_float(
                layer, module.config, act_quant_map, **decoder_layer_scales[i])
        return int8_module

class Int8LlamaForCausalLM(LlamaPreTrainedModel):
    def __init__(self, config, act_quant_map):
        super().__init__(config)
        self.config = config
        self.model = Int8LlamaModel(config, act_quant_map)
        # no need to quant
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()
    
    @staticmethod
    def from_float(module, decoder_layer_scales, act_quant_map):
        int8_module = Int8LlamaForCausalLM(module.config, act_quant_map)
        print("start trans into int8, this might take a while")
        int8_module.model = Int8LlamaModel.from_float(
            module.model, decoder_layer_scales, act_quant_map)
        int8_module.lm_head = module.lm_head
        return int8_module
    
    get_input_embeddings = LlamaForCausalLM.get_input_embeddings
    set_input_embeddings = LlamaForCausalLM.set_input_embeddings
    get_output_embeddings = LlamaForCausalLM.get_output_embeddings
    set_output_embeddings = LlamaForCausalLM.set_output_embeddings
    set_decoder = LlamaForCausalLM.set_decoder
    get_decoder = LlamaForCausalLM.get_decoder
    forward = LlamaForCausalLM.forward
    prepare_inputs_for_generation = LlamaForCausalLM.prepare_inputs_for_generation
    _reorder_cache = LlamaForCausalLM._reorder_cache
