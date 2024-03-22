import torch
from torch import nn
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaMLP,
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaPreTrainedModel,
    LlamaModel,
    LlamaForCausalLM,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.activations import ACT2FN
from typing import Optional
from autosmoothquant.layers.nn.linear import W8A8BFP32OFP32LinearWithQuantScale, W8A8BFP32OFP32Linear
from transformers.utils import logging
logger = logging.get_logger(__name__)

class Int8LlamaRMSNorm(LlamaRMSNorm):
    
    @staticmethod
    def from_float(module: LlamaRMSNorm,
                   output_scale: float):
        int8_module = Int8LlamaRMSNorm(module.weight.numel(), module.variance_epsilon)

        int8_module.weight.to(module.weight.dtype)
        int8_module.weight = nn.Parameter(module.weight / output_scale)

        return int8_module

class Int8LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: dict[str, str],
        layer_idx: Optional[int] = None
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        self.qkv_quant_type = quant_config["qkv"]
        self.o_quant_type = quant_config["out"]
        self.k_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.num_heads * self.head_dim, act_quant=self.qkv_quant_type)
        self.v_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.num_heads * self.head_dim, act_quant=self.qkv_quant_type)
        self.q_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.num_heads * self.head_dim, act_quant=self.qkv_quant_type)
        self.o_proj = W8A8BFP32OFP32LinearWithQuantScale(self.num_heads * self.head_dim, self.hidden_size, act_quant=self.o_quant_type)
        self._init_rope()
    
    _init_rope = LlamaAttention._init_rope
    _shape = LlamaAttention._shape
    forward = LlamaAttention.forward
    
    @staticmethod
    @torch.no_grad()
    def from_float(module: LlamaAttention,
                   config: LlamaConfig,
                   quant_config: dict[str, str],
                   attn_input_scale: float,
                   q_output_scale: float,
                   k_output_scale: float,
                   v_output_scale: float,
                   out_input_scale: float):
        int8_module = Int8LlamaAttention(config, quant_config)
        # we do not impelement attn for now bacuase we want use paged attention
        int8_module.q_proj = W8A8BFP32OFP32Linear.from_float(module.q_proj, attn_input_scale, act_quant=int8_module.qkv_quant_type)
        int8_module.k_proj = W8A8BFP32OFP32Linear.from_float(module.k_proj, attn_input_scale, act_quant=int8_module.qkv_quant_type)
        int8_module.v_proj = W8A8BFP32OFP32Linear.from_float(module.v_proj, attn_input_scale, act_quant=int8_module.qkv_quant_type)
        int8_module.o_proj = W8A8BFP32OFP32LinearWithQuantScale.from_float(
            module.o_proj, out_input_scale, act_quant=int8_module.o_quant_type)
        return int8_module
    
class Int8LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig, quant_config: dict[str, str]):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_quant_type = quant_config["fc1"]
        self.down_quant_type = quant_config["fc2"]
        self.gate_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.intermediate_size, act_quant=self.gate_up_quant_type)
        self.up_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.intermediate_size, act_quant=self.gate_up_quant_type)
        self.down_proj = W8A8BFP32OFP32LinearWithQuantScale(self.intermediate_size, self.hidden_size, act_quant=self.down_quant_type)
        self.act_fn = ACT2FN[config.hidden_act]
    
    forward = LlamaMLP.forward

    @staticmethod
    @torch.no_grad()
    def from_float(module: LlamaMLP,
                   config: LlamaConfig,
                   quant_config: dict[str, str],
                   gate_input_scale: float,
                   down_input_scale: float):
        int8_module = Int8LlamaMLP(config, quant_config)
        int8_module.gate_proj = W8A8BFP32OFP32Linear.from_float(module.gate_proj, gate_input_scale, act_quant=int8_module.gate_up_quant_type)
        int8_module.up_proj = W8A8BFP32OFP32Linear.from_float(module.up_proj, gate_input_scale, act_quant=int8_module.gate_up_quant_type)
        int8_module.down_proj = W8A8BFP32OFP32LinearWithQuantScale.from_float(
            module.down_proj, 
            down_input_scale,
            act_quant=int8_module.down_quant_type)
        return int8_module
    
class Int8LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, quant_config: dict[str, str], layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        # only support LlamaAttention for now. TODO: support LlamaFlashAttention2 and LlamaSdpaAttention
        self.self_attn = Int8LlamaAttention(config, quant_config, layer_idx)
        self.mlp = Int8LlamaMLP(config, quant_config)
        self.input_layernorm = Int8LlamaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Int8LlamaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    forward = LlamaDecoderLayer.forward

    @staticmethod
    def from_float(module: LlamaDecoderLayer,
                   config: LlamaConfig,
                   quant_config: dict[str, str],
                   attn_input_scale: float,
                   q_output_scale: float,
                   k_output_scale: float,
                   v_output_scale: float,
                   out_input_scale: float,
                   gate_input_scale: float,
                   down_input_scale: float
                   ):
        int8_module = Int8LlamaDecoderLayer(
            config,
            quant_config,
            module.self_attn.layer_idx
        )
        int8_module.self_attn = Int8LlamaAttention.from_float(
            module.self_attn, 
            config,
            quant_config,
            attn_input_scale,
            q_output_scale,
            k_output_scale,
            v_output_scale,
            out_input_scale
        )
        int8_module.mlp = Int8LlamaMLP.from_float(
            module.mlp, 
            config,
            quant_config,
            gate_input_scale,
            down_input_scale
        )
        if quant_config["qkv"] == "per-tensor":
            int8_module.input_layernorm = Int8LlamaRMSNorm.from_float(
                module.input_layernorm,
                attn_input_scale
            )
        else:
            int8_module.input_layernorm = module.input_layernorm
        if quant_config["fc1"] == "per-tensor":
            int8_module.post_attention_layernorm = Int8LlamaRMSNorm.from_float(
                module.post_attention_layernorm,
                gate_input_scale
            )
        else:
            int8_module.post_attention_layernorm = module.post_attention_layernorm
        return int8_module
    
class Int8LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig, quant_config: dict[str, str]):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Int8LlamaDecoderLayer(config, quant_config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    get_input_embeddings = LlamaModel.get_input_embeddings
    set_input_embeddings = LlamaModel.set_input_embeddings
    forward = LlamaModel.forward
    
    @staticmethod
    def from_float(module, decoder_layer_scales, quant_config):
        int8_module = Int8LlamaModel(module.config, quant_config)
        
        int8_module.embed_tokens = module.embed_tokens
        int8_module.norm = module.norm
        
        for i, layer in enumerate(module.layers):
            int8_module.layers[i] = Int8LlamaDecoderLayer.from_float(
                layer, module.config, quant_config, **decoder_layer_scales[i])
        return int8_module

class Int8LlamaForCausalLM(LlamaPreTrainedModel):
    def __init__(self, config, quant_config):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.model = Int8LlamaModel(config, quant_config)
        # no need to quant
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    get_input_embeddings = LlamaForCausalLM.get_input_embeddings
    set_input_embeddings = LlamaForCausalLM.set_input_embeddings
    get_output_embeddings = LlamaForCausalLM.get_output_embeddings
    set_output_embeddings = LlamaForCausalLM.set_output_embeddings
    set_decoder = LlamaForCausalLM.set_decoder
    get_decoder = LlamaForCausalLM.get_decoder
    forward = LlamaForCausalLM.forward
    prepare_inputs_for_generation = LlamaForCausalLM.prepare_inputs_for_generation
    _reorder_cache = LlamaForCausalLM._reorder_cache
    
    @staticmethod
    def from_float(module, decoder_layer_scales, quant_config):
        int8_module = Int8LlamaForCausalLM(module.config, quant_config)
        print("start trans into int8, this might take a while")
        int8_module.model = Int8LlamaModel.from_float(
            module.model, decoder_layer_scales, quant_config)
        int8_module.lm_head = module.lm_head
        return int8_module   
