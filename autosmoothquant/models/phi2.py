""" PyTorch Phi model."""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers.models.phi.modeling_phi import (
    PhiMLP,
    PhiAttention,
    PhiDecoderLayer,
    PhiPreTrainedModel,
    PhiModel,
    PhiForCausalLM,
)
import sys
sys.path.append("./smoothquant")
from transformers.activations import ACT2FN
from layers.nn.linear import W8A8BFP32OFP32LinearWithQuantScale, W8A8BFP32OFP32Linear
from transformers.utils import logging
from transformers.models.phi.configuration_phi import PhiConfig

logger = logging.get_logger(__name__)
class Int8PhiLayerNorm(nn.LayerNorm):
    @staticmethod
    def from_float(module: nn.LayerNorm, output_scale: float):
        assert module.normalized_shape[0] == module.weight.numel()
        assert module.normalized_shape[0] == module.bias.numel()
        q_module = Int8PhiLayerNorm(module.normalized_shape[0], module.eps)
        q_module.weight = nn.Parameter(module.weight / output_scale)
        q_module.bias = nn.Parameter(module.bias / output_scale)
        return q_module
class Int8PhiMLP(nn.Module):
    def __init__(self, config, quant_config: dict[str, str]):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1_quant_type = quant_config["fc1"]
        self.fc2_quant_type = quant_config["fc2"]
        self.fc1 = W8A8BFP32OFP32Linear(config.hidden_size, config.intermediate_size, act_quant=self.fc1_quant_type)
        self.fc2 = W8A8BFP32OFP32LinearWithQuantScale(config.intermediate_size, config.hidden_size,act_quant=self.fc2_quant_type)

    forward = PhiMLP.forward

    @staticmethod
    @torch.no_grad()
    def from_float(module: PhiMLP,
                   config: PhiConfig,
                   quant_config: dict[str, str],
                   fc1_input_scale: float,
                   fc2_input_scale: float):
        int8_module = Int8PhiMLP(config, quant_config)
        int8_module.fc1 = W8A8BFP32OFP32Linear.from_float(
            module.fc1, fc1_input_scale, act_quant=int8_module.fc1_quant_type)
        int8_module.fc2 = W8A8BFP32OFP32LinearWithQuantScale.from_float(
            module.fc2, fc2_input_scale, act_quant=int8_module.fc2_quant_type)
        return int8_module

class Int8PhiAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: PhiConfig, quant_config: dict[str, str], layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
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
        self.partial_rotary_factor = config.partial_rotary_factor
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.qkv_quant_type = quant_config["qkv"]
        self.o_quant_type = quant_config["out"]
        self.q_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.num_heads * self.head_dim, use_bias=True, act_quant=self.qkv_quant_type)
        self.k_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, use_bias=True, act_quant=self.qkv_quant_type)
        self.v_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, use_bias=True, act_quant=self.qkv_quant_type)
        self.dense = W8A8BFP32OFP32LinearWithQuantScale(self.num_heads * self.head_dim, self.hidden_size, use_bias=True, act_quant=self.o_quant_type)

        self.qk_layernorm = config.qk_layernorm
        # false
        if self.qk_layernorm:
            self.q_layernorm = nn.LayerNorm(
                config.hidden_size // self.num_heads, eps=config.layer_norm_eps, elementwise_affine=True
            )
            self.k_layernorm = nn.LayerNorm(
                config.hidden_size // self.num_heads, eps=config.layer_norm_eps, elementwise_affine=True
            )

        self._init_rope()

    _init_rope = PhiAttention._init_rope
    forward = PhiAttention.forward

    @staticmethod
    @torch.no_grad()
    def from_float(module: PhiAttention,
                   config: PhiConfig,
                   quant_config: dict[str, str],
                   attn_input_scale: float,
                   q_output_scale: float,
                   k_output_scale: float,
                   v_output_scale: float,
                   dense_input_scale: float):
        int8_module = Int8PhiAttention(config, quant_config)
        # we do not impelement attn for now bacuase we want use paged attention
        int8_module.q_proj = W8A8BFP32OFP32Linear.from_float(module.q_proj, attn_input_scale, act_quant=int8_module.qkv_quant_type)
        int8_module.k_proj = W8A8BFP32OFP32Linear.from_float(module.k_proj, attn_input_scale, act_quant=int8_module.qkv_quant_type)
        int8_module.v_proj = W8A8BFP32OFP32Linear.from_float(module.v_proj, attn_input_scale, act_quant=int8_module.qkv_quant_type)
        int8_module.dense = W8A8BFP32OFP32LinearWithQuantScale.from_float(
            module.dense, dense_input_scale, act_quant=int8_module.o_quant_type)
        return int8_module
class Int8PhiDecoderLayer(nn.Module):
    def __init__(self, config: PhiConfig, quant_config: dict[str, str], layer_idx: int):
        super().__init__()
        self.self_attn = Int8PhiAttention(config, quant_config, layer_idx=layer_idx)
        self.mlp = Int8PhiMLP(config, quant_config)
        self.input_layernorm = Int8PhiLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    forward = PhiDecoderLayer.forward

    @staticmethod
    def from_float(module: PhiDecoderLayer,
                   config: PhiConfig,
                   quant_config: dict[str, str],
                   attn_input_scale: float,
                   q_output_scale: float,
                   k_output_scale: float,
                   v_output_scale: float,
                   dense_input_scale: float,
                   fc1_input_scale: float,
                   fc2_input_scale: float
                   ):
        int8_module = Int8PhiDecoderLayer(
            config,
            quant_config,
            module.self_attn.layer_idx
        )
        int8_module.self_attn = Int8PhiAttention.from_float(
            module.self_attn,
            config,
            quant_config,
            attn_input_scale,
            q_output_scale,
            k_output_scale,
            v_output_scale,
            dense_input_scale
        )
        int8_module.mlp = Int8PhiMLP.from_float(
            module.mlp,
            config,
            quant_config,
            fc1_input_scale,
            fc2_input_scale
        )
        if quant_config["qkv"] == "per-tensor":
            int8_module.input_layernorm = Int8PhiLayerNorm.from_float(
                module.input_layernorm,
                attn_input_scale
            )
        else:
            int8_module.input_layernorm = module.input_layernorm

        return int8_module
class Int8PhiModel(PhiPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`PhiDecoderLayer`]

    Args:
        config: PhiConfig
    """

    def __init__(self, config: PhiConfig, quant_config: dict[str, str]):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_dropout = nn.Dropout(config.embd_pdrop)
        self.layers = nn.ModuleList(
            [Int8PhiDecoderLayer(config, quant_config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self._use_sdpa = config._attn_implementation == "sdpa"

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    get_input_embeddings = PhiModel.get_input_embeddings
    set_input_embeddings = PhiModel.set_input_embeddings
    forward = PhiModel.forward

    @staticmethod
    def from_float(module, decoder_layer_scales, quant_config):
        int8_module = Int8PhiModel(module.config, quant_config)

        int8_module.embed_tokens = module.embed_tokens
        int8_module.final_layernorm = module.final_layernorm

        for i, layer in enumerate(module.layers):
            int8_module.layers[i] = Int8PhiDecoderLayer.from_float(
                layer, module.config, quant_config, **decoder_layer_scales[i])
        return int8_module
class Int8PhiForCausalLM(PhiPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, quant_config):
        super().__init__(config)
        self.model = Int8PhiModel(config, quant_config)
        self.vocab_size = config.vocab_size
        # no need to quant
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

        # Initialize weights and apply final processing
        self.post_init()

    get_input_embeddings = PhiForCausalLM.get_input_embeddings
    set_input_embeddings = PhiForCausalLM.set_input_embeddings
    get_output_embeddings = PhiForCausalLM.get_output_embeddings
    set_output_embeddings = PhiForCausalLM.set_output_embeddings
    set_decoder = PhiForCausalLM.set_decoder
    get_decoder = PhiForCausalLM.get_decoder
    forward = PhiForCausalLM.forward
    prepare_inputs_for_generation = PhiForCausalLM.prepare_inputs_for_generation
    _reorder_cache = PhiForCausalLM._reorder_cache

    @staticmethod
    def from_float(module, decoder_layer_scales, quant_config):
        int8_module = Int8PhiForCausalLM(module.config, quant_config)
        print("start trans into int8, this might take a while")
        int8_module.model = Int8PhiModel.from_float(
            module.model, decoder_layer_scales, quant_config)
        int8_module.lm_head = module.lm_head
        return int8_module