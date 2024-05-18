import torch
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RMSNorm,
    Qwen2MLP,
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2PreTrainedModel,
    Qwen2Model,
    Qwen2ForCausalLM,
    Qwen2RotaryEmbedding
)
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.activations import ACT2FN
from typing import Optional
import sys

from layers.nn.linear import W8A8BFP32OFP32LinearWithQuantScale, W8A8BFP32OFP32Linear
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Int8Qwen2RMSNorm(Qwen2RMSNorm):

    @staticmethod
    def from_float(module: Qwen2RMSNorm,
                   output_scale: float):
        int8_module = Int8Qwen2RMSNorm(module.weight.numel(), module.variance_epsilon)

        int8_module.weight.to(module.weight.dtype)
        int8_module.weight = nn.Parameter(module.weight / output_scale)

        return int8_module


class Int8Qwen2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            config: Qwen2Config,
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
        self.k_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.num_heads * self.head_dim, use_bias=True,
                                           act_quant=self.qkv_quant_type)
        self.v_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.num_heads * self.head_dim, use_bias=True,
                                           act_quant=self.qkv_quant_type)
        self.q_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.num_heads * self.head_dim, use_bias=True,
                                           act_quant=self.qkv_quant_type)
        self.o_proj = W8A8BFP32OFP32LinearWithQuantScale(self.num_heads * self.head_dim, self.hidden_size,
                                                         act_quant=self.o_quant_type)
        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
    forward = Qwen2Attention.forward

    @staticmethod
    @torch.no_grad()
    def from_float(module: Qwen2Attention,
                   config: Qwen2Config,
                   quant_config: dict[str, str],
                   attn_input_scale: float,
                   q_output_scale: float,
                   k_output_scale: float,
                   v_output_scale: float,
                   out_input_scale: float):
        int8_module = Int8Qwen2Attention(config, quant_config)
        # we do not impelement attn for now bacuase we want use paged attention
        int8_module.q_proj = W8A8BFP32OFP32Linear.from_float(module.q_proj, attn_input_scale,
                                                             act_quant=int8_module.qkv_quant_type)
        int8_module.k_proj = W8A8BFP32OFP32Linear.from_float(module.k_proj, attn_input_scale,
                                                             act_quant=int8_module.qkv_quant_type)
        int8_module.v_proj = W8A8BFP32OFP32Linear.from_float(module.v_proj, attn_input_scale,
                                                             act_quant=int8_module.qkv_quant_type)
        int8_module.o_proj = W8A8BFP32OFP32LinearWithQuantScale.from_float(
            module.o_proj, out_input_scale, act_quant=int8_module.o_quant_type)
        return int8_module


class Int8Qwen2MLP(nn.Module):
    def __init__(self, config: Qwen2Config, quant_config: dict[str, str]):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_quant_type = quant_config["fc1"]
        self.down_quant_type = quant_config["fc2"]
        self.gate_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.intermediate_size,
                                              act_quant=self.gate_up_quant_type)
        self.up_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.intermediate_size, act_quant=self.gate_up_quant_type)
        self.down_proj = W8A8BFP32OFP32LinearWithQuantScale(self.intermediate_size, self.hidden_size,
                                                            act_quant=self.down_quant_type)
        self.act_fn = ACT2FN[config.hidden_act]

    forward = Qwen2MLP.forward

    @staticmethod
    @torch.no_grad()
    def from_float(module: Qwen2MLP,
                   config: Qwen2Config,
                   quant_config: dict[str, str],
                   gate_input_scale: float,
                   down_input_scale: float):
        int8_module = Int8Qwen2MLP(config, quant_config)
        int8_module.gate_proj = W8A8BFP32OFP32Linear.from_float(module.gate_proj, gate_input_scale,
                                                                act_quant=int8_module.gate_up_quant_type)
        int8_module.up_proj = W8A8BFP32OFP32Linear.from_float(module.up_proj, gate_input_scale,
                                                              act_quant=int8_module.gate_up_quant_type)
        int8_module.down_proj = W8A8BFP32OFP32LinearWithQuantScale.from_float(
            module.down_proj,
            down_input_scale,
            act_quant=int8_module.down_quant_type)
        return int8_module


class Int8Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, quant_config: dict[str, str], layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        # only support LlamaAttention for now. TODO: support LlamaFlashAttention2 and LlamaSdpaAttention
        self.self_attn = Int8Qwen2Attention(config, quant_config, layer_idx)
        self.mlp = Int8Qwen2MLP(config, quant_config)
        self.input_layernorm = Int8Qwen2RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Int8Qwen2RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    forward = Qwen2DecoderLayer.forward

    @staticmethod
    def from_float(module: Qwen2DecoderLayer,
                   config: Qwen2Config,
                   quant_config: dict[str, str],
                   attn_input_scale: float,
                   q_output_scale: float,
                   k_output_scale: float,
                   v_output_scale: float,
                   out_input_scale: float,
                   gate_input_scale: float,
                   down_input_scale: float
                   ):
        int8_module = Int8Qwen2DecoderLayer(
            config,
            quant_config,
            module.self_attn.layer_idx
        )
        int8_module.self_attn = Int8Qwen2Attention.from_float(
            module.self_attn,
            config,
            quant_config,
            attn_input_scale,
            q_output_scale,
            k_output_scale,
            v_output_scale,
            out_input_scale
        )
        int8_module.mlp = Int8Qwen2MLP.from_float(
            module.mlp,
            config,
            quant_config,
            gate_input_scale,
            down_input_scale
        )
        if quant_config["qkv"] == "per-tensor":
            int8_module.input_layernorm = Int8Qwen2RMSNorm.from_float(
                module.input_layernorm,
                attn_input_scale
            )
        else:
            int8_module.input_layernorm = module.input_layernorm
        if quant_config["fc1"] == "per-tensor":
            int8_module.post_attention_layernorm = Int8Qwen2RMSNorm.from_float(
                module.post_attention_layernorm,
                gate_input_scale
            )
        else:
            int8_module.post_attention_layernorm = module.post_attention_layernorm
        return int8_module


class Int8Qwen2Model(Qwen2PreTrainedModel):
    def __init__(self, config: Qwen2Config, quant_config: dict[str, str]):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Int8Qwen2DecoderLayer(config, quant_config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    get_input_embeddings = Qwen2Model.get_input_embeddings
    set_input_embeddings = Qwen2Model.set_input_embeddings
    forward = Qwen2Model.forward


    @staticmethod
    def from_float(module, decoder_layer_scales, quant_config):
        int8_module = Int8Qwen2Model(module.config, quant_config)

        int8_module.embed_tokens = module.embed_tokens
        int8_module.norm = module.norm

        for i, layer in enumerate(module.layers):
            int8_module.layers[i] = Int8Qwen2DecoderLayer.from_float(
                layer, module.config, quant_config, **decoder_layer_scales[i])
        return int8_module


class Int8Qwen2ForCausalLM(Qwen2PreTrainedModel):
    def __init__(self, config, quant_config):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.model = Int8Qwen2Model(config, quant_config)
        # no need to quant
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    get_input_embeddings = Qwen2ForCausalLM.get_input_embeddings
    set_input_embeddings = Qwen2ForCausalLM.set_input_embeddings
    get_output_embeddings = Qwen2ForCausalLM.get_output_embeddings
    set_output_embeddings = Qwen2ForCausalLM.set_output_embeddings
    set_decoder = Qwen2ForCausalLM.set_decoder
    get_decoder = Qwen2ForCausalLM.get_decoder
    forward = Qwen2ForCausalLM.forward
    prepare_inputs_for_generation = Qwen2ForCausalLM.prepare_inputs_for_generation
    _reorder_cache = Qwen2ForCausalLM._reorder_cache

    @staticmethod
    def from_float(module, decoder_layer_scales, quant_config):
        int8_module = Int8Qwen2ForCausalLM(module.config, quant_config)
        print("start trans into int8, this might take a while")
        int8_module.model = Int8Qwen2Model.from_float(
            module.model, decoder_layer_scales, quant_config)
        int8_module.lm_head = module.lm_head
        return int8_module
