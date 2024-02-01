import torch
from torch import nn
from transformers.models.opt.modeling_opt import (
    OPTConfig,
    OPTForCausalLM,
    OPTModel,
    OPTPreTrainedModel,
    OPTLearnedPositionalEmbedding,
    OPTAttention,
    OPTDecoderLayer,
    OPTDecoder,
)
from typing import Optional, Tuple, List
from autosmoothquant.layers.nn.linear import W8A8BFP32OFP32Linear, W8A8BFP32OFP32LinearWithQuantScale
from transformers.utils import logging
from transformers.activations import ACT2FN

logger = logging.get_logger(__name__)

class Int8OPTLayerNorm(nn.LayerNorm):

    @staticmethod
    def from_float(module: nn.LayerNorm, output_scale: float):
        assert module.normalized_shape[0] == module.weight.numel()
        assert module.normalized_shape[0] == module.bias.numel()
        q_module = Int8OPTLayerNorm(module.normalized_shape[0], module.eps)
        q_module.weight = nn.Parameter(module.weight / output_scale)
        q_module.bias = nn.Parameter(module.bias / output_scale)
        return q_module

class Int8OPTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: OPTConfig,
        quant_config: dict[str, str],
        is_decoder: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.config = config

        def _handle_deprecated_argument(config_arg_name, config, fn_arg_name, kwargs):
            """
            If a the deprecated argument `fn_arg_name` is passed, raise a deprecation
            warning and return that value, otherwise take the equivalent config.config_arg_name
            """
            val = None
            if fn_arg_name in kwargs:
                logging.warning(
                    "Passing in {} to {self.__class__.__name__} is deprecated and won't be supported from v4.38."
                    " Please set it in the config instead"
                )
                val = kwargs.pop(fn_arg_name)
            else:
                val = getattr(config, config_arg_name)
            return val

        self.embed_dim = _handle_deprecated_argument("hidden_size", config, "embed_dim", kwargs)
        self.num_heads = _handle_deprecated_argument("num_attention_heads", config, "num_heads", kwargs)
        self.dropout = _handle_deprecated_argument("attention_dropout", config, "dropout", kwargs)
        self.enable_bias = _handle_deprecated_argument("enable_bias", config, "bias", kwargs)

        self.head_dim = self.embed_dim // self.num_heads
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.qkv_quant_type = quant_config["qkv"]
        self.o_quant_type = quant_config["out"]
        self.k_proj = W8A8BFP32OFP32Linear(self.embed_dim, self.embed_dim, use_bias=self.enable_bias, act_quant=self.qkv_quant_type)
        self.v_proj = W8A8BFP32OFP32Linear(self.embed_dim, self.embed_dim, use_bias=self.enable_bias, act_quant=self.qkv_quant_type)
        self.q_proj = W8A8BFP32OFP32Linear(self.embed_dim, self.embed_dim, use_bias=self.enable_bias, act_quant=self.qkv_quant_type)
        self.out_proj = W8A8BFP32OFP32LinearWithQuantScale(self.embed_dim, self.embed_dim, use_bias=self.enable_bias, act_quant=self.o_quant_type)

    _shape = OPTAttention._shape
    forward = OPTAttention.forward

    @staticmethod
    @torch.no_grad()
    def from_float(module: OPTAttention,
                   config: OPTConfig,
                   quant_config: dict[str, str],
                   input_scale: float,
                   q_output_scale: float,
                   k_output_scale: float,
                   v_output_scale: float,
                   out_input_scale: float):
        int8_module = Int8OPTAttention(config, quant_config, is_decoder=True)
        module.q_proj.weight *= module.scaling
        module.q_proj.bias *= module.scaling
        int8_module.q_proj = W8A8BFP32OFP32Linear.from_float(
            module.q_proj, input_scale, act_quant=int8_module.qkv_quant_type)
        int8_module.k_proj = W8A8BFP32OFP32Linear.from_float(
            module.k_proj, input_scale, act_quant=int8_module.qkv_quant_type)
        int8_module.v_proj = W8A8BFP32OFP32Linear.from_float(
            module.v_proj, input_scale, act_quant=int8_module.qkv_quant_type)
        int8_module.out_proj = W8A8BFP32OFP32LinearWithQuantScale.from_float(
            module.out_proj, out_input_scale, act_quant=int8_module.o_quant_type)
        return int8_module

class Int8OPTDecoderLayer(nn.Module):
    def __init__(self, config: OPTConfig, quant_config: dict[str, str]):
        super().__init__()
        self.embed_dim = config.hidden_size
        # only support OPTAttention for now. TODO: support OptFlashAttention2
        self.self_attn = Int8OPTAttention(
            config=config,
            quant_config=quant_config,
            is_decoder=True
        )

        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]

        self.self_attn_layer_norm = Int8OPTLayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )
        self.fc1_quant_type = quant_config["fc1"]
        self.fc2_quant_type = quant_config["fc2"]
        self.fc1 = W8A8BFP32OFP32Linear(self.embed_dim, config.ffn_dim, use_bias=config.enable_bias, act_quant=self.fc1_quant_type)
        self.fc2 = W8A8BFP32OFP32LinearWithQuantScale(config.ffn_dim, self.embed_dim, use_bias=config.enable_bias, act_quant=self.fc2_quant_type)
        self.final_layer_norm = Int8OPTLayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)

    forward = OPTDecoderLayer.forward

    @staticmethod
    def from_float(module: OPTDecoderLayer,
                   config: OPTConfig,
                   quant_config: dict[str, str],
                   attn_input_scale: float,
                   q_output_scale: float,
                   k_output_scale: float,
                   v_output_scale: float,
                   out_input_scale: float,
                   fc1_input_scale: float,
                   fc2_input_scale: float):
        int8_module = Int8OPTDecoderLayer(
            config,
            quant_config
        )
        int8_module.self_attn = Int8OPTAttention.from_float(
            module.self_attn, config, quant_config, attn_input_scale, q_output_scale, k_output_scale, v_output_scale, out_input_scale)
        int8_module.fc1 = W8A8BFP32OFP32Linear.from_float(
            module.fc1, fc1_input_scale, act_quant=int8_module.fc1_quant_type)
        int8_module.fc2 = W8A8BFP32OFP32LinearWithQuantScale.from_float(
            module.fc2, fc2_input_scale, act_quant=int8_module.fc2_quant_type)
        if quant_config["qkv"] == "per-tensor":
            int8_module.self_attn_layer_norm = Int8OPTLayerNorm.from_float(
                module.self_attn_layer_norm, attn_input_scale)
        else:
            int8_module.self_attn_layer_norm = module.self_attn_layer_norm
        if quant_config["fc1"] == "per-tensor":
            int8_module.final_layer_norm = Int8OPTLayerNorm.from_float(
                module.final_layer_norm, fc1_input_scale)
        else:
            int8_module.final_layer_norm = module.final_layer_norm
        return int8_module

class Int8OPTDecoder(OPTPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Int8OPTDecoderLayer`]

    """

    def __init__(self, config, quant_config):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
            )
        else:
            self.final_layer_norm = None

        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.layers = nn.ModuleList([Int8OPTDecoderLayer(config, quant_config) for _ in range(config.num_hidden_layers)])
        
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    get_input_embeddings = OPTDecoder.get_input_embeddings
    set_input_embeddings = OPTDecoder.set_input_embeddings
    forward = OPTDecoder.forward

    @staticmethod
    def from_float(module, decoder_layer_scales, quant_config):
        int8_module = Int8OPTDecoder(module.config, quant_config)
        int8_module.embed_tokens = module.embed_tokens
        int8_module.embed_positions = module.embed_positions
        int8_module.project_out = module.project_out
        int8_module.final_layer_norm = module.final_layer_norm
        for i, layer in enumerate(module.layers):
            int8_module.layers[i] = Int8OPTDecoderLayer.from_float(
                layer, module.config, quant_config, **decoder_layer_scales[i])
        return int8_module


class Int8OPTModel(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig, quant_config: dict[str, str]):
        super().__init__(config)
        self.decoder = Int8OPTDecoder(config, quant_config)
        # Initialize weights and apply final processing
        self.post_init()

    get_input_embeddings = OPTModel.get_input_embeddings
    set_input_embeddings = OPTModel.set_input_embeddings
    get_decoder = OPTModel.get_decoder
    forward = OPTModel.forward

    @staticmethod
    def from_float(module, decoder_layer_scales, quant_config):
        int8_module = Int8OPTModel(module.config, quant_config)
        int8_module.decoder = Int8OPTDecoder.from_float(
            module.decoder, decoder_layer_scales, quant_config)
        return int8_module


class Int8OPTForCausalLM(OPTPreTrainedModel):
    # _keys_to_ignore_on_load_missing = [r"lm_head.weight"]
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, quant_config):
        super().__init__(config)
        self.model = Int8OPTModel(config, quant_config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    get_input_embeddings = OPTForCausalLM.get_input_embeddings
    set_input_embeddings = OPTForCausalLM.set_input_embeddings
    get_output_embeddings = OPTForCausalLM.get_output_embeddings
    set_output_embeddings = OPTForCausalLM.set_output_embeddings
    set_decoder = OPTForCausalLM.set_decoder
    get_decoder = OPTForCausalLM.get_decoder
    forward = OPTForCausalLM.forward
    prepare_inputs_for_generation = OPTForCausalLM.prepare_inputs_for_generation
    _reorder_cache = OPTForCausalLM._reorder_cache

    @staticmethod
    def from_float(module, decoder_layer_scales, quant_config):
        int8_module = Int8OPTForCausalLM(module.config, quant_config)
        print("start trans into int8, this might take a while")
        int8_module.model = Int8OPTModel.from_float(
            module.model, decoder_layer_scales, quant_config)
        int8_module.lm_head = module.lm_head
        return int8_module