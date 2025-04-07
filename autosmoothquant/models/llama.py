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
    LlamaRotaryEmbedding
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.activations import ACT2FN
from typing import Optional
from autosmoothquant.layers.nn.linear import (
    W8A8BFP32OFP32LinearWithQuantScale,
    W8A8BFP32OFP32Linear,
    FP8LinearDynamic,
    FP8LinearStatic, FP8E5M2Linear
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


class QuantizedLlamaRMSNorm(LlamaRMSNorm):

    @staticmethod
    def from_float(module: LlamaRMSNorm,
                   input_scale: float):
        quantized_module = QuantizedLlamaRMSNorm(module.weight.numel(), module.variance_epsilon)

        quantized_module.weight.to(module.weight.dtype)
        quantized_module.weight = nn.Parameter(module.weight / input_scale)

        return quantized_module


class QuantizedLlamaAttention(nn.Module):
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
        if quant_config["type"] == "fp8_e4m3":
            if quant_config["activation_scheme"] == "static":
                self.k_proj = FP8LinearStatic(self.hidden_size, self.num_key_value_heads * self.head_dim)
                self.v_proj = FP8LinearStatic(self.hidden_size, self.num_key_value_heads * self.head_dim)
                self.q_proj = FP8LinearStatic(self.hidden_size, self.num_heads * self.head_dim)
                self.o_proj = FP8LinearStatic(self.num_heads * self.head_dim, self.hidden_size)
            else:
                self.k_proj = FP8LinearDynamic(self.hidden_size, self.num_key_value_heads * self.head_dim,
                                               act_quant=self.qkv_quant_type)
                self.v_proj = FP8LinearDynamic(self.hidden_size, self.num_key_value_heads * self.head_dim,
                                               act_quant=self.qkv_quant_type)
                self.q_proj = FP8LinearDynamic(self.hidden_size, self.num_heads * self.head_dim,
                                               act_quant=self.qkv_quant_type)
                self.o_proj = FP8LinearDynamic(self.num_heads * self.head_dim, self.hidden_size,
                                               act_quant=self.o_quant_type)

        elif quant_config["type"] == "fp8_e5m2":
            self.k_proj = FP8E5M2Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
            self.v_proj = FP8E5M2Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
            self.q_proj = FP8E5M2Linear(self.hidden_size, self.num_heads * self.head_dim)
            self.o_proj = FP8E5M2Linear(self.num_heads * self.head_dim, self.hidden_size)

        else:
            self.k_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.num_key_value_heads * self.head_dim,
                                               act_quant=self.qkv_quant_type)
            self.v_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.num_key_value_heads * self.head_dim,
                                               act_quant=self.qkv_quant_type)
            self.q_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.num_heads * self.head_dim,
                                               act_quant=self.qkv_quant_type)
            self.o_proj = W8A8BFP32OFP32LinearWithQuantScale(self.num_heads * self.head_dim, self.hidden_size,
                                                             act_quant=self.o_quant_type)
        # self._init_rope()

    _init_rope = LlamaAttention._init_rope
    # _shape = LlamaAttention._shape
    forward = LlamaAttention.forward
    
    @staticmethod
    @torch.no_grad()
    def from_float_to_int8(module: LlamaAttention,
                           config: LlamaConfig,
                           quant_config: dict[str, str],
                           attn_input_scale: float,
                           q_output_scale: float,
                           k_output_scale: float,
                           v_output_scale: float,
                           out_input_scale: float):
        int8_module = QuantizedLlamaAttention(config, quant_config)
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

    @staticmethod
    @torch.no_grad()
    def from_float_to_fp8(module: LlamaAttention,
                          config: LlamaConfig,
                          quant_config: dict[str, str]
                          ):
        fp8_module = QuantizedLlamaAttention(config, quant_config)
        # fp8_e5m2
        if quant_config["type"] == "fp8_e5m2":
            """
            fp8_e5m2 only support per tensor
            """
            assert quant_config["qkv"] == "per-tensor"
            assert quant_config["out"] == "per-tensor"
            fp8_module.q_proj = FP8E5M2Linear.from_float(module.q_proj)
            fp8_module.k_proj = FP8E5M2Linear.from_float(module.k_proj)
            fp8_module.v_proj = FP8E5M2Linear.from_float(module.v_proj)
            fp8_module.o_proj = FP8E5M2Linear.from_float(module.o_proj)

        # fp8_e4m3
        elif quant_config["type"] == "fp8_e4m3":
            if quant_config["activation_scheme"] == "static":
                """
                fp8_e4m3 static only support per tensor
                """
                assert quant_config["qkv"] == "per-tensor"
                assert quant_config["out"] == "per-tensor"
                fp8_module.q_proj = FP8LinearStatic.from_float(module.q_proj)
                fp8_module.k_proj = FP8LinearStatic.from_float(module.k_proj)
                fp8_module.v_proj = FP8LinearStatic.from_float(module.v_proj)
                fp8_module.o_proj = FP8LinearStatic.from_float(module.o_proj)

            # Dynamic
            else:
                fp8_module.q_proj = FP8LinearDynamic.from_float(module.q_proj)
                fp8_module.k_proj = FP8LinearDynamic.from_float(module.k_proj)
                fp8_module.v_proj = FP8LinearDynamic.from_float(module.v_proj)
                fp8_module.o_proj = FP8LinearDynamic.from_float(module.o_proj)

        else:
            raise ValueError(f"Unsupported quant type: {quant_config['type']}")
        return fp8_module


class QuantizedLlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig, quant_config: dict[str, str]):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_quant_type = quant_config["fc1"]
        self.down_quant_type = quant_config["fc2"]
        if quant_config["type"] == "fp8_e4m3":
            if quant_config["activation_scheme"] == "static":
                self.gate_proj = FP8LinearStatic(self.hidden_size, self.intermediate_size)
                self.up_proj = FP8LinearStatic(self.hidden_size, self.intermediate_size)
                self.down_proj = FP8LinearStatic(self.intermediate_size, self.hidden_size)
            else:
                self.gate_proj = FP8LinearDynamic(self.hidden_size, self.intermediate_size,
                                                  act_quant=self.gate_up_quant_type)
                self.up_proj = FP8LinearDynamic(self.hidden_size, self.intermediate_size,
                                                act_quant=self.gate_up_quant_type)
                self.down_proj = FP8LinearDynamic(self.intermediate_size, self.hidden_size,
                                                  act_quant=self.down_quant_type)

        elif quant_config["type"] == "fp8_e5m2":
            self.gate_proj = FP8E5M2Linear(self.hidden_size, self.intermediate_size)
            self.up_proj = FP8E5M2Linear(self.hidden_size, self.intermediate_size)
            self.down_proj = FP8E5M2Linear(self.intermediate_size, self.hidden_size)

        elif quant_config["type"] == "int8":
            self.gate_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.intermediate_size,
                                                  act_quant=self.gate_up_quant_type)
            self.up_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.intermediate_size,
                                                act_quant=self.gate_up_quant_type)
            self.down_proj = W8A8BFP32OFP32LinearWithQuantScale(self.intermediate_size, self.hidden_size,
                                                                act_quant=self.down_quant_type)

        else:
            raise ValueError(f"Unsupported quant type: {quant_config['type']}")

        self.act_fn = ACT2FN[config.hidden_act]

    forward = LlamaMLP.forward

    @staticmethod
    @torch.no_grad()
    def from_float_to_int8(module: LlamaMLP,
                           config: LlamaConfig,
                           quant_config: dict[str, str],
                           gate_input_scale: float,
                           down_input_scale: float):
        int8_module = QuantizedLlamaMLP(config, quant_config)
        int8_module.gate_proj = W8A8BFP32OFP32Linear.from_float(module.gate_proj, gate_input_scale,
                                                                act_quant=int8_module.gate_up_quant_type)
        int8_module.up_proj = W8A8BFP32OFP32Linear.from_float(module.up_proj, gate_input_scale,
                                                              act_quant=int8_module.gate_up_quant_type)
        int8_module.down_proj = W8A8BFP32OFP32LinearWithQuantScale.from_float(
            module.down_proj,
            down_input_scale,
            act_quant=int8_module.down_quant_type)
        return int8_module

    @staticmethod
    @torch.no_grad()
    def from_float_to_fp8(module: LlamaMLP,
                          config: LlamaConfig,
                          quant_config: dict[str, str]):
        fp8_module = QuantizedLlamaMLP(config, quant_config)

        # fp8_e5m2
        if quant_config["type"] == "fp8_e5m2":
            """
            fp8_e5m2 only support per tensor
            """
            assert quant_config["qkv"] == "per-tensor"
            assert quant_config["out"] == "per-tensor"
            fp8_module.gate_proj = FP8E5M2Linear.from_float(module.gate_proj)
            fp8_module.up_proj = FP8E5M2Linear.from_float(module.up_proj)
            fp8_module.down_proj = FP8E5M2Linear.from_float(module.down_proj)

        # fp8_e4m3
        elif quant_config["type"] == "fp8_e4m3":
            if quant_config["activation_scheme"] == "static":
                """
                fp8_e4m3 static only support per tensor
                """
                assert quant_config["qkv"] == "per-tensor"
                assert quant_config["out"] == "per-tensor"
                fp8_module.gate_proj = FP8LinearStatic.from_float(module.gate_proj)
                fp8_module.up_proj = FP8LinearStatic.from_float(module.up_proj)
                fp8_module.down_proj = FP8LinearStatic.from_float(module.down_proj)

            # Dynamic
            else:
                fp8_module.gate_proj = FP8LinearDynamic.from_float(module.gate_proj)
                fp8_module.up_proj = FP8LinearDynamic.from_float(module.up_proj)
                fp8_module.down_proj = FP8LinearDynamic.from_float(module.down_proj)

        else:
            raise ValueError(f"Unsupported quant type: {quant_config['type']}")
        return fp8_module


class QuantizedLlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, quant_config: dict[str, str], layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        # only support LlamaAttention for now. TODO: support LlamaFlashAttention2 and LlamaSdpaAttention
        self.self_attn = QuantizedLlamaAttention(config, quant_config, layer_idx)
        self.mlp = QuantizedLlamaMLP(config, quant_config)
        self.input_layernorm = LlamaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    forward = LlamaDecoderLayer.forward

    @staticmethod
    def from_float_to_int8(module: LlamaDecoderLayer,
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
        quantized_module = QuantizedLlamaDecoderLayer(
            config,
            quant_config,
            module.self_attn.layer_idx
        )
        quantized_module.self_attn = QuantizedLlamaAttention.from_float_to_int8(
            module.self_attn,
            config,
            quant_config,
            attn_input_scale,
            q_output_scale,
            k_output_scale,
            v_output_scale,
            out_input_scale
        )
        quantized_module.mlp = QuantizedLlamaMLP.from_float_to_int8(
            module.mlp,
            config,
            quant_config,
            gate_input_scale,
            down_input_scale
        )
        # only per tensor need to apply input sclae to layernorm
        if quant_config["qkv"] == "per-tensor":
            quantized_module.input_layernorm = QuantizedLlamaRMSNorm.from_float(
                module.input_layernorm,
                attn_input_scale
            )
        else:
            quantized_module.input_layernorm = module.input_layernorm
        if quant_config["fc1"] == "per-tensor":
            quantized_module.post_attention_layernorm = QuantizedLlamaRMSNorm.from_float(
                module.post_attention_layernorm,
                gate_input_scale
            )
        else:
            quantized_module.post_attention_layernorm = module.post_attention_layernorm
        return quantized_module

    @staticmethod
    def from_float_to_fp8(module: LlamaDecoderLayer, config: LlamaConfig, quant_config: dict[str, str]):
        quantized_module = QuantizedLlamaDecoderLayer(
            config,
            quant_config,
            module.self_attn.layer_idx
        )
        quantized_module.self_attn = QuantizedLlamaAttention.from_float_to_fp8(module.self_attn, config, quant_config)
        quantized_module.mlp = QuantizedLlamaMLP.from_float_to_fp8(module.mlp, config, quant_config)
        quantized_module.input_layernorm = module.input_layernorm
        quantized_module.post_attention_layernorm = module.post_attention_layernorm
        return quantized_module


class QuantizedLlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig, quant_config: dict[str, str]):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([QuantizedLlamaDecoderLayer(config, quant_config, layer_idx) for layer_idx in
                                     range(config.num_hidden_layers)])
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    get_input_embeddings = LlamaModel.get_input_embeddings
    set_input_embeddings = LlamaModel.set_input_embeddings
    forward = LlamaModel.forward
    _update_causal_mask = LlamaModel._update_causal_mask

    @staticmethod
    def from_float_to_int8(module, decoder_layer_scales, quant_config):
        quantized_module = QuantizedLlamaModel(module.config, quant_config)

        quantized_module.embed_tokens = module.embed_tokens
        quantized_module.norm = module.norm

        for i, layer in enumerate(module.layers):
            quantized_module.layers[i] = QuantizedLlamaDecoderLayer.from_float_to_int8(
                layer, module.config, quant_config, **decoder_layer_scales[i])
        return quantized_module

    @staticmethod
    def from_float_to_fp8(module, quant_config):
        quantized_module = QuantizedLlamaModel(module.config, quant_config)

        quantized_module.embed_tokens = module.embed_tokens
        quantized_module.norm = module.norm

        for i, layer in enumerate(module.layers):
            quantized_module.layers[i] = QuantizedLlamaDecoderLayer.from_float_to_fp8(layer, module.config,
                                                                                      quant_config)
        return quantized_module


class QuantizedLlamaForCausalLM(LlamaPreTrainedModel):
    def __init__(self, config, quant_config):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.model = QuantizedLlamaModel(config, quant_config)
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
    def from_float_to_int8(module, decoder_layer_scales, quant_config):
        quantized_module = QuantizedLlamaForCausalLM(module.config, quant_config)
        print("start perform weight quantization, this might take a while")
        quantized_module.model = QuantizedLlamaModel.from_float_to_int8(
            module.model, decoder_layer_scales, quant_config)
        quantized_module.lm_head = module.lm_head
        return quantized_module

    @staticmethod
    def from_float_to_fp8(module, quant_config):
        quantized_module = QuantizedLlamaForCausalLM(module.config, quant_config)
        print("start perform weight quantization, this might take a while")
        quantized_module.model = QuantizedLlamaModel.from_float_to_fp8(
            module.model, quant_config)
        quantized_module.lm_head = module.lm_head
        return quantized_module
