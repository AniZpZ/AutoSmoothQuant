# coding=utf-8
from torch import nn
from transformers.models.mixtral.modeling_mixtral import (
    MixtralRMSNorm,
    MixtralRotaryEmbedding,
    MixtralAttention,
    MixtralBlockSparseTop2MLP,
    MixtralSparseMoeBlock,
    MixtralDecoderLayer,
    MixtralPreTrainedModel,
    MixtralModel,
    MixtralForCausalLM,
)
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.activations import ACT2FN
from typing import Optional, List
from autosmoothquant.layers.nn.linear import W8A8BFP32OFP32LinearWithQuantScale, W8A8BFP32OFP32Linear
from transformers.utils import logging

logger = logging.get_logger(__name__)

class Int8MixtralRMSNorm(MixtralRMSNorm):
    
    @staticmethod
    def from_float(module: MixtralRMSNorm,
                   output_scale: float):
        int8_module = Int8MixtralRMSNorm(module.weight.numel(), module.variance_epsilon)
        int8_module.weight.to(module.weight.dtype)
        int8_module.weight = nn.Parameter(module.weight / output_scale)
        return int8_module

class Int8MixtralAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: MixtralConfig, quant_config: dict[str, str], layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.num_heads * self.head_dim, act_quant=quant_config["qkv"])
        self.k_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, act_quant=quant_config["qkv"])
        self.v_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, act_quant=quant_config["qkv"])
        self.o_proj = W8A8BFP32OFP32LinearWithQuantScale(self.num_heads * self.head_dim, self.hidden_size, act_quant=quant_config["out"])

        self.rotary_emb = MixtralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        
    forward = MixtralAttention.forward
    _shape = MixtralAttention._shape

    @staticmethod
    def from_float(module: MixtralAttention,
                   config: MixtralConfig,
                   quant_config: dict[str, str],
                   attn_input_scale: float,
                   q_output_scale: float,
                   k_output_scale: float,
                   v_output_scale: float,
                   out_input_scale: float):
        int8_module = Int8MixtralAttention(config, quant_config, module.layer_idx)
        int8_module.q_proj = W8A8BFP32OFP32Linear.from_float(module.q_proj, attn_input_scale, act_quant=quant_config["qkv"])
        int8_module.k_proj = W8A8BFP32OFP32Linear.from_float(module.k_proj, attn_input_scale, act_quant=quant_config["qkv"])
        int8_module.v_proj = W8A8BFP32OFP32Linear.from_float(module.v_proj, attn_input_scale, act_quant=quant_config["qkv"])
        int8_module.o_proj = W8A8BFP32OFP32LinearWithQuantScale.from_float(module.o_proj, out_input_scale, act_quant=quant_config["out"])
        return int8_module

class Int8MixtralBlockSparseTop2MLP(nn.Module):
    def __init__(self, config: MixtralConfig, quant_config: dict[str, str]):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.w1 = W8A8BFP32OFP32Linear(self.hidden_dim, self.ffn_dim, act_quant=quant_config["fc1"])
        self.w2 = W8A8BFP32OFP32LinearWithQuantScale(self.ffn_dim, self.hidden_dim, act_quant=quant_config["fc2"])
        self.w3 = W8A8BFP32OFP32Linear(self.hidden_dim, self.ffn_dim, act_quant=quant_config["fc1"])

        self.act_fn = ACT2FN[config.hidden_act]

    forward = MixtralBlockSparseTop2MLP.forward

    @staticmethod
    def from_float(module: MixtralBlockSparseTop2MLP,
                   config: MixtralConfig,
                   quant_config: dict[str, str],
                   moe_input_scale: float,
                   down_input_scale: float):
        int8_module = Int8MixtralBlockSparseTop2MLP(config, quant_config)
        int8_module.w1 = W8A8BFP32OFP32Linear.from_float(module.w1, moe_input_scale, act_quant=quant_config["fc1"])
        int8_module.w2 = W8A8BFP32OFP32LinearWithQuantScale.from_float(module.w2, down_input_scale, act_quant=quant_config["fc2"])
        int8_module.w3 = W8A8BFP32OFP32Linear.from_float(module.w3, moe_input_scale, act_quant=quant_config["fc1"])
        return int8_module 
    

class Int8MixtralSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config, quant_config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # We do not apply quant to gate for now, as it greatly affects the model performance
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList(
            [Int8MixtralBlockSparseTop2MLP(config, quant_config) for _ in range(self.num_experts)])

    forward = MixtralSparseMoeBlock.forward

    @staticmethod
    def from_float(module: MixtralSparseMoeBlock,
                   config: MixtralConfig,
                   quant_config: dict[str, str],
                   moe_input_scale: float,
                   down_input_scales: List[float]):
        int8_module = Int8MixtralSparseMoeBlock(config, quant_config)
        int8_module.gate = module.gate
        for i, expert in enumerate(module.experts):
            int8_module.experts[i] = Int8MixtralBlockSparseTop2MLP.from_float(
                expert, config, quant_config, moe_input_scale, down_input_scales[i]
            )
        return int8_module
    
    
class Int8MixtralDecoderLayer(nn.Module):

    def __init__(self, config: MixtralConfig, quant_config: dict[str, str], layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # only support MixtralAttention for now. TODO: support MixtralFlashAttention2
        self.self_attn = Int8MixtralAttention(config, quant_config, layer_idx)
        self.block_sparse_moe = Int8MixtralSparseMoeBlock(config, quant_config)
        self.input_layernorm = Int8MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Int8MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    forward = MixtralDecoderLayer.forward

    @staticmethod
    def from_float(module: MixtralDecoderLayer,
                   config: MixtralConfig,
                   quant_config: dict[str, str],
                   attn_input_scale: float,
                   q_output_scale: float,
                   k_output_scale: float,
                   v_output_scale: float,
                   out_input_scale: float,
                   moe_input_scale: float,
                   down_input_scales: List[float]
                   ):
        int8_module = Int8MixtralDecoderLayer(
            config,
            quant_config,
            module.self_attn.layer_idx
        )
        int8_module.self_attn = Int8MixtralAttention.from_float(
            module.self_attn, 
            config,
            quant_config,
            attn_input_scale,
            q_output_scale,
            k_output_scale,
            v_output_scale,
            out_input_scale
        )
        int8_module.block_sparse_moe = Int8MixtralSparseMoeBlock.from_float(
            module.block_sparse_moe, 
            config,
            quant_config,
            moe_input_scale,
            down_input_scales
        )
        if quant_config["qkv"] == "per-tensor":
            int8_module.input_layernorm = Int8MixtralRMSNorm.from_float(
                module.input_layernorm,
                attn_input_scale
            )
        else:
            int8_module.input_layernorm = module.input_layernorm
        if quant_config["fc1"] == "per-tensor":
            int8_module.post_attention_layernorm = Int8MixtralRMSNorm.from_float(
                module.post_attention_layernorm,
                moe_input_scale
            )
        else:
            int8_module.post_attention_layernorm = module.post_attention_layernorm
        return int8_module   


class Int8MixtralModel(MixtralPreTrainedModel):

    def __init__(
        self,
        config: MixtralConfig,
        quant_config: dict[str, str]
    ) -> None:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Int8MixtralDecoderLayer(config, quant_config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    get_input_embeddings = MixtralModel.get_input_embeddings
    set_input_embeddings = MixtralModel.set_input_embeddings
    forward = MixtralModel.forward

    @staticmethod
    def from_float(module, decoder_layer_scales, quant_config):
        int8_module = Int8MixtralModel(module.config, quant_config)
        int8_module.embed_tokens = module.embed_tokens
        int8_module.norm = module.norm
        for i, layer in enumerate(module.layers):
            int8_module.layers[i] = Int8MixtralDecoderLayer.from_float(
                layer, module.config, quant_config, **decoder_layer_scales[i])
        return int8_module

class Int8MixtralForCausalLM(MixtralPreTrainedModel):

    def __init__(
        self,
        config: MixtralConfig,
        quant_config: dict[str, str]
    ):
        super().__init__(config)
        self.config = config
        self.model = Int8MixtralModel(config, quant_config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        # Initialize weights and apply final processing
        self.post_init()

    get_input_embeddings = MixtralForCausalLM.get_output_embeddings
    set_input_embeddings = MixtralForCausalLM.set_input_embeddings
    get_output_embeddings = MixtralForCausalLM.get_output_embeddings
    set_output_embeddings = MixtralForCausalLM.set_output_embeddings
    set_decoder = MixtralForCausalLM.set_decoder
    get_decoder = MixtralForCausalLM.get_decoder
    prepare_inputs_for_generation = MixtralForCausalLM.prepare_inputs_for_generation
    _reorder_cache = MixtralForCausalLM._reorder_cache
    forward = MixtralForCausalLM.forward

    @staticmethod
    def from_float(module, decoder_layer_scales, quant_config):
        int8_module = Int8MixtralForCausalLM(module.config, quant_config)
        print("start trans into int8, this might take a while")
        int8_module.model = Int8MixtralModel.from_float(
            module.model, decoder_layer_scales, quant_config)
        int8_module.lm_head = module.lm_head
        return int8_module