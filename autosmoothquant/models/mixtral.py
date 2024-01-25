# coding=utf-8
import torch
import math
from torch import nn
from transformers.models.mixtral.modeling_mixtral import (
    MixtralRMSNorm,
    MixtralRotaryEmbedding,
    MixtralAttention,
    MixtralBLockSparseTop2MLP,
    MixtralSparseMoeBlock,
    MixtralDecoderLayer,
    MixtralPreTrainedModel,
    MixtralModel,
    MixtralForCausalLM,
    MixtralForSequenceClassification
)
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.activations import SiLUActivation
from typing import Optional, Tuple, List
from autosmoothquant.layers.nn.linear import W8A8BFP32OFP32LinearWithQuantScale, W8A8BFP32OFP32Linear
from transformers.utils import logging
logger = logging.get_logger(__name__)

KVCache = Tuple[torch.Tensor, torch.Tensor]

class MixtralMLP(nn.Module):

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.ffn_dim = intermediate_size
        self.hidden_dim = hidden_size

        self.w1 = ReplicatedLinear(self.hidden_dim,
                                   self.ffn_dim,
                                   bias=False,
                                   linear_method=linear_method)
        self.w2 = ReplicatedLinear(self.ffn_dim,
                                   self.hidden_dim,
                                   bias=False,
                                   linear_method=linear_method)
        self.w3 = ReplicatedLinear(self.hidden_dim,
                                   self.ffn_dim,
                                   bias=False,
                                   linear_method=linear_method)

        # TODO: Use vllm's SiluAndMul
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        w1_out, _ = self.w1(hidden_states)
        w1_out = self.act_fn(w1_out)
        w3_out, _ = self.w3(hidden_states)
        current_hidden_states = w1_out * w3_out
        current_hidden_states, _ = self.w2(current_hidden_states)
        return current_hidden_states


class MixtralMoE(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.config = config
        self.rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_total_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        if self.tp_size > self.num_total_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {self.num_total_experts}.")
        # Split experts equally between ranks
        self.expert_indicies = np.array_split(range(
            self.num_total_experts), self.tp_size)[self.rank].tolist()
        if not self.expert_indicies:
            raise ValueError(
                f"Rank {self.rank} has no experts assigned to it.")

        self.experts = nn.ModuleList([
            MixtralMLP(self.num_total_experts,
                       config.hidden_size,
                       config.intermediate_size,
                       linear_method=linear_method)
            if idx in self.expert_indicies else None
            for idx in range(self.num_total_experts)
        ])
        self.gate = ReplicatedLinear(config.hidden_size,
                                     self.num_total_experts,
                                     bias=False,
                                     linear_method=None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits, _ = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights,
                                                       self.top_k,
                                                       dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        final_hidden_states = None
        for expert_idx in self.expert_indicies:
            expert_layer = self.experts[expert_idx]
            expert_mask = (selected_experts == expert_idx)
            expert_weights = (routing_weights * expert_mask).sum(dim=-1,
                                                                 keepdim=True)

            current_hidden_states = expert_layer(hidden_states).mul_(
                expert_weights)
            if final_hidden_states is None:
                final_hidden_states = current_hidden_states
            else:
                final_hidden_states.add_(current_hidden_states)

        return tensor_model_parallel_all_reduce(final_hidden_states).view(
            batch_size, sequence_length, hidden_dim)


class MixtralAttention(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 max_position: int = 4096 * 32,
                 rope_theta: float = 10000,
                 linear_method: Optional[LinearMethodBase] = None,
                 sliding_window: Optional[int] = None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            linear_method=linear_method,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            linear_method=linear_method,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=int(self.rope_theta),
            is_neox_style=True,
        )
        self.attn = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            sliding_window=self.sliding_window,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class Int8MixtralDecoderLayer(nn.Module):

    def __init__(self, config: MixtralConfig, quant_config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # only support MixtralAttention for now. TODO: support MixtralFlashAttention2 and MixtralSdpaAttention
        self.self_attn = MIXTRAL_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.block_sparse_moe = MixtralSparseMoeBlock(config)
        self.input_layernorm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
            quant_config
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
        if quant_config["qkv_proj"] == "per-tensor":
            int8_module.input_layernorm = Int8LlamaRMSNorm.from_float(
                module.input_layernorm,
                attn_input_scale
            )
        else:
            int8_module.input_layernorm = module.input_layernorm
        if quant_config["gate_up_proj"] == "per-tensor":
            int8_module.post_attention_layernorm = Int8LlamaRMSNorm.from_float(
                module.post_attention_layernorm,
                gate_input_scale
            )
        else:
            int8_module.post_attention_layernorm = module.post_attention_layernorm
        return int8_module


class Int8MixtralModel(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        quant_config
    ) -> None:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MixtralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.quant_config = quant_config

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def from_float(module, decoder_layer_scales, quant_config):
        int8_module = Int8MixtralModel(module.config, quant_config)
        
        int8_module.embed_tokens = module.embed_tokens
        int8_module.norm = module.norm
        
        for i, layer in enumerate(module.layers):
            int8_module.layers[i] = Int8MixtralDecoderLayer.from_float(
                layer, module.config, quant_config, **decoder_layer_scales[i])
        return int8_module

    get_input_embeddings = MixtralModel.get_input_embeddings
    set_input_embeddings = MixtralModel.set_input_embeddings
    forward = MixtralModel.forward()

class Int8MixtralForCausalLM(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        quant_config
    ):
        super().__init__()
        self.config = config
        self.model = Int8MixtralModel(config, quant_config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def from_float(module, decoder_layer_scales, quant_config):
        int8_module = Int8MixtralForCausalLM(module.config, quant_config)
        print("start trans into int8, this might take a while")
        int8_module.model = Int8MixtralModel.from_float(
            module.model, decoder_layer_scales, quant_config)
        int8_module.lm_head = module.lm_head
        return int8_module

    get_input_embeddings = MixtralForCausalLM.get_output_embeddings
    set_input_embeddings = MixtralForCausalLM.set_input_embeddings
    get_output_embeddings = MixtralForCausalLM.get_output_embeddings
    set_output_embeddings = MixtralForCausalLM.set_output_embeddings
    set_decoder = MixtralForCausalLM.set_decoder
    get_decoder = MixtralForCausalLM.get_decoder
    prepare_inputs_for_generation = MixtralForCausalLM.prepare_inputs_for_generation
    _reorder_cache = MixtralForCausalLM._reorder_cache
    forward = MixtralForCausalLM.forward