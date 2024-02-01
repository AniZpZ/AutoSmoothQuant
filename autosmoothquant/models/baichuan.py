import torch
import math
from torch import nn
from transformers.activations import ACT2FN
from transformers.utils import logging
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from typing import Optional, Tuple, List, Union

from autosmoothquant.layers.nn.linear import W8A8BFP32OFP32LinearWithQuantScale, W8A8BFP32OFP32Linear, W8A8BFP32OFP32QKVLinear
from autosmoothquant.thirdparty.baichuan.modeling_baichuan import (
    RMSNorm,
    RotaryEmbedding,
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

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos_, sin_, position_ids):
    cos = cos_.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin_.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q.float() * cos) + (rotate_half(q.float()) * sin)
    k_embed = (k.float() * cos) + (rotate_half(k.float()) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)

class Int8BaichuanRMSNorm(RMSNorm):    
    
    @staticmethod
    def from_float(module: RMSNorm,
                   output_scale: float):
        int8_norm = Int8BaichuanRMSNorm(module.weight.numel(), module.epsilon)

        int8_norm.weight.to(module.weight.dtype)
        int8_norm.weight = torch.nn.Parameter(module.weight / output_scale)

        return int8_norm

# attention is the same as opt
class Int8BaichuanAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(
        self,
        config: BaichuanConfig,
        quant_config: dict[str, str],
        position_embedding: str
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.model_max_length
        self.position_embedding = position_embedding

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size {self.hidden_size} is not divisible by num_heads {self.num_heads}"
            )
        self.qkv_quant_type = quant_config["qkv"]
        self.o_quant_type = quant_config["out"]
        self.qkv_size = [self.num_heads * self.head_dim] * 3
        self.W_pack = W8A8BFP32OFP32QKVLinear(self.qkv_size, self.hidden_size, 3 * self.num_heads * self.head_dim, act_quant=self.qkv_quant_type)
        self.o_proj = W8A8BFP32OFP32LinearWithQuantScale(self.num_heads * self.head_dim, self.hidden_size, act_quant=self.o_quant_type)

        if self.position_embedding == "ROPE":
            self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    _shape = BaichuanAttention._shape
    
    @staticmethod
    @torch.no_grad()
    def from_float(module: BaichuanAttention,
                   config: BaichuanConfig,
                   quant_config: dict[str, str],
                   position_embedding: str,
                   attn_input_scale: float,
                   attn_output_scale: float,
                   out_input_scale: float):
        int8_module = Int8BaichuanAttention(config, quant_config, position_embedding)
        # we do not impelement attn for now bacuase we want to use paged attention
        int8_module.W_pack = W8A8BFP32OFP32QKVLinear.from_float(
          module.W_pack, attn_input_scale, int8_module.qkv_size, act_quant=int8_module.qkv_quant_type)
        int8_module.o_proj = W8A8BFP32OFP32LinearWithQuantScale.from_float(
            module.o_proj, out_input_scale, act_quant=int8_module.o_quant_type)
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

        if self.position_embedding == "ROPE":
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None
        if xops is not None and self.training:
            attn_weights = None
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

            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1).to(value_states.dtype)
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
        quant_config: dict[str, str]
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_up_quant_type = quant_config["fc1"]
        self.down_quant_type = quant_config["fc2"]
        self.gate_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.intermediate_size, act_quant=self.gate_up_quant_type)
        self.down_proj = W8A8BFP32OFP32LinearWithQuantScale(self.intermediate_size, self.hidden_size, act_quant=self.down_quant_type)
        self.up_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.intermediate_size, act_quant=self.gate_up_quant_type)
        self.act_fn = ACT2FN[hidden_act]
    
    @staticmethod
    @torch.no_grad()
    def from_float(module: MLP,
                   config: BaichuanConfig,
                   quant_config: dict[str, str],
                   gate_input_scale: float,
                   down_input_scale: float):
        int8_module = Int8BaichuanMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config
        )
        int8_module.gate_proj = W8A8BFP32OFP32Linear.from_float(module.gate_proj, gate_input_scale, act_quant=int8_module.gate_up_quant_type)
        int8_module.up_proj = W8A8BFP32OFP32Linear.from_float(module.up_proj, gate_input_scale, act_quant=int8_module.gate_up_quant_type)
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


class Int8BaichuanLayer(nn.Module):
    def __init__(self, config: BaichuanConfig, quant_config: dict[str, str], position_embedding: str):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Int8BaichuanAttention(
            config=config, 
            position_embedding=position_embedding, 
            quant_config=quant_config
        )
        self.mlp = Int8BaichuanMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config
        )
        self.input_layernorm = Int8BaichuanRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Int8BaichuanRMSNorm(config.hidden_size, config.rms_norm_eps)
    
    @staticmethod
    def from_float(module: BaichuanLayer,
                   config: BaichuanConfig,
                   quant_config: dict[str, str],
                   position_embedding: str,
                   attn_input_scale: float,
                   attn_output_scale: float,
                   out_input_scale: float,
                   gate_input_scale: float,
                   down_input_scale: float
                   ):
        int8_module = Int8BaichuanLayer(
            config,
            quant_config,
            position_embedding
        )

        int8_module.self_attn = Int8BaichuanAttention.from_float(
            module.self_attn, 
            config,
            quant_config,
            position_embedding,
            attn_input_scale,
            attn_output_scale,
            out_input_scale
        )
        
        int8_module.mlp = Int8BaichuanMLP.from_float(
            module.mlp, 
            config,
            quant_config,
            gate_input_scale,
            down_input_scale
        )
        if quant_config["qkv"] == "per-tensor":
            int8_module.input_layernorm = Int8BaichuanRMSNorm.from_float(
                module.input_layernorm,
                attn_input_scale
            )
        else:
            int8_module.input_layernorm = module.input_layernorm
        if quant_config["fc1"] == "per-tensor":
            int8_module.post_attention_layernorm = Int8BaichuanRMSNorm.from_float(
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
    def __init__(self, config: BaichuanConfig, position_embedding: str, quant_config: dict[str, str]):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.n_head = config.num_attention_heads
        self.embed_tokens = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = torch.nn.ModuleList(
            [Int8BaichuanLayer(config, quant_config, position_embedding) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()
        self.max_cache_pos = config.model_max_length
        self.first_run = True
        self.alibi_mask = None
        self.position_embedding = position_embedding
    
    @staticmethod
    def from_float(module, decoder_layer_scales, quant_config, position_embedding):
        int8_module = Int8BaichuanModel(module.config, position_embedding, quant_config)
        
        int8_module.embed_tokens = module.embed_tokens
        int8_module.norm = module.norm
        
        for i, layer in enumerate(module.layers):
            int8_module.layers[i] = Int8BaichuanLayer.from_float(
                layer, module.config, quant_config, position_embedding, **decoder_layer_scales[i])
        return int8_module
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot provide both input_ids and inputs_embeds simultaneously"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You need to provide input_ids or inputs_embeds")

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        seq_length_with_past = seq_length

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # baichuan 13b use alibi
        if self.position_embedding == "ALIBI":
            alibi_mask = self.get_alibi_mask(inputs_embeds, seq_length_with_past)

            if attention_mask is not None:
                if len(attention_mask.shape) == 2:
                    expanded_mask = attention_mask.to(alibi_mask.dtype)
                    expanded_mask = torch.tril(
                        torch.gt(expanded_mask[:, :, None] * expanded_mask[:, None, :], 0)
                    ) * torch.eq(expanded_mask[:, :, None] - expanded_mask[:, None, :], 0)
                else:
                    expanded_mask = attention_mask
                bsz = inputs_embeds.size(0)
                src_len, tgt_len = alibi_mask.size()[-2:]
                expanded_mask = (
                    expanded_mask.unsqueeze(1)
                    .expand(bsz, 1, src_len, tgt_len)
                    .to(alibi_mask.dtype)
                )
                inverted_mask = 1.0 - expanded_mask
                inverted_mask = inverted_mask.masked_fill(
                    inverted_mask.to(torch.bool), torch.finfo(alibi_mask.dtype).min
                )
                attention_mask = inverted_mask + alibi_mask.unsqueeze(0)
            else:
                attention_mask = alibi_mask
        else:
            # baichuan 7b use rope
            if attention_mask is None:
                attention_mask = torch.ones(
                    (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
                )
            attention_mask = self._prepare_rope_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    
    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_rope_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask


    get_input_embeddings = BaichuanModel.get_input_embeddings
    set_input_embeddings = BaichuanModel.set_input_embeddings
    get_alibi_mask = BaichuanModel.get_alibi_mask
    

class Int8BaichuanForCausalLM(BaichuanPreTrainedModel):
    def __init__(self, 
                 config, 
                 quant_config: dict[str, str], 
                 *model_args, 
                 **model_kwargs):
        super().__init__(config, *model_args, **model_kwargs)
        if config.hidden_size == 4096:  # 7b
            self.position_embedding = "ROPE"
        else:  # 13b
            self.position_embedding = "ALIBI"
        self.model = Int8BaichuanModel(config, 
                                       self.position_embedding, 
                                       quant_config)
        self.lm_head = NormHead(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()
    
    @staticmethod
    def from_float(module, decoder_layer_scales, quant_config: dict[str, str]):
        int8_module = Int8BaichuanForCausalLM(module.config, quant_config)
        print("start trans into int8, this might take a while")
        if module.config.hidden_size == 4096:
            position_embedding = "ROPE"
        else:
            position_embedding = "ALIBI"
        int8_module.model = Int8BaichuanModel.from_float(
            module.model, decoder_layer_scales, quant_config, position_embedding)
        int8_module.lm_head = module.lm_head
        return int8_module
    
    get_input_embeddings =  BaichuanForCausalLM.get_input_embeddings
    set_input_embeddings =  BaichuanForCausalLM.set_input_embeddings
    get_output_embeddings =  BaichuanForCausalLM.get_output_embeddings
    set_output_embeddings =  BaichuanForCausalLM.set_output_embeddings
    set_decoder =  BaichuanForCausalLM.set_decoder
    get_decoder =  BaichuanForCausalLM.get_decoder
    forward =  BaichuanForCausalLM.forward
    prepare_inputs_for_generation =  BaichuanForCausalLM.prepare_inputs_for_generation
