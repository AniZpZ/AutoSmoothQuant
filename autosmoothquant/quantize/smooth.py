import torch
import torch.nn as nn

from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer, MixtralRMSNorm
from autosmoothquant.thirdparty.baichuan.modeling_baichuan import RMSNorm, BaichuanLayer


@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, model_type = "transformers", alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()
    if model_type == "llama":
        assert isinstance(ln, LlamaRMSNorm)
    elif model_type == "baichuan":
        assert isinstance(ln, RMSNorm)
    elif model_type == "mixtral":
        assert isinstance(ln, MixtralRMSNorm)
    else:
        assert isinstance(ln, nn.LayerNorm)

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat([fc.weight.abs().max(
        dim=0, keepdim=True)[0] for fc in fcs], dim=0)
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (act_scales.pow(alpha) / weight_scales.pow(1-alpha)
              ).clamp(min=1e-5).to(device).to(dtype)

    ln.weight.div_(scales)
    if model_type == "transformers":
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

@torch.no_grad()
def smooth_lm(model, scales, alpha=0.5):
    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            attn_ln = module.self_attn_layer_norm
            qkv = [module.self_attn.q_proj,
                   module.self_attn.k_proj, module.self_attn.v_proj]
            qkv_input_scales = scales[name + '.self_attn.q_proj']
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, "transformers", alpha)

            ffn_ln = module.final_layer_norm
            fc1 = module.fc1
            fc1_input_scales = scales[name + '.fc1']
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, "transformers", alpha)
        elif isinstance(module, LlamaDecoderLayer):
            print(f"smooth llama model: {name}")
            attn_ln = module.input_layernorm #attention forward norm
            qkv = [module.self_attn.q_proj,
                   module.self_attn.k_proj, module.self_attn.v_proj]
            qkv_input_scales = scales[name + '.self_attn.q_proj']
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, "llama", alpha)

            ffn_ln = module.post_attention_layernorm #feed forward norm
            fcs = [module.mlp.gate_proj, module.mlp.up_proj]
            fcs_input_scales = scales[name + '.mlp.gate_proj']
            smooth_ln_fcs(ffn_ln, fcs, fcs_input_scales, "llama", alpha)
        elif isinstance(module, BaichuanLayer):
            print(f"smooth baichuan model: {name}")
            attn_ln = module.input_layernorm
            qkv = module.self_attn.W_pack
            qkv_input_scales = scales[name + '.self_attn.W_pack']
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, "baichuan", alpha)

            ffn_ln = module.post_attention_layernorm #feed forward norm
            fcs = [module.mlp.gate_proj, module.mlp.up_proj]
            fcs_input_scales = scales[name + '.mlp.gate_proj']
            smooth_ln_fcs(ffn_ln, fcs, fcs_input_scales, "baichuan", alpha)
        elif isinstance(module, MixtralDecoderLayer):
            print(f"smooth mixtral model: {name}")
            attn_ln = module.input_layernorm
            qkv = [module.self_attn.q_proj,
                   module.self_attn.k_proj, module.self_attn.v_proj]
            qkv_input_scales = scales[name + '.self_attn.q_proj']
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, "mixtral", alpha)

            ffn_ln = module.post_attention_layernorm #feed forward norm
            fcs = [module.block_sparse_moe.gate]
            for expert in module.block_sparse_moe.experts:
                fcs.append(expert.w1)
                fcs.append(expert.w3)
            fcs_input_scales = scales[name + '.block_sparse_moe.gate']
            smooth_ln_fcs(ffn_ln, fcs, fcs_input_scales, "mixtral", alpha)

            
