from .baichuan import Int8BaichuanForCausalLM
from .llama import Int8LlamaForCausalLM
from .mixtral import Int8MixtralForCausalLM
from .opt import Int8OPTForCausalLM
from autosmoothquant.thirdparty.baichuan.configuration_baichuan import BaichuanConfig

_MODEL_REGISTRY = {
    "LlamaForCausalLM": Int8LlamaForCausalLM,
    "LLaMAForCausalLM": Int8LlamaForCausalLM,
    "BaichuanForCausalLM": Int8BaichuanForCausalLM,
    "OPTForCausalLM": Int8OPTForCausalLM,
    "MixtralForCausalLM": Int8MixtralForCausalLM
}

_MODEL_TYPE = {
    "LlamaForCausalLM": "llama",
    "LLaMAForCausalLM": "llama",
    "BaichuanForCausalLM": "baichuan",
    "OPTForCausalLM": "transformers",
    "MixtralForCausalLM": "mixtral"
}

_CONFIG_REGISTRY = {
    "baichuan": BaichuanConfig,
}
