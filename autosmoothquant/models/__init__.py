from .baichuan import Int8BaichuanForCausalLM
from .llama import Int8LlamaForCausalLM
from .mixtral import Int8MixtralForCausalLM
from .opt import Int8OPTForCausalLM
from .phi2 import Int8PhiForCausalLM
from .qwen2 import Int8Qwen2ForCausalLM
from autosmoothquant.thirdparty.baichuan.configuration_baichuan import BaichuanConfig

_MODEL_REGISTRY = {
    "LlamaForCausalLM": Int8LlamaForCausalLM,
    "LLaMAForCausalLM": Int8LlamaForCausalLM,
    "BaichuanForCausalLM": Int8BaichuanForCausalLM,
    "OPTForCausalLM": Int8OPTForCausalLM,
    "MixtralForCausalLM": Int8MixtralForCausalLM,
    "PhiForCausalLM": Int8PhiForCausalLM,
    "Qwen2ForCausalLM": Int8Qwen2ForCausalLM
}

_MODEL_TYPE = {
    "LlamaForCausalLM": "llama",
    "LLaMAForCausalLM": "llama",
    "BaichuanForCausalLM": "baichuan",
    "OPTForCausalLM": "transformers",
    "MixtralForCausalLM": "mixtral",
    "PhiForCausalLM": "phi",
    "Qwen2ForCausalLM": "qwen2"
}

_CONFIG_REGISTRY = {
    "baichuan": BaichuanConfig,
}
