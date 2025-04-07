from typing import Optional, Type
import json
import torch
from torch import nn
import numpy as np
import random

from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig

from autosmoothquant.models import _MODEL_TYPE, _MODEL_REGISTRY, _CONFIG_REGISTRY

def get_config(model_path: str,
               trust_remote_code: bool = True,
               revision: Optional[str] = None) -> PretrainedConfig:
    try:
        config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, revision=revision)
    except ValueError as e:
        if (not trust_remote_code and
                "requires you to execute the configuration file" in str(e)):
            err_msg = (
                "Failed to load the model config. If the model is a custom "
                "model not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI.")
            raise RuntimeError(err_msg) from e
        else:
            raise e
    if config.model_type in _CONFIG_REGISTRY:
        config_class = _CONFIG_REGISTRY[config.model_type]
        config = config_class.from_pretrained(model_path, revision=revision)
    return config

def parse_quant_config(config_path):
  data = {}
  with open(config_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
  return data

def build_model_and_tokenizer(model_name, trust_remote_code: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code, model_max_length=512)
    kwargs = {"torch_dtype": torch.float16, "device_map": "sequential"}
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=trust_remote_code, **kwargs)
    return model, tokenizer

def get_model_architecture(config) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[arch], _MODEL_TYPE[arch]
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_MODEL_REGISTRY.keys())}")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True