import torch
import argparse
import os

from pathlib import Path

from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from torch.nn.functional import pad

from autosmoothquant.models.llama import Int8LlamaForCausalLM
from autosmoothquant.models.baichuan import Int8BaichuanForCausalLM
from autosmoothquant.quantize.smooth import smooth_lm
from autosmoothquant.quantize.calibration import get_static_decoder_layer_scales

_MODEL_REGISTRY = {
    "LlamaForCausalLM": Int8LlamaForCausalLM,
    "LLaMAForCausalLM": Int8LlamaForCausalLM,
    "BaichuanForCausalLM": Int8BaichuanForCausalLM,
    "OptForCausalLM": Int8OptForCausalLM
}

_CONFIG_REGISTRY = {
    "mpt": MptConfig,
    "baichuan": BaiChuanConfig,
    "aquila": AquilaConfig,
    "qwen": QWenConfig,
}

def build_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    kwargs = {"torch_dtype": torch.float16, "device_map": "sequential"}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str,
                        default='models/llama-13b', help='model path contains weights and config etc')
    parser.add_argument('--quantize-model', type=bool,
                        default=True, help='whether to quant model or not')    
    parser.add_argument('--generate-scale', type=bool,
                        default=True, help='whether to generate scale or not')                  
    parser.add_argument('--dataset-path', type=str, default='dataset/val.jsonl.zst',
                        help='location of the calibration dataset')
    parser.add_argument('--scale-output', type=str, default='scales/llama-13b',
                        help='where to save the act scales, activate when generating scales')
    parser.add_argument("--scale-input", type=str, default='scales/llama-13b',
                        help='where to save the act scales, activate when quantizing models')
    parser.add_argument('--num-samples', type=int, default=512)
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument("--model-output", type=str, default='quantized_model/llama-13b',
                        help='where to save the quantized models, activate when quantizing models')

    args = parser.parse_args()
    return args

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
        config = config_class.from_pretrained(model, revision=revision)
    return config

def get_model_architecture(config) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[arch]
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_MODEL_REGISTRY.keys())}")

@torch.no_grad()
def main():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(args.model_path)

    if not os.path.exists(args.dataset_path):
        print(f'Cannot find the dataset at {args.dataset_path}')
        print('Please download the dataset and put the validation set at the path')
        raise FileNotFoundError
    
    if args.generate_scale:
        act_scales = get_act_scales(model, tokenizer, args.dataset_path,
                                args.num_samples, args.seq_len)
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        torch.save(act_scales, args.output_path)
    
    if args.quantize_model:
        act_scales = torch.load(args.act_scales)
        smooth_lm(model, act_scales, 0.5)
        config = get_config(args.model_path)
        quant_model_class = get_model_architecture(config)
        decoder_layer_scales, _ = get_static_decoder_layer_scales(model,
                                                                  tokenizer,
                                                                  args.dataset_path,
                                                                  num_samples=args.num_samples,
                                                                  seq_len=args.seq_len)
        output_path = Path(args.output_path) / (Path(args.model_name).name + "-smoothquant")

        int8_model = quant_model_class.from_float(model, decoder_layer_scales)
        int8_model.save_pretrained(output_path)


if __name__ == '__main__':
    main()