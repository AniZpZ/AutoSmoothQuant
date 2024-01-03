import torch
import argparse
import os

from pathlib import Path

from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from torch.nn.functional import pad

from autosmoothquant.models.llama import Int8LlamaForCausalLM
from autosmoothquant.quantize.smooth import smooth_lm
from autosmoothquant.quantize.calibration import get_static_llama_decoder_layer_scales

# Configure activation quant map, options: 'per-token' or 'per-tensor'
act_quant_map = {
    "qkv_proj": "per-token",
    "o_proj": "per-token",
    "gate_up_proj": "per-token",
    "down_proj": "per-token"
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default='fp16_models/llama-13b')
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--act-scales", type=str,
                        default='act_scales/llama-13b.pt')
    parser.add_argument("--output-path", type=str, default='int8_models')
    parser.add_argument('--dataset-path', type=str, default='dataset/val.jsonl.zst',
                        help='location of the calibration dataset, we use the validation set of the Pile dataset')
    args = parser.parse_args()
    model = LlamaForCausalLM.from_pretrained(
        args.model_name, device_map="auto", torch_dtype=torch.float16)
    act_scales = torch.load(args.act_scales)
    smooth_lm(model, act_scales, 0.5)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if not os.path.exists(args.dataset_path):
        print(f'Cannot find the dataset at {args.dataset_path}')
        print('Please download the Pile dataset and put the validation set at the path')
        print('You can download the validation dataset of the Pile at https://mystic.the-eye.eu/public/AI/pile/val.jsonl.zst')
        raise FileNotFoundError

    decoder_layer_scales, raw_scales = get_static_llama_decoder_layer_scales(model,
                                                                            tokenizer,
                                                                            args.dataset_path,
                                                                            num_samples=args.num_samples,
                                                                            seq_len=args.seq_len)
    output_path = Path(args.output_path) / (Path(args.model_name).name + "-smoothquant")

    int8_model = Int8LlamaForCausalLM.from_float(model, decoder_layer_scales, act_quant_map)
    int8_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Saved int8 model at {output_path}")