import torch
import argparse
import os
from pathlib import Path

from autosmoothquant.quantize.smooth import smooth_lm
from autosmoothquant.quantize.calibration import get_act_scales, get_static_decoder_layer_scales
from autosmoothquant.utils import get_config, get_model_architecture, build_model_and_tokenizer, parse_quant_config

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
    parser.add_argument("--smooth-strength", type=float, default=0.5,
                        help='migration strength of smoothquant, should be in a range of (0, 1)')
    args = parser.parse_args()
    return args

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
        os.makedirs(os.path.dirname(args.scale_output), exist_ok=True)
        torch.save(act_scales, args.scale_output)
    
    if args.quantize_model:
        act_scales = torch.load(args.scale_input)
        smooth_lm(model, act_scales, args.smooth_strength)
        config = get_config(args.model_path)
        quant_model_class, model_type = get_model_architecture(config)
        decoder_layer_scales, _ = get_static_decoder_layer_scales(model,
                                                                  tokenizer,
                                                                  args.dataset_path,
                                                                  num_samples=args.num_samples,
                                                                  seq_len=args.seq_len,
                                                                  model_type=model_type)
        output_path = Path(args.model_output) / (Path(args.model_path).name + "-smoothquant")

        config_path = os.path.join(args.model_path, "quant_config.json")
        quant_config = parse_quant_config(config_path)
        int8_model = quant_model_class.from_float(model, decoder_layer_scales, quant_config)
        
        int8_model.save_pretrained(output_path)

if __name__ == '__main__':
    main()
