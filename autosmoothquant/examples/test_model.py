import os
import torch
import argparse
import json

from autosmoothquant.models import Int8LlamaForCausalLM, Int8OPTForCausalLM, Int8BaichuanForCausalLM, Int8MixtralForCausalLM
from autosmoothquant.utils import parse_quant_config
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str,
                        default='int8-models/llama-13b', help='path contains model weight and quant config')
    parser.add_argument('--tokenizer-path', type=str,
                        default='int8-models/llama-13b', help='path contains tokenizer')
    parser.add_argument('--model-class', type=str,
                        default='llama', help='currently support: llama, baichuan, opt, mixtral')
    parser.add_argument('--prompt', type=str,
                        default='You are right, But Genshin Impact is', help='prompts')   
    args = parser.parse_args()
    return args

@torch.no_grad()
def main():
    args = parse_args()
    config_path = os.path.join(args.model_path, "quant_config.json")
    quant_config = parse_quant_config(config_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Except GEMM uses int8, the default data type is torch.float32 for quant now.
    # Consider setting the default data type to torch.float16 to speed up, but this may decrease model performance.
    # torch.set_default_dtype(torch.float16)
    if args.model_class == "llama":
      model = Int8LlamaForCausalLM.from_pretrained(args.model_path, quant_config, attn_implementation="eager", device_map="sequential")
    elif args.model_class == "baichuan":
      model = Int8BaichuanForCausalLM.from_pretrained(args.model_path, quant_config, attn_implementation="eager", device_map="sequential")
    elif args.model_class == "opt":
      model = Int8OPTForCausalLM.from_pretrained(args.model_path, quant_config, attn_implementation="eager", device_map="sequential")
    elif args.model_class == "mixtral":
      model = Int8MixtralForCausalLM.from_pretrained(args.model_path, quant_config, attn_implementation="eager", device_map="sequential")
    else:
      raise ValueError(
        f"Model type {args.model_class} are not supported for now.")
    inputs = tokenizer(
      args.prompt,
      padding=True,
      truncation=True,
      max_length=2048,
      return_tensors="pt").to("cuda")
    output_ids = model.generate(**inputs, max_new_tokens=20)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    print(outputs)


if __name__ == '__main__':
    main()