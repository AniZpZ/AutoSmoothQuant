import os
import torch
from autosmoothquant.models.llama import Int8LlamaForCausalLM
from autosmoothquant.models.baichuan import Int8BaichuanForCausalLM
from autosmoothquant.models.opt import Int8OPTForCausalLM
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM
from autosmoothquant.thirdparty.baichuan.modeling_baichuan import BaichuanForCausalLM

# quant_config = {
#            "qkv_proj": "per-tensor",
#            "o_proj": "per-tensor",
#            "gate_up_proj": "per-tensor",
#            "down_proj": "per-tensor"
#        }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str,
                        default='int8-models/llama-13b', help='path contains model weight and quant config')
    parser.add_argument('--tokenizer-path', type=str,
                        default='int8-models/llama-13b', help='path contains tokenizer')
    parser.add_argument('--model-class', type=str,
                        default='llama', help='currently support: llama, baichuan, opt')
    parser.add_argument('--prompt', type=str,
                        default='You are right, But Genshin Impact is', help='prompts')   
    args = parser.parse_args()
    return args

def parse_quant_config(config_path):
  data = {}
  with open(config_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
  return data

@torch.no_grad()
def main():
    args = parse_args()
    config_path = os.path.join(args.model_path, "quant_config.json")
    quant_config = parse_quant_config(config_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    if args.model_class == "llama":
      model = LlamaForCausalLM.from_pretrained(args.model_path, quant_config, device_map="auto", torch_dtype="auto")
    elif args.model_class == "baichuan":
      model = Int8BaichuanForCausalLM.from_pretrained(args.model_path, quant_config, device_map="auto", torch_dtype="auto")
    elif args.model_class == "opt":
      model = Int8BaichuanForCausalLM.from_pretrained(args. model_path, quant_config, device_map="auto", torch_dtype="auto")
    else:
      raise ValueError(
        f"Model type {args.model_class} are not supported for now. ")

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