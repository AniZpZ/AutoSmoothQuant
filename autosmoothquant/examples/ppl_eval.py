import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
sys.path.append("./autosmoothquant")
from models.phi2 import Int8PhiForCausalLM
from models.llama import Int8LlamaForCausalLM
from models.qwen2 import Int8Qwen2ForCausalLM
import tqdm
import os
from datasets import load_dataset
import argparse
from utils import get_config, get_model_architecture, build_model_and_tokenizer, parse_quant_config
from transformers.models.phi.modeling_phi import PhiForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--model_path", type=str, default="quantized_model/qwen2/qwen2-smoothquant")


args = parser.parse_args()
alpha = args.alpha
model_path = args.model_path


class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=40):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)

        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        n_samples = self.n_samples if self.n_samples else self.dataset.size(1) // 2048
        for i in tqdm.tqdm(range(n_samples), desc="Evaluating..."):
            batch = self.dataset[:, (i * 2048) : ((i + 1) * 2048)].to(model.device)
            with torch.no_grad():
                lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)

        return torch.exp(torch.stack(nlls).sum() / (n_samples * 2048))


tokenizer = AutoTokenizer.from_pretrained(model_path)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
evaluator = Evaluator(dataset, tokenizer, "cuda")
config_path = os.path.join(args.model_path, "quant_config.json")
quant_config = parse_quant_config(config_path)

model = Int8Qwen2ForCausalLM.from_pretrained(args.model_path, quant_config,
                                                       device_map="sequential")
# model = Int8PhiForCausalLM.from_pretrained(args.model_path, quant_config,
#                                                        device_map="sequential")
# model = Int8PhiForCausalLM.from_pretrained(args.model_path, quant_config, attn_implementation="eager",
#                                                        device_map="sequential")
# model = PhiForCausalLM.from_pretrained(
#             args.model_path, device_map="auto", torch_dtype=torch.float16)
# model = Qwen2ForCausalLM.from_pretrained(
#             args.model_path, device_map="auto", torch_dtype=torch.float16)
ppl = evaluator.evaluate(model)
print(f"Perplexity: {ppl}")
