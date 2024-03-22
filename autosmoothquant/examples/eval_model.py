
import torch
import os
from tqdm import tqdm
import argparse
import collections

from benchmarks import evaluator
from benchmarks.models.quant_model import quant_model
from benchmarks.utils import make_table, pattern_match
from lm_eval.tasks import TaskManager
from autosmoothquant.utils import parse_quant_config, get_loaders

def parse_args():                                                                                                                                                                                                                                                                                                                               
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str,
                        default='int8-models/llama-13b', help='path contains model weight and quant config')
    parser.add_argument('--tokenizer-path', type=str,
                        default='int8-models/llama-13b', help='path contains tokenizer')
    parser.add_argument("--tasks", default="")
    parser.add_argument("--eval_ppl", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def update_results(results, new_result):
    for key, value in new_result.items():
        if key in results:
            results[key].update(value)
        else:
            results.update({key: value})


@torch.no_grad()
def benchmarks(lm, args):
    # for task in ["wikitext2", "c4"]:
    results = {}
    if args.eval_ppl:
        for task in ["wikitext2"]:
            _, testloader = get_loaders(
                task,
                seed=args.seed,
                model=args.tokenizer_path,
                seqlen=lm.max_length,
            )
            if "c4" in task:
                testenc = testloader
            else:
                testenc = testloader.input_ids

            nsamples = testenc.numel() // lm.max_length
            lm.model.eval()
            nlls = []
            for i in tqdm(range(nsamples)):
                batched_inps = testenc[
                    :, (i * lm.max_length) : ((i + 1) * lm.max_length)
                ].to("cuda:0")
                batched_labels = testenc[
                    :, (i * lm.max_length) : ((i + 1) * lm.max_length)
                ].to(lm.model.lm_head.weight.device)
                loss = lm.model(batched_inps, labels=batched_labels).loss
                neg_log_likelihood = loss.float() * lm.max_length
                nlls.append(neg_log_likelihood)
                if i == args.limit:
                    break

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.max_length))

            result = collections.defaultdict(dict)
            versions = collections.defaultdict(dict)
            n_shot = collections.defaultdict(dict)
            result[task]["ppl"] = ppl.item()
            versions[task] = 0
            n_shot[task] = 0
            t_results = {
                "results": dict(result),
                "versions": dict(versions),
                "n-shot": dict(n_shot),
            }
            update_results(results, t_results)
    if args.tasks != "":
        t_results = evaluator.simple_evaluate(
            lm,
            tasks=pattern_match(args.tasks.split(","), TaskManager().all_tasks),
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            batch_size=args.batch_size,
        )
        update_results(results, t_results)
    return results


if __name__ == "__main__":
    args = parse_args()
    config_path = os.path.join(args.model_path, "quant_config.json")
    quant_config = parse_quant_config(config_path)

    lm = quant_model(
        pretrained=args.model_path,
        batch_size=args.batch_size,
        quant_config=quant_config,
        tokenizer=args.tokenizer_path,
    )
    results = benchmarks(lm, args)
    print(make_table(results))
