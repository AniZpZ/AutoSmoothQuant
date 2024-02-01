# AutoSmoothQuant

AutoSmoothQuant is an easy-to-use package for implementing smoothquant for LLMs. AutoSmoothQuant speeds up model inference under various workloads. AutoSmoothQuant was created and improved upon from the [original work](https://github.com/mit-han-lab/smoothquant) from MIT.

## Install
### Prerequisites
- Your GPU(s) must be of Compute Capability 8.0 or higher. Amphere and later architectures are supported.
- Your CUDA version must be CUDA 11.4 or later.
### Build from source
Currently this repo only support build form source. We will release package soon.

```
git clone https://github.com/AniZpZ/AutoSmoothQuant.git
cd AutoSmoothQuant
pip install -e .
```

## Usage
### quantize model
First add a config file named "quant_config.json" to model path.
For currenttly supported models, config should be like:

```json
{
  "qkv": "per-tensor",
  "out": "per-tensor",
  "fc1": "per-tensor",
  "fc2": "per-tensor"
}
```

"qkv" stands for QKV matmul of attention, "out" stands for out matmul of attention.
"fc1" and "fc2" are the layers of the FFNs, which might be referred to as "gate_up" and "down" in Llama-like models.
You can set the value to "per-tensor" or "per-token" to perform the quant granularity you want.

Once config is set, generate scales and do model quantization with following command:
```
cd autosmoothquant/examples
python3 smoothquant_model.py --model-path=/path/to/model --quantize-model=True --generate-scale=True --dataset-path=/path/to/dataset
```

use following command for more information 
```
python smoothquant_model.py -help 
```
### inference
- inference with vLLM 
  
  Comming soon (this [PR](https://github.com/vllm-project/vllm/pull/1508) could be reference)

- inference in this repo
```
cd autosmoothquant/examples
python3 test_model.py --model-path=/path/to/model --tokenizer-path=/path/to/tokenizer --model-class=llama --prompt="something to say"
```

### benchmark
  Comming soon  (this [PR](https://github.com/vllm-project/vllm/pull/1508) could be reference)

## Supported models
Model support list:

| Models   | Sizes                       |
| ---------| ----------------------------|
| LLaMA-2  | 7B/13B/70B                  |
| LLaMA    | 7B/13B/30B/65B              |
| Mistral  | Soon                        |
| OPT      | 6.7B/13B/30B                |
| Baichuan-2 | 13B (7B Soon)             |
| Baichuan | 13B (7B Soon)               |

## Performance and inference efficency
Detailed data comming soon

Cases:

[codellama-13b with A40](https://github.com/vllm-project/vllm/pull/1508#issuecomment-1824133140). Tested with vLLM

[llama-13b with A100](https://github.com/vllm-project/vllm/pull/1508#issuecomment-1853826414). Tested with vLLM






## Reference
If you find SmoothQuant useful or relevant to your research, please cite their paper:

```bibtex
@InProceedings{xiao2023smoothquant,
    title = {{S}mooth{Q}uant: Accurate and Efficient Post-Training Quantization for Large Language Models},
    author = {Xiao, Guangxuan and Lin, Ji and Seznec, Mickael and Wu, Hao and Demouth, Julien and Han, Song},
    booktitle = {Proceedings of the 40th International Conference on Machine Learning},
    year = {2023}
}
```
