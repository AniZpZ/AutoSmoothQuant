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
```
cd smoothquant/examples
python3 export_int8_model.py 
```
### inference
TBA

## Supported models
Model support list:

| Models   | Sizes                       |
| ---------| ----------------------------|
| LLaMA-2  | 7B/13B/70B                  |
| LLaMA    | 7B/13B/30B/65B              |
| Mistral  | Soon                        |
| OPT      | 6.7B/13B/30B |
| Baichuan-2 | 13B                       |
| Baichuan | 13B                         |

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
