<p align="center">
  <img src="assets/logo.png" width="256px">
</p>

<h3 align="center">OceanPile: A Large-Scale Multimodal Ocean Corpus for Foundation Models</h3>

<div align="center">

🌐 [Home Page](https://github.com/OceanGPT/OceanPile) 📄 [Paper](https://arxiv.org/abs) 🤗 [Hugging Face](https://huggingface.co/collections/zjunlp/oceanpile) 🌊 [Website](http://data.oceangpt.blue/en/)

</div>

<img src="assets/intro.png" style="width:120%; height: auto;" align=center>

**OceanPile**, a large-scale multimodal corpus designed for ocean intelligence. It comprises three key components: **OceanCorpus**, a unified collection integrating sonar data, underwater imagery, marine science visuals, and scientific text from diverse authoritative sources; **OceanInstruction**, a high-quality instruction dataset synthesized via a novel pipeline guided by a hierarchical **Ocean Concept Knowledge Graph**; and **OceanBench**, a manually curated evaluation benchmark for rigorous assessment.

# 🔔 News
- 04-2026, We released the OceanPile [datasets](https://huggingface.co/collections/zjunlp/oceanpile).
- 02-2026, We launched the OceanPile project.

**Contents:**
- [🌟Overview](#-overview)
- [🔔 News](#-news)
- [📺 Quick Start](#-quick-start)
- [📚 Datasets](#-datasets)
- [🚩 Citation](#-citation)

# 🌟 Overview

<div align="center">
<img src="assets/framework.png" width="90%">
</div>

As illustrated, our approach begins with constructing a domain-specific knowledge graph by extracting and enriching concepts from authoritative scientific literature and structured marine data. Guided by this knowledge graph, we synthesize and validate instruction-response pairs, ensuring high-quality data that reflects the nature of marine science.

# 📚 DataSets
### 📘 Dataset Summary
|                              | # images | # samples               | # tokens | Download |
|------------------------------|----------|-------------------------|----------|----------|
| OceanCorpus                  | -        | &gt; 300K PDF documents | &gt; 50B | [🤗 Download](https://huggingface.co/datasets/zjunlp/OceanCorpus) |
| OceanInstruction       | 25,730        | 141,124                   | -        | [🤗 Download](https://huggingface.co/datasets/zjunlp/OceanInstruction) |
| OceanBench                   | 1,367    | 1,469                    | -        | [🤗 Download](https://huggingface.co/datasets/zjunlp/OceanBenchmark) |

More details about these datasets can be found in our [Paper](https://arxiv.org/abs) or [Hugging Face](https://huggingface.co/collections/zjunlp/oceanpile).

### 🤖 Model Zoo

| Model Name | Base Model  | Domain             | Download |
|------------|-------------|--------------------|----------|
| OceanGPT-o-8B-OceanPile-Sci | Qwen3-VL-8B-Instruct | Marine Science VQA | [🤗 Download](https://huggingface.co/zjunlp/OceanGPT-o-8B-OceanPile-Sci) |
| OceanGPT-basic-30B-OceanPile-Sci | Qwen3-30B-A3B-Instruct | Marine Science QA  | [🤗 Download](https://huggingface.co/zjunlp/OceanGPT-basic-30B-OceanPile-Sci) |
| OceanGPT-o-8B-OceanPile-Sonar | Qwen3-VL-8B-Instruct  | Sonar Image VQA    | [🤗 Download](https://huggingface.co/zjunlp/OceanGPT-o-8B-OceanPile-Sonar) |
| OceanGPT-o-8B-OceanPile-Bio | Qwen3-VL-8B-Instruct  | Marine Biology VQA | [🤗 Download](https://huggingface.co/zjunlp/OceanGPT-o-8B-OceanPile-Bio) |

# 📺 Quick Start
```
conda create -n py3.11 python=3.11
conda activate py3.11
pip install transformers
pip install datasets
```

#### Download the Datasets from HuggingFace
```python
# Loading with Python:
from datasets import load_dataset
dataset = load_dataset("zjunlp/OceanBenchmark")
```

#### Download the Models from HuggingFace
```python
# Loading with Python:
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "zjunlp/OceanGPT-o-8B-OceanPile-Sci", dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained("zjunlp/OceanGPT-o-8B-OceanPile-Sci")

```

#### Inference
```python
# Loading with Python:
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "zjunlp/OceanGPT-o-8B-OceanPile-Sci", dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained("zjunlp/OceanGPT-o-8B-OceanPile-Sci")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "file:///path/to/your/image.jpg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

### 🔏 License
This dataset is released under MIT License.

# 🚩 Citation

If this OceanPile paper or datasets is helpful, please kindly cite as this:

```bibtex

```

💐 Citation for Models:
```bibtex
@article{bi2024oceangpt,
  title={OceanGPT: A Large Language Model for Ocean Science Tasks},
  author={Bi, Zhen and Zhang, Ningyu and Xue, Yida and Ou, Yixin and Ji, Daxiong and Zheng, Guozhou and Chen, Huajun},
  journal={arXiv preprint arXiv:2310.02031},
  year={2024}
}
```
