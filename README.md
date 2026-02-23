<p align="center">
  <img src="assets/logo.png" width="256px">
</p>

<h3 align="center">A Multimodal Large-Scale Corpus for Data-Driven Ocean Intelligence</h3>

<div align="center">

🌐 [Home Page](https://oceangpt.github.io) 📄 [ArXiv Paper](https://arxiv.org/abs) 🤗 [Hugging Face](https://huggingface.co/datasets/zjunlp)

</div>

<img src="assets/intro.png" style="width:120%; height: auto;" align=center>

**OceanPile**, a large-scale multimodal corpus designed for ocean intelligence. It comprises three key components: **OceanCorpus**, a unified collection integrating sonar data, underwater imagery, marine science visuals, and scientific text from diverse authoritative sources; **OceanInstruction**, a high-quality instruction dataset synthesized via a novel pipeline guided by a hierarchical **Ocean Concept Knowledge Graph**; and **OceanBench**, a manually curated evaluation benchmark for rigorous assessment.

# 🔔 News

- 02-2026, we launched the OceanPile project.

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

# 📺 Quick Start
```
conda create -n py3.11 python=3.11
conda activate py3.11
pip install -r requirements.txt
```

### Download the Datasets
#### Download from HuggingFace
```shell
# use git lfs
git lfs install
git clone https://huggingface.co/zjunlp
```
or
```shell
# use huggingface-cli
pip install -U huggingface_hub
huggingface-cli download --resume-download zjunlp/xxx --local-dir xxx --local-dir-use-symlinks False
```
or
```bash
pip install datasets
```
```python
# Loading with Python:
from datasets import load_dataset
ds = load_dataset(
    "zjunlp/xxx",
)
```
# 📚 DataSets
## 📘 Dataset Summary
|                              | # images | # docs | # tokens | Download |
|------------------------------|----------|--------|----------|----------|
| OceanCorpus                  | 571M     | 101.2M | 43B      | [🤗 Download](https://huggingface.co/datasets/zjunlp) |
| OceanInstruction (textual data)       | -        | 77.7M  | 33B      | [🤗 Download](https://huggingface.co/datasets/zjunlp) |
| OceanInstruction (visual data)       | 29.9M    | 7.3M   | 2.4B     | [🤗 Download](https://huggingface.co/datasets/zjunlp) |
| OceanInstruction (task-specific data) | 29.9M    | 7.3M   | 2.4B     | [🤗 Download](https://huggingface.co/datasets/zjunlp) |
| OceanBench                   | 29.9M    | 7.3M   | 2.4B     | [🤗 Download](https://huggingface.co/datasets/zjunlp) |

More details about these datasets and our processing steps [can be found in our paper](https://arxiv.org/abs).
### 🏗️ Dataset Structure
- **image** : high-resolution image croped from oceanography paper/webstite/dataset (open-assess)

### 🔏 License
This dataset is released under
[Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

- **NonCommercial**
  - This dataset may not be used for commercial purposes. Prohibited uses include, but are not limited to, selling the dataset, incorporating it into commercial products or services, or using it in workflows whose primary purpose is to obtain direct commercial advantage.

- **Share Alike**
  - If you remix, transform, or build upon this dataset, or distribute adapted versions of it, you must release your contributions under the same CC BY-NC-SA 4.0 license.

- **Important Notes**

  - The source papers for this dataset are published under open-access licenses, and the data are likewise subject to the licensing terms of the original papers.

  - Models trained using this dataset should respect the NonCommercial restriction when used or redistributed.

  - Users are responsible for ensuring compliance with the license in their specific use cases.


# 🚩 Citation

If this OceanPile paper or datasets is helpful, please kindly cite as this:

```bibtex

```
