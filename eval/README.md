## 📋 Evaluation Tasks Overview

| # | Task Type | Description |
|---|-----------|-------------|
| 1 | **Marine Science QA** | Text-based question answering on marine science topics |
| 2 | **Marine Science VQA** | Visual question answering on marine science topics |
| 3 | **Sonar Image VQA** | Visual question answering for sonar images |
| 4 | **Marine Biology VQA** | Visual question answering for marine biology specimens |

### 🟢 Marine Science VQA Evaluation(API)
```bash
python eval/sci_eval.py --input_dir "YOUR_DATA_DIR" --type qa --eval_model gpt-4o
```

### 🟢 Marine Science QA Evaluation (API)
```bash
python eval/sci_eval.py --input_dir "YOUR_DATA_DIR" --type vqa --eval_model gpt-4o
```

### 🔵 Marine Science VQA Evaluation (Local Model)
```bash
python eval/sci_eval.py --input_dir "YOUR_DATA_DIR" --type vqa \
    --eval_model qwen3-vl --local \
    --local_model_path "YOUR_LOCAL_MODEL_PATH"
```

---

### 🎯 Sonar VQA Evaluation (API)

```bash
python sonar_eval.py --input_file "YOUR_DATA_PATH" --api_key "YOUR_API_KEY" --api_base "YOUR_API_BASE" --model_name gpt-4o
```

---

### 🐠 Marine Biology VQA Evaluation (API)

```bash
python rgb_eval.py --input_file "YOUR_DATA_PATH" --api_key "YOUR_API_KEY" --api_base "YOUR_API_BASE" --model_name gpt-4o
```