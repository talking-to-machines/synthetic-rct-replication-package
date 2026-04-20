# Computational requirements

---

## 1. Hardware

| Stage | GPU | VRAM | GPU count | CPU | RAM |
|-------|-----|------|-----------|-----|-----|
| Fine-tuning (8B) | | | | | |
| Fine-tuning (70B) | | | | | |
| Inference (8B) | | | | | |
| Inference (70B) | | | | | |
| GPT-5 API | N/A | N/A | N/A | | |
| R analysis | N/A | N/A | N/A | | |

---

## 2. Runtime

| Stage | Model | Wall time | GPU-hours |
|-------|-------|-----------|-----------|
| Preprocessing | all | | N/A |
| Corpus assembly | all | | N/A |
| Fine-tuning | llama_8b | | |
| Fine-tuning | llama_70b | | |
| Fine-tuning | qwen_8b | | |
| Fine-tuning | qwen_70b | | |
| Inference | llama_8b (instruct) | | |
| Inference | llama_8b (finetuned) | | |
| Inference | llama_70b (instruct) | | |
| Inference | llama_70b (finetuned) | | |
| Inference | qwen_8b (instruct) | | |
| Inference | qwen_8b (finetuned) | | |
| Inference | qwen_70b (instruct) | | |
| Inference | qwen_70b (finetuned) | | |
| Inference | gpt5 (instruct) | | N/A |
| R analysis | all | | N/A |
| **Total** | | | |

---

## 3. Cost estimate

**Cloud equivalent.** Estimated cost if reproduced on cloud infrastructure.

| Resource | Instance type | Rate | Hours | Cost |
|----------|---------------|------|-------|------|
| Fine-tuning (8B) | | | | |
| Fine-tuning (70B) | | | | |
| Inference (all open) | | | | |
| GPT-5 API | N/A | $/M tokens | | |
| **Total** | | | | |

---

## 4. Software versions

### Python

| Package | Version |
|---------|---------|
| Python | |
| PyTorch | |
| CUDA | |
| transformers | |
| peft | |
| accelerate | |
| datasets | |
| openai | |
| pandas | |
| numpy | |
| scikit-learn | |

Full pinned versions in `requirements.txt`.

### R

| Package | Version |
|---------|---------|
| R | |
| miceadds | |
| fixest | |
| ggplot2 | |
| data.table | |
| jsonlite | |
| xtable | |

Full pinned versions in `renv.lock`.

### System

| Component | Version |
|-----------|---------|
| OS | |
| NVIDIA driver | |
| cuDNN | |

