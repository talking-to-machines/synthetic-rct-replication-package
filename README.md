# README.md (root)

```
# [Paper title]

[Authors, affiliations]

[One-paragraph abstract]

[Badges: licence, Python version, R version]

## Overview

[Placeholder]

## Key hyperparameters

### Fine-tuning (LoRA)

LoRA:
- rank (r)
- alpha
- dropout
- target modules

Training:
- epochs
- batch size
- gradient accumulation steps
- effective batch size
- learning rate
- weight decay
- warmup ratio
- learning rate scheduler
- optimizer
- precision
- max sequence length
- seeds

### Inference

Open models:
- target tokens
- renormalisation
- precision
- batch size
- max sequence length

GPT-5 (API):
- temperature
- seed
- max tokens

Note: the inference stage includes logit extraction as an intermediate
step for all open models. Logit columns are included in the synthetic
data output for all models; for GPT-5 they are NA, since the API does
not expose token-level logits.

## Quick start

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    make all

## Data availability

Per-source access statement for each survey and RCT:
- Source name, citation, DOI
- Access conditions (public / restricted / on request)
- Licence
- Download instructions or contact

## Computational requirements

Hardware used (GPU model, VRAM, count).
Expected runtime per pipeline stage.
Estimated cost (cloud equivalent).
Pointer to docs/computational_requirements.md for full detail.

## Step-by-step reproduction

What each make target does:
  make data      Human data -> cleaned, split
  make corpus    Surveys + RCT training splits -> fine-tuning corpus
  make train     LoRA fine-tuning, all open models
  make infer     Inference for all models (logit extraction for open
                 models, API query for GPT-5) -> data/synthetic/
  make analysis  All metrics, figures, tables (R)
  make all       Full pipeline in sequence

## Pretrained adapters

Project publishes one LoRA adapter per fine-tuned open model (4 in total:
llama_8b, llama_70b, qwen_8b, qwen_70b). Each adapter is hosted on the
HuggingFace Hub and mirrored in outputs/adapters/. A table lists, for
each adapter: the model name, the HuggingFace Hub URL, the number of
trainable parameters, and the on-disk size of the adapter weights.

To load an adapter:

    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, "outputs/adapters/llama_8b")

## Figure and table index

Pointer to docs/figure_table_map.md.

## Licence

Code licence (MIT or Apache-2.0).
Data licence (per source -- stated in data/README.md).
```
