# Model card template

This template is used to create model cards for all 9 model x condition
configurations in the project. There are three card types:

- **Instruct (open-source):** Pre-trained instruction-tuned model, no
  fine-tuning. 4 cards: llama_8b, llama_70b, qwen_8b, qwen_70b.
- **Fine-tuned (LoRA adapter):** LoRA adapter trained on the project's
  fine-tuning corpus. 4 cards: llama_8b, llama_70b, qwen_8b, qwen_70b.
- **Instruct (API):** Proprietary model accessed via API. 1 card: gpt5.

Sections marked [FINETUNED ONLY] or [API ONLY] should be included only
for the relevant card type. All other sections apply to all types.

---

# YAML front-matter

The YAML block at the top of each .md file is parsed by HuggingFace Hub
for discovery, filtering, and display. Fields vary by card type.

## Instruct (open-source)

```yaml
---
language:
  - en
license: "TBD"
library_name: transformers
base_model: <HuggingFace model ID>             # e.g. meta-llama/Llama-3.1-8B-Instruct
pipeline_tag: text-generation
tags:
  - instruct
  - rct-replication
  - silicon-sampling
---
```

## Fine-tuned (LoRA adapter)

```yaml
---
language:
  - en
license: "TBD"
library_name: peft
base_model: <HuggingFace model ID>             # e.g. meta-llama/Llama-3.1-8B-Instruct
base_model_relation: adapter
pipeline_tag: text-generation
tags:
  - lora
  - rct-replication
  - silicon-sampling
datasets:
  - <dataset ID or repo path>
---
```

## Instruct (API)

```yaml
---
language:
  - en
license: "TBD"
tags:
  - api
  - instruct
  - rct-replication
  - silicon-sampling
---
```

---

# Markdown body

Below is the full template. Replace placeholders in angle brackets.
Delete section-level instructions (in italics) when populating an
actual card. All hyperparameter values should be resolved -- not
pointers to config.yaml.


```markdown
# <Model name> -- <condition>

## Model description

<One-paragraph description. What this model is, what base model it
builds on, what condition it represents (instruct or fine-tuned), and
what it is intended for.>

- **Base model:** <HuggingFace model ID and exact revision hash>
- **Model family:** <llama / qwen / gpt5>
- **Parameters (base):** <total parameter count>
- **Condition:** <instruct / finetuned>
- **Developed by:** <authors>
- **Funded by:** <funding sources>
- **Paper:** <citation or link>
- **Repository:** <GitHub URL>
- **Licence:** <licence name>


## Intended uses

### Direct use

*This model is used to generate synthetic experimental responses that
replicate human behaviour in randomised controlled trials (RCTs). It
is evaluated by comparing its predicted response distributions against
human ground-truth data.*

### Out-of-scope uses

*This model is not intended for:*
- *Deployment in production systems*
- *Clinical or policy decision-making*
- *Replacing human participants where ethical oversight is required*
- *Generating synthetic data for purposes unrelated to RCT replication*


## Training details [FINETUNED ONLY]

*This section is included only for fine-tuned LoRA adapter cards.
Instruct and API cards omit it entirely.*

### Training data

*Describe the fine-tuning corpus composition. List which surveys and
RCT training splits contributed, the total number of examples, and the
proportion from each source.*

| Source | Type | N examples | Proportion |
|--------|------|------------|------------|
| | | | |
| | | | |
| **Total** | | | |

*Point to `data/finetuning/corpus_description.md` for full detail.*

### LoRA configuration

*All values resolved for this specific model.*

| Parameter | Value |
|-----------|-------|
| Rank (r) | |
| Alpha | |
| Dropout | |
| Target modules | |

### Training hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | |
| Batch size (per device) | |
| Gradient accumulation steps | |
| Effective batch size | |
| Learning rate | |
| Weight decay | |
| Warmup ratio | |
| LR scheduler | |
| Optimizer | |
| Precision | |
| Max sequence length | |
| Seeds | |

### Training procedure

*Any additional training procedure details: preprocessing, data
loading strategy, early stopping criteria (if any), etc.*


## Inference configuration

*Describe how predictions are generated from this model.*

### Open-source models (instruct and finetuned)

| Parameter | Value |
|-----------|-------|
| Target tokens | |
| Renormalisation | p(yes) / (p(yes) + p(no)) |
| Precision | |
| Batch size | |
| Max sequence length | |

*Inference includes logit extraction as an intermediate step. For each
subject, the model produces logits over the target tokens, which are
then renormalised to probabilities.*

### API models (GPT-5) [API ONLY]

| Parameter | Value |
|-----------|-------|
| API model string | |
| Temperature | |
| Seed | |
| Max tokens | |
| Query dates | |

*The API does not expose token-level logits. Logit columns in the
synthetic output CSVs are NA for this model.*


## Evaluation

### Testing data

*Evaluation uses the held-out test splits of 4 RCTs. Each test split
is constructed via subject-level holdout, stratified by treatment.
See config.yaml for split parameters.*

### Metrics

| Metric | Description |
|--------|-------------|
| ATE delta | Difference in average treatment effect between synthetic and human data |
| Hellinger distance | Per-treatment-arm distributional distance between synthetic and human |
| Weighted F1 | Individual-level prediction fidelity |
| PPI++ power gain | Statistical power gain from integrating synthetic predictions via prediction-powered inference |

### Results

| RCT | ATE delta | Hellinger | Weighted F1 | PPI++ gain |
|-----|-----------|-----------|-------------|------------|
| rct_01 | | | | |
| rct_02 | | | | |
| rct_03 | | | | |
| rct_04 | | | | |


## Technical specifications

### Hardware

| Component | Detail |
|-----------|--------|
| GPU | |
| VRAM | |
| GPU count | |

### Training time [FINETUNED ONLY]

| Metric | Value |
|--------|-------|
| Wall time | |
| GPU-hours | |

### Inference time

| Metric | Value |
|--------|-------|
| Wall time (all 4 RCTs) | |
| GPU-hours | |


## Environmental impact [FINETUNED ONLY]

*Estimated using the ML Impact Calculator
(https://mlco2.github.io/impact/).*

| Metric | Value |
|--------|-------|
| Hardware type | |
| Hours used | |
| Cloud provider | |
| Region | |
| Estimated CO2 (kg) | |


## Bias, risks, and limitations

### Known limitations

*Describe known failure modes, sensitivity to prompt format,
demographic gaps in training data, etc.*

### Population coverage

*Describe the demographic composition of the training data (fine-tuned
models) or the base model's known training data characteristics
(instruct models). Flag any populations that are underrepresented
or absent.*

### Reproducibility risks [API ONLY]

*For API models: document version drift, deprecation risk, and the
fact that results may not be exactly reproducible even at
temperature=0 due to API-side non-determinism.*


## How to use

### Instruct (open-source)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("<base_model_id>",
                                              torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("<base_model_id>")
```

### Fine-tuned (LoRA adapter)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("<base_model_id>",
                                             torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, "outputs/adapters/<model_id>")
tokenizer = AutoTokenizer.from_pretrained("<base_model_id>")
```

### API (GPT-5)

```python
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="<api_model_string>",
    temperature=0.0,
    seed=42,
    max_tokens=5,
    messages=[{"role": "user", "content": "<prompt>"}]
)
```


## Model card authors

<Names of the people who wrote this model card.>


## Model card contact

<Email or other contact for questions about this model card.>
```
