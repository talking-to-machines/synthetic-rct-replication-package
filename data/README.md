# Data

## Overview

This directory contains all data used in the project, organised by
stage in the pipeline:

```
data/
├── human/          Original human data (read-only inputs)
│   ├── surveys/    Survey data used for fine-tuning
│   └── rcts/       RCT data used for fine-tuning and evaluation
│
├── processed/      Cleaned, standardised, and split
│   ├── surveys/    Cleaned surveys + fine-tuning splits
│   └── rcts/       Cleaned RCTs + train/test splits
│
├── finetuning/     Assembled fine-tuning corpus
│
├── prompts/        Per-source prompt templates
│   ├── surveys/
│   └── rcts/
│
└── synthetic/      LLM-generated predictions
```

---

## Data sources

### Surveys

| ID | Name | Access | Licence |
|----|------|--------|---------|
| survey_01 | "TBD" | "TBD" | "TBD" |
| survey_02 | "TBD" | "TBD" | "TBD" |

### RCTs

| ID | Name | Access | Licence |
|----|------|--------|---------|
| rct_01 | "TBD" | "TBD" | "TBD" |
| rct_02 | "TBD" | "TBD" | "TBD" |
| rct_03 | "TBD" | "TBD" | "TBD" |
| rct_04 | "TBD" | "TBD" | "TBD" |

RCTs serve a dual role: their training splits contribute to the
fine-tuning corpus, while their test splits are reserved exclusively
for evaluation.

Surveys are used entirely for fine-tuning; they have no test split.

Full provenance details for each source are in
`data/human/{surveys,rcts}/{source_id}/{source_id}_provenance.md`.

---

## Directory contents

### `human/`

Original data as published or obtained from the source. Read-only --
no files in this directory are modified by the pipeline.

**RCTs:**
```
human/rcts/rct_01/
├── rct_01_data.csv             Original response data
├── rct_01_instrument.pdf       Survey/experiment instrument
└── rct_01_provenance.md        Citation, DOI, access, IRB, licence
```

**Surveys:**
```
human/surveys/survey_01/
├── survey_01_data.csv          Original response data
└── survey_01_provenance.md     Citation, DOI, access, licence
```

### `processed/`

Cleaned, standardised versions of the original data, plus train/test
splits formatted for model consumption. Codebooks document the cleaned
versions (not the originals).

**RCTs:**
```
processed/rcts/rct_01/
├── rct_01_clean.csv            Cleaned, standardised data
├── rct_01_train.jsonl          Training split (feeds fine-tuning corpus)
├── rct_01_test.jsonl           Test split (used for evaluation only)
├── rct_01_codebook.md          Source metadata + pointer to CSV
└── rct_01_codebook.csv         Variable definitions for rct_01_clean.csv
```

**Surveys:**
```
processed/surveys/survey_01/
├── survey_01_clean.csv         Cleaned, standardised data
├── survey_01_train.jsonl       Full dataset formatted for fine-tuning
├── survey_01_codebook.md       Source metadata + pointer to CSV
└── survey_01_codebook.csv      Variable definitions for survey_01_clean.csv
```

### `finetuning/`

The assembled fine-tuning corpus, built from all survey training files
and all RCT training splits.

```
finetuning/
├── train.jsonl                 Combined corpus (one example per line)
└── corpus_description.md       Sources, counts, assembly procedure
```

### `prompts/`

Prompt templates with `{placeholders}` filled at inference time from
subject-level data. Templates are constant across all models.

```
prompts/
├── rcts/
│   ├── rct_01.txt
│   ├── rct_02.txt
│   ├── rct_03.txt
│   └── rct_04.txt
└── surveys/
    ├── survey_01.txt
    └── survey_02.txt
```

### `synthetic/`

LLM-generated predictions. One CSV per RCT x model x condition
combination. Flat directory -- no subdirectories.

```
synthetic/
├── codebook_synthetic.md            Source metadata + pointer to CSV
├── codebook_synthetic.csv           Variable definitions (shared schema)
├── rct_01_llama_8b_instruct.csv
├── rct_01_llama_8b_finetuned.csv
├── rct_01_llama_70b_instruct.csv
├── rct_01_llama_70b_finetuned.csv
├── rct_01_qwen_8b_instruct.csv
├── rct_01_qwen_8b_finetuned.csv
├── rct_01_qwen_70b_instruct.csv
├── rct_01_qwen_70b_finetuned.csv
├── rct_01_gpt5_instruct.csv
└── ...                              (same pattern for rct_02--rct_04)
```

---

## File formats

| Directory | Format | Description |
|-----------|--------|-------------|
| `human/` | CSV (or original format) | Data as published |
| `processed/` | CSV (cleaned) + JSONL (splits) | One JSON object per subject per line |
| `finetuning/` | JSONL | One training example per line |
| `prompts/` | Plain text | Templates with `{placeholders}` |
| `synthetic/` | CSV | One row per subject |

---

## Codebooks

Every data file in the project has a codebook documenting its
variables. All codebooks follow the same two-component structure:

1. **Markdown header** (`{name}_codebook.md`): a Source table
   identifying the file being documented, the script that generates
   it, and a one-line description. Points to the CSV below.
2. **CSV file** (`{name}_codebook.csv`): the variables table,
   machine-readable, with five columns:

| Column | Definition |
|--------|------------|
| `Variable` | Column name exactly as it appears in the CSV header |
| `Type` | One of: `string`, `integer`, `float`, `boolean`, `date` |
| `Description` | What the variable represents |
| `Values` | Permitted values, range, or format |
| `Missing` | How missing values are coded, or `none expected` |

### Codebook locations

| Data | Codebook | Scope |
|------|----------|-------|
| `processed/rcts/{rct_id}/` | `{rct_id}_codebook.md` + `.csv` | Per RCT |
| `processed/surveys/{survey_id}/` | `{survey_id}_codebook.md` + `.csv` | Per survey |
| `synthetic/` | `codebook_synthetic.md` + `.csv` | Shared across all synthetic files |

### Synthetic data schema

All synthetic CSV files share the schema defined in
`data/synthetic/codebook_synthetic.csv`. Variables are grouped as
follows:

- **Subject and outcome:** `subject_id`, `treatment`, `outcome`,
  `prediction`
- **Logits and probabilities:** `logit_yes`, `logit_no`, `prob_yes`
- **Run metadata:** `date`
- **Model descriptors:** `model_family`, `model_version`, `model_size`,
  `model_id`, `fine_tuned`
- **LoRA configuration:** `lora_r`, `lora_alpha`, `lora_dropout`,
  `lora_target_modules` (NA if `fine_tuned=FALSE`)
- **Training hyperparameters:** `train_epochs`, `train_batch_size`,
  `train_grad_accum_steps`, `train_effective_batch_size`, `train_lr`,
  `train_lr_scheduler`, `train_warmup_ratio`, `train_max_grad_norm`,
  `train_weight_decay`, `train_optimizer`, `train_precision`,
  `train_max_seq_length`, `train_seed` (NA if `fine_tuned=FALSE`)
- **Inference hyperparameters:** `infer_precision`, `infer_batch_size`,
  `infer_max_seq_length` (open models); `infer_temperature`,
  `infer_seed`, `infer_max_tokens` (API models)

For GPT-5, `logit_yes` and `logit_no` are `NA` because the API does
not expose token-level logits. All training and LoRA columns are `NA`
for instruct (non-fine-tuned) models.

---

## Train/test splits

RCT data is split into training and test sets using subject-level
holdout, stratified by treatment. Default parameters (overridable
per RCT in `config.yaml`):

| Parameter | Value |
|-----------|-------|
| Method | Subject-level holdout |
| Test fraction | 0.2 |
| Stratification | By treatment |

Training splits feed the fine-tuning corpus via `make corpus`.
Test splits are used exclusively for evaluation via `make analysis`.

Surveys have no test split. The entire cleaned dataset is converted
to a training file.

---

## Naming conventions

### Original human data

```
{source_id}_data.csv
{source_id}_instrument.pdf          (RCTs only)
{source_id}_provenance.md
```

### Processed data

```
{source_id}_clean.csv
{source_id}_train.jsonl
{source_id}_test.jsonl              (RCTs only)
{source_id}_codebook.md
{source_id}_codebook.csv
```

### Synthetic data

```
{rct_id}_{model}_{condition}.csv
```

Where `{model}` is one of: `llama_8b`, `llama_70b`, `qwen_8b`,
`qwen_70b`, `gpt5`. And `{condition}` is one of: `instruct`,
`finetuned` (GPT-5 has `instruct` only).

---

## Reproduction

```
make data       Cleans and splits human data -> data/processed/
make corpus     Assembles fine-tuning corpus  -> data/finetuning/
make infer      Runs inference                -> data/synthetic/
```
