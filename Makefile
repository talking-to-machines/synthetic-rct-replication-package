# Pipeline orchestrator. Run `make all` from the repository root to
# reproduce the full pipeline, or invoke any stage individually.
# All configuration is read from config.yaml; no arguments are passed here.

.PHONY: all data corpus train infer analysis clean

# Full pipeline in sequence: preprocess -> build corpus -> fine-tune -> infer -> analyse.
all: data corpus train infer analysis

# Clean and split original human data (surveys and RCTs).
# Reads:  data/human/
# Writes: data/processed/
data:
	python -m src.preprocess --config config.yaml

# Assemble the fine-tuning corpus from survey training files and RCT
# training splits.
# Reads:  data/processed/*/train.jsonl
# Writes: data/finetuning/train.jsonl
corpus:
	python -m src.build_corpus --config config.yaml

# LoRA fine-tuning for all open models defined in config.yaml.
# Reads:  data/finetuning/train.jsonl
# Writes: outputs/adapters/, outputs/logs/training/
train:
	python -m src.train --config config.yaml

# Inference for all model x condition x RCT combinations. Open models use
# logit extraction; GPT-5 uses the API. Logit columns are NA for GPT-5.
# Reads:  data/processed/rcts/*/test.jsonl, data/prompts/rcts/
# Writes: data/synthetic/, outputs/logs/inference/
infer:
	python -m src.infer --config config.yaml

# All evaluation metrics (ATE, Hellinger, weighted F1, PPI++), figures, and
# tables. Single R entry point that sources everything in analysis/.
# Reads:  data/processed/rcts/*/test.jsonl, data/synthetic/
# Writes: outputs/evaluation/, outputs/figures/, outputs/tables/
analysis:
	Rscript analysis/main.R

# Remove derived analysis outputs. Does not touch adapters, logs, or any
# data under data/ -- those are expensive to regenerate.
clean:
	rm -rf outputs/evaluation/* outputs/figures/* outputs/tables/*
