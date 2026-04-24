"""Inference for all model x condition x RCT combinations.

Reads:  data/processed/rcts/*/test.jsonl, data/prompts/rcts/
Writes: data/synthetic/, outputs/logs/inference/

Extracted from archived `synthetic_experiment_gpt.py` and
`synthetic_experiment_togetherai.py`. Entry points preserved as
`run_gpt_inference` and `run_togetherai_inference` for backwards
compatibility with those scripts. `run_togetherai_inference_codebook`
produces data/synthetic/ output conforming to
data/synthetic/codebook_synthetic.csv.
"""

import argparse
import datetime
import json
import math
import os

import pandas as pd
import yaml
from openai import OpenAI

from src.data.cleaning import load_data
from src.data.formatting import generate_synthetic_experiment_prompts
from src.models.api_client import (
    batch_query,
    create_batch_file,
    inference_endpoint_query,
)
from src.utils.config import OPENAI_API_KEY


def _parse_logit_response(json_str: str) -> pd.Series:
    """Parse Yes/No logits from a JSON string of per-position logprobs.

    Strategy: scan the sampled-token sequence for the first "Yes" or "No"
    (case-insensitive). If found, use that position's top-k to extract
    Yes/No logprobs. If neither appears anywhere in the sequence, fall back
    to position 0 (first token).
    """
    try:
        parsed = json.loads(json_str)
        per_position = parsed.get("per_position_logprobs", [])

        if not per_position:
            return pd.Series(
                {
                    "llm_response_parsed": parsed.get("response", ""),
                    "logprob_yes": None,
                    "logprob_no": None,
                    "prob_yes": None,
                    "prob_no": None,
                }
            )

        chosen_idx = None
        for i, pos in enumerate(per_position):
            tok = (pos.get("sampled_token") or "").strip().lower()
            if tok in ("yes", "no"):
                chosen_idx = i
                break
        if chosen_idx is None:
            chosen_idx = 0

        chosen = per_position[chosen_idx]
        chosen_token = chosen.get("sampled_token") or ""

        yes_probs = []
        no_probs = []
        for entry in chosen.get("top_logprobs", []):
            token = (entry.get("token") or "").strip().lower()
            if token == "yes":
                yes_probs.append(math.exp(entry["logprob"]))
            elif token == "no":
                no_probs.append(math.exp(entry["logprob"]))

        prob_yes = sum(yes_probs) if yes_probs else None
        prob_no = sum(no_probs) if no_probs else None

        return pd.Series(
            {
                "llm_response_parsed": chosen_token,
                "logprob_yes": math.log(prob_yes) if prob_yes is not None else None,
                "logprob_no": math.log(prob_no) if prob_no is not None else None,
                "prob_yes": prob_yes,
                "prob_no": prob_no,
            }
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        return pd.Series(
            {
                "llm_response_parsed": "",
                "logprob_yes": None,
                "logprob_no": None,
                "prob_yes": None,
                "prob_no": None,
            }
        )


def run_gpt_inference(request: dict) -> pd.DataFrame:
    """Run inference for GPT models via the OpenAI Batch API.

    Expects `request` with keys: prompt_file, question, data_file_path,
    experiment_round, version, model, scenario.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    data, var_labels = load_data(request["data_file_path"])

    with open(request["prompt_file"]) as f:
        prompt_cfg = json.load(f)

    if prompt_cfg["study"] not in ["duch_et_al_2023"]:
        raise ValueError(f"Study {prompt_cfg['study']} is not supported.")

    prompts = generate_synthetic_experiment_prompts(
        data,
        prompt_cfg["profile_vars"],
        prompt_cfg["system_template"],
        prompt_cfg["user_template"],
        prompt_cfg["treatment"],
        id_column="SubjectID",
        treatment_column="individual_treatment",
        var_labels=var_labels,
    )

    is_logit = prompt_cfg["study"] == "duch_et_al_2023"
    batch_file_dir = create_batch_file(
        prompts,
        system_message_field="system_message",
        user_message_field="question_prompt",
        batch_file_name="batch_input_llm_replication_experiment.jsonl",
        logit=is_logit,
        model=request["model"],
    )

    llm_responses = batch_query(
        client,
        batch_input_file_dir=batch_file_dir,
        batch_output_file_dir="batch_output_llm_replication_experiment.jsonl",
        logit=is_logit,
    )
    llm_responses.rename(columns={"query_response": "llm_response"}, inplace=True)

    id_col = "SubjectID"
    prompts_with_responses = pd.merge(prompts, llm_responses, on="custom_id")
    data_with_responses = pd.merge(
        data, prompts_with_responses, on=id_col, suffixes=("", "_y")
    )

    data_with_responses["user_response"] = data_with_responses[request["question"][0]]

    if is_logit:
        logit_columns = data_with_responses["llm_response"].apply(_parse_logit_response)
        data_with_responses = pd.concat([data_with_responses, logit_columns], axis=1)

    data_with_responses["model"] = request["model"]
    data_with_responses["scenario"] = request["scenario"]

    output_path = os.path.join(
        "data/synthetic", f"{request['experiment_round']}_{request['version']}.xlsx"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data_with_responses.to_excel(output_path, index=False)
    return data_with_responses


def _model_family_version_size(model_cfg: dict, model_key: str) -> tuple:
    """Derive (model_family, model_version, model_size) for the codebook columns."""
    family = model_cfg.get("family", "")
    key_l = model_key.lower()
    if "70b" in key_l:
        size = "70B"
    elif "8b" in key_l:
        size = "8B"
    else:
        size = None
    version_map = {"llama": "3.1", "qwen": "3", "gpt5": "5"}
    version = version_map.get(family)
    return family, version, size


def _renormalised_prob_yes(row: pd.Series) -> float | None:
    p_yes, p_no = row.get("prob_yes"), row.get("prob_no")
    if p_yes is None or p_no is None:
        return None
    denom = p_yes + p_no
    if denom <= 0:
        return None
    return p_yes / denom


def run_togetherai_inference_codebook(
    config_path: str,
    rct_id: str,
    model_key: str,
    model_id: str,
    condition: str = "finetuned",
    data_file_path: str | None = None,
    output_csv: str | None = None,
    ft_corpus: str | None = None,
) -> pd.DataFrame:
    """Run Together AI inference on an RCT holdout; emit codebook-schema CSV.

    Writes to `data/synthetic/{rct_id}_{model_key}_{condition}.csv` unless
    `output_csv` overrides. Columns follow
    `data/synthetic/codebook_synthetic.csv`.

    Args:
        config_path: Path to config.yaml.
        rct_id: Key under `rcts:` in config.yaml (e.g. "duch_et_al_2023").
        model_key: Key under `models:` in config.yaml (e.g. "llama_8b_base").
        model_id: Together AI model identifier to query (e.g. the fine-tuned
            model name returned by the fine-tuning job).
        condition: "instruct", "finetuned", or "instruction_tuned".
            For "instruction_tuned", the per-dataset training/lora overrides
            under config.yaml `instruction_tuning.datasets.{ft_corpus}` are
            layered on top of the resolved training/lora config.
        data_file_path: Override for the RCT data CSV. Defaults to the
            `data_file` entry under the RCT config.
        output_csv: Override for the output CSV path.
        ft_corpus: Label for the training corpus used to produce `model_id`
            (e.g. "alpaca"). Populates the codebook's `ft_corpus` column.
            Required when condition="instruction_tuned".
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if rct_id not in cfg["rcts"]:
        raise KeyError(
            f"RCT {rct_id!r} not in config.yaml. Known: {sorted(cfg['rcts'])}"
        )
    if model_key not in cfg["models"]:
        raise KeyError(
            f"Model {model_key!r} not in config.yaml. Known: {sorted(cfg['models'])}"
        )

    rct_cfg = cfg["rcts"][rct_id]
    model_cfg = cfg["models"][model_key]
    training = {**cfg["training"], **model_cfg.get("training", {})}
    lora = {**cfg["lora"], **model_cfg.get("lora", {})}
    inference = {**cfg.get("inference", {}), **model_cfg.get("inference", {})}

    # For instruction_tuned runs, layer the per-dataset training/lora
    # overrides from config.yaml's instruction_tuning.datasets block on top.
    if condition == "instruction_tuned":
        if not ft_corpus:
            raise ValueError(
                "condition='instruction_tuned' requires --ft-corpus to identify "
                "which instruction_tuning dataset produced the model."
            )
        datasets = cfg.get("instruction_tuning", {}).get("datasets", {})
        if ft_corpus not in datasets:
            raise KeyError(
                f"ft_corpus {ft_corpus!r} not in config.yaml "
                f"instruction_tuning.datasets. Known: {sorted(datasets)}"
            )
        dataset_cfg = datasets[ft_corpus]
        training = {**training, **dataset_cfg.get("training", {})}
        lora = {**lora, **dataset_cfg.get("lora", {})}

    with open(rct_cfg["prompt_file"]) as f:
        prompt_cfg = json.load(f)

    data_path = data_file_path or rct_cfg["data_file"]
    data, var_labels = load_data(data_path)

    id_col = "SubjectID"
    treatment_col = "individual_treatment"

    prompts = generate_synthetic_experiment_prompts(
        data,
        prompt_cfg["profile_vars"],
        prompt_cfg["system_template"],
        prompt_cfg["user_template"],
        prompt_cfg["treatment"],
        id_column=id_col,
        treatment_column=treatment_col,
        var_labels=var_labels,
    )

    experiment_round = rct_id
    version_suffix = f"_{ft_corpus}" if ft_corpus and condition != "instruct" else ""
    experiment_version = f"{model_key}_{condition}{version_suffix}"
    prompts_with_responses = inference_endpoint_query(
        prompts=prompts,
        system_message_field="system_message",
        user_message_field="question_prompt",
        experiment_round=experiment_round,
        experiment_version=experiment_version,
        model_name="together_logit",
        together_model_id=model_id,
        temperature=inference.get("temperature", 1.0),
        max_tokens=inference.get("max_tokens", 1),
        logprobs_top_k=inference.get("logprobs_top_k", 5),
    )

    data[id_col] = data[id_col].astype(str)
    prompts_with_responses[id_col] = prompts_with_responses[id_col].astype(str)
    merged = pd.merge(data, prompts_with_responses, on=id_col, suffixes=("", "_y"))

    logit_cols = merged["llm_response"].apply(_parse_logit_response)
    merged = pd.concat([merged, logit_cols], axis=1)

    outcome_col = rct_cfg["outcome"]
    family, version, size = _model_family_version_size(model_cfg, model_key)
    is_finetuned = condition in ("finetuned", "instruction_tuned")
    target_modules = ",".join(lora.get("target_modules", [])) if is_finetuned else None
    effective_bs = (
        training["batch_size"] * training.get("gradient_accumulation_steps", 1)
        if is_finetuned
        else None
    )
    seed = (training.get("seeds") or [None])[0] if is_finetuned else None

    ft_base_model = model_cfg.get("base_model") if is_finetuned else None
    ft_corpus_value = ft_corpus if is_finetuned else None

    out = pd.DataFrame(
        {
            "subject_id": merged[id_col].astype(str),
            "treatment": merged["treatment"],
            "outcome": merged[outcome_col],
            "prediction": merged["llm_response_parsed"],
            "logit_yes": merged["logprob_yes"],
            "logit_no": merged["logprob_no"],
            "prob_yes": merged.apply(_renormalised_prob_yes, axis=1),
            "date": datetime.date.today().isoformat(),
            "rct_id": rct_id,
            "model_family": family,
            "model_version": version,
            "model_size": size,
            "model_id": model_id,
            "fine_tuned": is_finetuned,
            "ft_base_model": ft_base_model,
            "ft_corpus": ft_corpus_value,
            "lora_r": lora.get("r") if is_finetuned else None,
            "lora_alpha": lora.get("alpha") if is_finetuned else None,
            "lora_dropout": lora.get("dropout") if is_finetuned else None,
            "lora_target_modules": target_modules,
            "train_epochs": training.get("epochs") if is_finetuned else None,
            "train_batch_size": training.get("batch_size") if is_finetuned else None,
            "train_grad_accum_steps": (
                training.get("gradient_accumulation_steps", 1) if is_finetuned else None
            ),
            "train_effective_batch_size": effective_bs,
            "train_lr": training.get("learning_rate") if is_finetuned else None,
            "train_lr_scheduler": (
                training.get("lr_scheduler") if is_finetuned else None
            ),
            "train_warmup_ratio": (
                training.get("warmup_ratio") if is_finetuned else None
            ),
            "train_max_grad_norm": (
                training.get("max_grad_norm") if is_finetuned else None
            ),
            "train_weight_decay": (
                training.get("weight_decay") if is_finetuned else None
            ),
            "train_optimizer": training.get("optimizer") if is_finetuned else None,
            "train_precision": training.get("precision") if is_finetuned else None,
            "train_max_seq_length": (
                training.get("max_seq_length") if is_finetuned else None
            ),
            "train_seed": seed,
            "train_n_checkpoints": (
                training.get("n_checkpoints") if is_finetuned else None
            ),
            "train_n_evals": training.get("n_evals") if is_finetuned else None,
            "train_on_inputs": (
                training.get("train_on_inputs") if is_finetuned else None
            ),
            "infer_precision": inference.get("precision"),
            "infer_batch_size": inference.get("batch_size"),
            "infer_max_seq_length": inference.get("max_seq_length"),
            "infer_temperature": inference.get("temperature"),
            "infer_max_tokens": inference.get("max_tokens"),
            "infer_logprobs_top_k": inference.get("logprobs_top_k"),
            "infer_target_tokens": ",".join(inference.get("target_tokens", [])) or None,
        }
    )

    output_path = (
        output_csv
        or f"data/synthetic/{rct_id}_{model_key}_{condition}{version_suffix}.csv"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote {len(out)} rows to {output_path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--rct-id", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument(
        "--model-id",
        required=True,
        help="Together AI model identifier (e.g. fine-tuned model name).",
    )
    parser.add_argument(
        "--condition",
        default="finetuned",
        choices=["instruct", "finetuned", "instruction_tuned"],
    )
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument(
        "--ft-corpus",
        default=None,
        help=(
            "Training corpus label. For condition=instruction_tuned, must be a key "
            "under config.yaml instruction_tuning.datasets (e.g. 'alpaca', "
            "'alpagasus') -- its training/lora overrides are applied on top of "
            "the model's config. Populates the ft_corpus output column."
        ),
    )
    args = parser.parse_args()

    run_togetherai_inference_codebook(
        config_path=args.config,
        rct_id=args.rct_id,
        model_key=args.model_key,
        model_id=args.model_id,
        condition=args.condition,
        data_file_path=args.data_file,
        output_csv=args.output_csv,
        ft_corpus=args.ft_corpus,
    )


if __name__ == "__main__":
    main()
