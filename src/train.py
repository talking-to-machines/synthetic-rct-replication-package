"""LoRA fine-tune a model on the combined survey + RCT corpus via SageMaker.

Dual-mode entry point. The same file acts as the local CLI launcher and as
the SageMaker training script:

  - Local CLI: builds the corpus, uploads to S3, submits a SageMaker
    HuggingFace training job pointed at this file.
  - In-container: TRL SFT + LoRA on $SM_CHANNEL_TRAINING, merges the adapter
    into the base, patches the tokenizer for legacy serving DLCs, and pushes
    the merged model to the Hub.

Dispatch is on the `SM_CHANNEL_TRAINING` env var, which SageMaker sets only
inside the training container.

Reads:  config.yaml (training, lora, finetuning, sagemaker, {rcts,surveys} blocks)
        {source.data_file}, {source.prompt_file}
Writes: data/processed/{kind}/{id}/{id}_train.json
        data/processed/{kind}/{id}/{id}_test.csv  (only when --train-test-split)
        data/finetuning/train.jsonl
        outputs/logs/training/{model_key}_ft_job.pkl
        Hub: {HF_USERNAME}/{model_key}{-rct_id}
"""

import argparse
import inspect
import json
import os
import pickle
from pathlib import Path


def _patch_tokenizer_to_legacy_format(model_dir: str) -> None:
    """Rewrite the saved tokenizer to a format older serving DLCs accept.

    transformers 5.x writes `tokenizer_config.json` with
    `tokenizer_class="TokenizersBackend"` and extracts the chat template into
    a separate `chat_template.jinja` file. Inference / dedicated-endpoint
    DLCs ship transformers < 5 and reject both.

    Args:
        model_dir: Directory containing the saved tokenizer files. Files are
            rewritten in place.
    """
    tok_cfg_path = Path(model_dir) / "tokenizer_config.json"
    chat_tpl_path = Path(model_dir) / "chat_template.jinja"
    if not tok_cfg_path.exists():
        print("[patch] tokenizer_config.json not found - skipping")
        return
    with open(tok_cfg_path) as f:
        tok_cfg = json.load(f)
    original_class = tok_cfg.get("tokenizer_class")
    tok_cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    if chat_tpl_path.exists():
        tok_cfg["chat_template"] = chat_tpl_path.read_text()
        chat_tpl_path.unlink()
    with open(tok_cfg_path, "w") as f:
        json.dump(tok_cfg, f, indent=2, ensure_ascii=False)
    print(
        f"[patch] tokenizer_class: {original_class!r} -> {tok_cfg['tokenizer_class']!r}; "
        f"chat_template inlined? {'chat_template' in tok_cfg}"
    )


def _train_in_container() -> None:
    """TRL SFT + LoRA training, executed inside the SageMaker DLC.

    Selects between DDP and FSDP based on the `--fsdp` flag:

      - DDP (default): each rank holds the full model. Used for 8B / 7B models.
      - FSDP (`--fsdp true`): shards the base across ranks. Used for 32B / 70B
        models that don't fit per-GPU. Requires SageMaker to launch via
        `distribution={"torch_distributed": {"enabled": True}}`.

    Post-training, both paths follow the same merge flow: rank 0 saves the
    adapter, reloads the base on CPU, applies the adapter, merges, and pushes
    the standalone merged model — the only artifact pushed to the Hub.
    """
    import gc
    import shutil

    import torch
    from datasets import load_dataset
    from huggingface_hub import create_repo, login as hf_login, upload_folder
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    _str2bool = lambda s: str(s).strip().lower() in ("true", "1", "yes")  # noqa: E731

    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj"
    )
    p.add_argument("--hub_model_id", type=str, default=None)
    p.add_argument("--gradient_checkpointing", type=_str2bool, default=False)
    p.add_argument("--fsdp", type=_str2bool, default=False)
    args = p.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = rank == 0

    training_dir = os.environ["SM_CHANNEL_TRAINING"]
    output_dir = os.environ["SM_MODEL_DIR"]
    hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN env var is missing. Did the estimator pass environment={'HF_TOKEN': ...}?"
        )
    hf_login(token=hf_token, add_to_git_credential=False)
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    if is_main:
        print(
            f"[setup] rank={rank} world_size={world_size}  base_model={args.base_model}  "
            f"HF token len={len(hf_token)} (suffix={hf_token[-4:]})"
        )

    dataset = load_dataset(
        "json",
        data_files=os.path.join(training_dir, "*.jsonl"),
        split="train",
    )
    if is_main:
        print(
            f"[setup] Loaded {len(dataset)} training rows; first row keys: {list(dataset[0].keys())}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        token=hf_token,
        torch_dtype=torch.bfloat16,
    )

    transformer_layer_cls = type(model.model.layers[0]).__name__
    if is_main and args.fsdp:
        print(f"[setup] FSDP transformer_layer_cls_to_wrap = {transformer_layer_cls}")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            m.strip() for m in args.lora_target_modules.split(",") if m.strip()
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_torch",
        "bf16": True,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "save_total_limit": 1,
        "save_strategy": "epoch",
        "logging_steps": 10,
        "report_to": "none",
        "push_to_hub": False,
    }
    # TRL renamed `max_seq_length` -> `max_length` between ~0.12 and 0.20.
    sft_params = inspect.signature(SFTConfig).parameters
    if "max_length" in sft_params:
        sft_kwargs["max_length"] = args.max_seq_length
    elif "max_seq_length" in sft_params:
        sft_kwargs["max_seq_length"] = args.max_seq_length

    # Restrict loss to the assistant turn; otherwise the model memorises the
    # prompt and reproduces it at inference. completion_only_loss works with
    # any chat template; assistant_only_loss requires {% generation %} markers
    # that OLMo-2 / Llama-3 stock templates lack.
    if "completion_only_loss" in sft_params:
        sft_kwargs["completion_only_loss"] = True
    elif "assistant_only_loss" in sft_params:
        sft_kwargs["assistant_only_loss"] = True

    if args.gradient_checkpointing:
        sft_kwargs["gradient_checkpointing"] = True
        sft_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    if args.fsdp:
        sft_kwargs["fsdp"] = "full_shard auto_wrap"
        sft_kwargs["fsdp_config"] = {
            "transformer_layer_cls_to_wrap": [transformer_layer_cls],
            "use_orig_params": True,
            "sync_module_states": True,
            "state_dict_type": "FULL_STATE_DICT",
            "limit_all_gathers": True,
            "forward_prefetch": True,
        }

    if is_main:
        seq_len_arg = "max_length" if "max_length" in sft_params else "max_seq_length"
        print(
            f"[setup] SFTConfig {seq_len_arg}={args.max_seq_length}  "
            f"per_device_bs={args.batch_size}  grad_accum={args.gradient_accumulation}  "
            f"world_size={world_size}  "
            f"effective_batch={args.batch_size * args.gradient_accumulation * world_size}"
        )
    sft_config = SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.model.config.use_cache = False
    trainer.train()

    # Save the adapter, then reload a fresh base on CPU for the merge instead
    # of merging off `trainer.model` directly — the latter produced corrupted
    # weights on OLMo-2 7B.
    adapter_dir = os.path.join(output_dir, "_adapter_tmp")
    trainer.save_model(adapter_dir)

    if not is_main:
        return

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Also push the unmerged adapter so it can be loaded standalone via PEFT
    # (e.g. through src/serving/inference.py) without going through the merge.
    if args.hub_model_id:
        adapter_hub_id = f"{args.hub_model_id}-adapter"
        print(
            f"[push-adapter] Uploading adapter to https://huggingface.co/{adapter_hub_id}"
        )
        tokenizer.save_pretrained(adapter_dir)
        create_repo(adapter_hub_id, token=hf_token, exist_ok=True, private=True)
        upload_folder(
            folder_path=adapter_dir,
            repo_id=adapter_hub_id,
            token=hf_token,
            commit_message="Upload LoRA adapter",
        )

    print("[merge] Reloading base model on CPU for LoRA merge")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    print(f"[merge] Loading adapter from {adapter_dir}")
    peft_model = PeftModel.from_pretrained(base, adapter_dir)
    print("[merge] merge_and_unload()...")
    merged = peft_model.merge_and_unload()
    shutil.rmtree(adapter_dir)

    print(f"[save] Saving merged model to {output_dir}")
    merged.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    _patch_tokenizer_to_legacy_format(output_dir)

    if args.hub_model_id:
        print(f"[push] Uploading to https://huggingface.co/{args.hub_model_id}")
        create_repo(args.hub_model_id, token=hf_token, exist_ok=True, private=True)
        upload_folder(
            folder_path=output_dir,
            repo_id=args.hub_model_id,
            token=hf_token,
            commit_message="Upload merged fine-tuned model + legacy-format tokenizer",
        )


def main() -> None:
    """Parse CLI arguments, build the corpus, and submit a SageMaker job.

    Builds the combined fine-tuning corpus (with optional per-source
    train/test split stratified by treatment), uploads it to S3, launches the
    SageMaker training job, pickles the job metadata, and polls until the
    job reaches a terminal state.

    Raises:
        RuntimeError: If the corpus is empty, or if the fine-tuning job
            fails or was stopped.
    """
    from src.build_corpus import build_corpus
    from src.models.finetuning import (
        launch_finetune,
        poll_finetune_until_done,
        resolve_params,
    )
    from src.utils.io import load_yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--model-key", type=str, default="llama_8b")
    parser.add_argument(
        "--rct-id",
        type=str,
        default=None,
        help="Optional RCT id appended to the pushed Hub repo name.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/finetuning/train.jsonl"),
    )
    parser.add_argument(
        "--train-test-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Hold out a per-source test set using `finetuning.test_fraction`. "
            "Pass --no-train-test-split to use the full data for training."
        ),
    )
    parser.add_argument("--job-pkl", type=Path, default=None)
    parser.add_argument("--poll-interval", type=int, default=60)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    params = resolve_params(cfg, args.model_key)
    job_pkl = args.job_pkl or Path(f"outputs/logs/training/{args.model_key}_ft_job.pkl")

    print(f"Model: {args.model_key} -> {params['base_model']}")
    print(f"Training: {params['training']}")
    print(f"LoRA: {params['lora']}")
    print(f"Instance: {params['training_instance_type']}")

    n = build_corpus(cfg, args.output_jsonl, train_test_split=args.train_test_split)
    if n == 0:
        raise RuntimeError(
            f"No training examples produced. Check finetuning sources in {args.config}."
        )
    print(f"Wrote {n} examples to {args.output_jsonl}")

    job_name, _ = launch_finetune(
        params=params,
        training_jsonl=args.output_jsonl,
        model_key=args.model_key,
        rct_id=args.rct_id,
    )

    job_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(job_pkl, "wb") as f:
        pickle.dump(
            {
                "job_name": job_name,
                "model_key": args.model_key,
                "rct_id": args.rct_id,
                "base_model": params["base_model"],
                "params": params,
            },
            f,
        )

    result = poll_finetune_until_done(job_name, args.poll_interval)
    if result is None:
        raise RuntimeError(f"Fine-tuning job {job_name} failed or was stopped.")
    print(f"S3 model artifact: {result['s3_model_artifact']}")


if __name__ == "__main__":
    if "SM_CHANNEL_TRAINING" in os.environ:
        _train_in_container()
    else:
        main()
