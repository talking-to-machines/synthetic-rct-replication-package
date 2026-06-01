"""Custom SageMaker inference handler: base model + LoRA adapter via PEFT.

Load the base in bf16, apply the adapter via `PeftModel.from_pretrained`
(no merge), generate, and return logprobs in the chat-completions shape TGI
emits so existing callers and parsers work unchanged. Used in place of TGI
for architectures whose attention layout TGI's LoRA loader can't patch
(e.g. OLMo-2).

Env vars (set by `deploy_sagemaker_endpoint`):
  BASE_MODEL_ID   Hub repo id of the base instruct model (required).
  ADAPTER_ID      Hub repo id of the LoRA adapter (optional; when unset
                  the handler serves the bare base).
  HF_TOKEN        Hub token; needed for private repos.
"""

import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def model_fn(model_dir: str) -> dict:
    """Load base + adapter once at endpoint start; cached for every request.

    `model_dir` is provided by SageMaker's MMS but ignored here — the base
    and adapter are pulled from the Hub via the `BASE_MODEL_ID` and
    `ADAPTER_ID` env vars.

    Args:
        model_dir: Path SageMaker extracts `model.tar.gz` into. Unused.

    Returns:
        A dict with keys `model` (the PEFT-wrapped base, or bare base when
        no adapter id is set) and `tokenizer`, passed as `ctx` to `predict_fn`.
    """
    base_id = os.environ["BASE_MODEL_ID"]
    adapter_id = os.environ.get("ADAPTER_ID") or None
    token = os.environ.get("HF_TOKEN") or None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[model_fn] device={device}  base={base_id}  adapter={adapter_id!r}")

    tokenizer = AutoTokenizer.from_pretrained(base_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        token=token,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None,
    )

    if adapter_id:
        print(f"[model_fn] Applying adapter {adapter_id}")
        model = PeftModel.from_pretrained(base, adapter_id, token=token)
    else:
        model = base

    model.eval()
    return {"model": model, "tokenizer": tokenizer}


def input_fn(input_data, content_type: str = "application/json"):
    """Parse the chat-completions request body.

    Args:
        input_data: Raw request body, either `bytes`/`bytearray` or `str`.
        content_type: MIME type; only `application/json` is supported.

    Returns:
        The decoded JSON dict.

    Raises:
        ValueError: If `content_type` is anything other than JSON.
    """
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type!r}")
    if isinstance(input_data, (bytes, bytearray)):
        input_data = input_data.decode("utf-8")
    return json.loads(input_data)


def predict_fn(data: dict, ctx: dict) -> dict:
    """Run one chat-completion call and return the response in TGI's shape.

    Reads `messages`, `max_tokens`, `temperature`, `top_logprobs`, and
    `logprobs` from `data` (chat-completions request body), generates, and
    returns a `{"choices": [...]}` dict matching the TGI shape consumed by
    `_parse_invoke_logprobs` in `src/models/api_client.py`.

    Args:
        data: Parsed request body from `input_fn`.
        ctx: The dict returned by `model_fn` (`{"model", "tokenizer"}`).

    Returns:
        A chat-completion response with one `choices` entry. When the
        request asked for logprobs, each generated position carries a
        `top_logprobs` list of `{token, logprob}` entries.
    """
    model = ctx["model"]
    tokenizer = ctx["tokenizer"]

    messages = data["messages"]
    max_tokens = int(data.get("max_tokens", 1))
    temperature = float(data.get("temperature", 1.0))
    top_logprobs_k = int(data.get("top_logprobs", 5))
    return_logprobs = bool(data.get("logprobs", False))

    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    do_sample = temperature > 0.0
    with torch.no_grad():
        gen = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            output_scores=return_logprobs,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_ids = gen.sequences[0, input_ids.shape[1] :]
    output_text = tokenizer.decode(new_ids, skip_special_tokens=True)

    choice = {
        "index": 0,
        "message": {"role": "assistant", "content": output_text},
        "finish_reason": "length",
    }

    if return_logprobs and gen.scores:
        per_position = []
        for i, step_logits in enumerate(gen.scores):
            log_softmax = torch.log_softmax(step_logits[0].float(), dim=-1)
            sampled_id = int(new_ids[i].item())
            sampled_token = tokenizer.decode([sampled_id])
            sampled_lp = float(log_softmax[sampled_id].item())
            top_lp, top_idx = log_softmax.topk(top_logprobs_k)
            top_entries = [
                {
                    "token": tokenizer.decode([int(idx.item())]),
                    "logprob": float(lp.item()),
                }
                for lp, idx in zip(top_lp, top_idx)
            ]
            per_position.append(
                {
                    "token": sampled_token,
                    "logprob": sampled_lp,
                    "top_logprobs": top_entries,
                }
            )
        choice["logprobs"] = {"content": per_position}

    return {"choices": [choice]}


def output_fn(prediction: dict, accept: str = "application/json") -> str:
    """Serialize the `predict_fn` response.

    Args:
        prediction: The dict returned by `predict_fn`.
        accept: Response MIME type; only `application/json` is supported.

    Returns:
        The JSON-serialized prediction as a string.

    Raises:
        ValueError: If `accept` is anything other than JSON.
    """
    if accept and accept != "application/json":
        raise ValueError(f"Unsupported Accept type: {accept!r}")
    return json.dumps(prediction)
