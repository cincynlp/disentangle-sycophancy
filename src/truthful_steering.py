from typing import Optional, Dict
import torch
import argparse
import json
import os
import torch
from transformers import AutoTokenizer
from transformers import LogitsProcessor, LogitsProcessorList

from .utils import text_processing, model_utils, dataset


# ----------- LogitsProcessor to restrict allowed tokens (e.g., letters) -----------
class OnlyAllowLetters(LogitsProcessor):
    def __init__(self, allowed_token_ids: list):
        super().__init__()
        self.allowed = list(
            sorted(set(int(x) for x in allowed_token_ids if x is not None))
        )

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # scores: (batch, vocab)
        if not self.allowed:
            return scores
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.allowed] = 0.0
        return scores + mask


# ----------- Additional imports for steering/generation -----------
import re
from transformers import AutoModelForCausalLM
from accelerate import Accelerator
from scipy.special import expit
import numpy as np
import logging

_LOG = logging.getLogger(__name__)


# ---- JSONL append helper ----
def _append_jsonl(path: str, obj: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")





# -------------------- TruthfulQA-style (multiple choice) helpers --------------------

# --- Helper: MC block line pattern, e.g. " (A) Answer text"
_MC_BLOCK_RE = re.compile(r"^\s*\(([A-Z])\)\s+", flags=re.IGNORECASE)


def _infer_letters_from_prompt(prompt_text: str) -> Optional[str]:
    """Try to infer a contiguous letter set (e.g., 'ABCD' or 'ABCDE') from an Options: block.
    Scans lines beginning with '(A) ', '(B) ', ... and returns the consecutive range found.
    """
    if not isinstance(prompt_text, str):
        return None
    lines = [ln.strip() for ln in prompt_text.splitlines()]
    letters = []
    for ln in lines:
        m = _MC_BLOCK_RE.match(ln)
        if m:
            letters.append(m.group(1).upper())
    if not letters:
        return None
    # Dedup preserving order
    seen = set()
    ordered = []
    for L in letters:
        if L not in seen:
            seen.add(L)
            ordered.append(L)
    # Check if ordered is a contiguous prefix from 'A'
    if not ordered:
        return None
    start = "A"
    expected = "".join(chr(ord("A") + i) for i in range(len(ordered)))
    if "".join(ordered[: len(expected)]) == expected:
        return expected
    # Fallback: if at least A..D seen, return that span
    if set(["A", "B", "C", "D"]).issubset(set(ordered)):
        return "ABCD"
    return None


# ----------- FEATURE EXTRACTION AND MASKING UTILS -----------
def build_special_mask(tokenizer, device):
    max_id = max(tokenizer.all_special_ids)
    mask_size = max(tokenizer.vocab_size, max_id + 1)
    special_mask = torch.zeros(mask_size, device=device, dtype=torch.bool)
    for sid in tokenizer.all_special_ids:
        special_mask[sid] = True
    return special_mask





def _maybe_append_assistant(prompt_text: str, do_append: bool) -> str:
    if not do_append:
        return prompt_text
    if "Assistant:" in prompt_text:
        return prompt_text
    return prompt_text + ("\nAssistant: ")


@torch.no_grad()
def generate_with_optional_steer(prompt_text, tokenizer, model, device, steer_cfg):
    """
    steer_cfg: {
        "enabled": bool,
        "alpha": float,
        "k": int,
        "layer": int,
        "w": np.ndarray,
        "max_new_tokens": int,
        "temperature": float,
        "top_p": float,
        "append_assistant_prefix": bool,
        "stop_strings": Optional[List[str]] (unused here),
        "capture_last_logits": bool,  # if True, also return logits for the first generated step
        "nullspace": {"enabled": bool, "W": torch.Tensor(D,k)},
        "return_extra": bool,  # (optional) if True, return extra dict with prompt/letters/ids
    }
    Returns (gen_text, gen_ids) or (gen_text, gen_ids, step0_logits) if capture_last_logits,
    or with extra dict if steer_cfg['return_extra'] is True.
    """
    prompt_text = _maybe_append_assistant(
        prompt_text, steer_cfg.get("append_assistant_prefix", False)
    )

    # Prefer explicit config, else try to infer from the prompt text
    letters_str = (
        steer_cfg.get("allowed_letters")
        or steer_cfg.get("mc_letters")
        or steer_cfg.get("letters_fallback")
        or _infer_letters_from_prompt(prompt_text)
    )

    # Ensure there is a non-empty Answer Choices line in the prompt
    try:
        # Accept multiple ways of passing full answer text
        answers_text_cfg = (
            steer_cfg.get("answers_text")
            or steer_cfg.get("base_answers")
            or (
                steer_cfg.get("base", {}).get("answers")
                if isinstance(steer_cfg.get("base", {}), dict)
                else None
            )
        )
        letters_fallback = steer_cfg.get("letters_fallback")  # e.g., 'ABCD'
        # Resolve final text to append for choices
        final_choices = (
            answers_text_cfg
            or letters_str
            or letters_fallback
            or steer_cfg.get("allowed_letters")
            or steer_cfg.get("mc_letters")
            or "ABCD"
        )
        if isinstance(final_choices, (list, tuple)):
            final_choices = " ".join(map(str, final_choices))
        final_choices = str(final_choices).strip()

        if "Answer Choices:" in prompt_text:
            lines = prompt_text.splitlines()
            for i, ln in enumerate(lines):
                if ln.strip().startswith("Answer Choices:"):
                    tail = ln.split("Answer Choices:", 1)[1]
                    # Look ahead to see if there is already a block of (A) ... answers
                    has_block_after = False
                    j = i + 1
                    while j < len(lines):
                        nxt = lines[j].strip()
                        # stop scanning at blank line or the start of the Answer section
                        if nxt == "" or re.match(
                            r"^Answer:\s*$", nxt, flags=re.IGNORECASE
                        ):
                            break
                        if _MC_BLOCK_RE.match(nxt):
                            has_block_after = True
                            break
                        j += 1
                    if has_block_after:
                        # Already filled; leave as-is
                        pass
                    else:
                        if tail.strip() == "":
                            # No inline tail; insert final_choices
                            if "\n" in final_choices:
                                # Insert the block on the next lines
                                insert_lines = final_choices.splitlines()
                                # Ensure there's exactly one blank line between header and block when needed
                                lines = lines[: i + 1] + insert_lines + lines[i + 1 :]
                            else:
                                # Single-line choices (e.g., 'ABCD')
                                sep = " " if not ln.endswith(" ") else ""
                                lines[i] = ln + sep + final_choices
                    break
            prompt_text = "\n".join(lines)
        else:
            # No Answer Choices line present: append one just before the final 'Answer:' if present, else at end
            if "\nAnswer:" in prompt_text:
                parts = prompt_text.rsplit("\nAnswer:", 1)
                prompt_text = (
                    parts[0].rstrip()
                    + f"\nAnswer Choices: {final_choices}\n\nAnswer:"
                    + parts[1]
                )
            else:
                prompt_text = (
                    prompt_text.rstrip()
                    + f"\nAnswer Choices: {final_choices}\n\nAnswer:"
                )
    except Exception:
        pass

    enc = tokenizer(prompt_text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    attn_mask = (
        enc.attention_mask.to(device) if hasattr(enc, "attention_mask") else None
    )

    print(prompt_text)

    # Steering window: by default, steer the last k tokens of *the current step*.
    # Do NOT anchor to prompt length; with KV cache, S==1 at decode steps.
    input_len = enc.input_ids.size(1)
    forced_start = steer_cfg.get("forced_window_start", None)

    hook_handle = None
    if steer_cfg.get("enabled", False):
        layers = model_utils._get_decoder_layers(model)
        L = int(steer_cfg["layer"])
        if L < 0 or L >= len(layers):
            # Attempt to reinterpret as hidden index
            try:
                L_mapped = model_utils._hidden_to_block_index(L, model)
            except Exception:
                L_mapped = max(0, min(len(layers) - 1, L))
            L = L_mapped
        target = layers[L]
        alpha = float(steer_cfg.get("alpha", 0.0))
        k = int(steer_cfg.get("k", 16))
        w = torch.tensor(
            steer_cfg["w"], dtype=model.dtype, device=next(target.parameters()).device
        )
        nullspace_cfg = steer_cfg.get("nullspace", None)

        def fwd_hook(module, inp, out):
            def _apply_add(hidden):
                B, S, D = hidden.shape
                start = max(0, S - k)
                if forced_start is not None:
                    start = max(int(forced_start), start)
                if start < S:
                    hidden = hidden.clone()
                    # ---- Nullspace ablation (project-out) ----
                    ns_cfg = steer_cfg.get("nullspace", None)
                    if ns_cfg and ns_cfg.get("enabled", False):
                        W = ns_cfg.get("W", None)  # (D, k)
                        if W is not None and W.numel() > 0:
                            seg = hidden[:, start:S, :]  # (B,T,D)
                            # H <- H - (H W) W^T
                            coeff = torch.matmul(seg, W)  # (B,T,k)
                            proj = torch.matmul(coeff, W.transpose(0, 1))  # (B,T,D)
                            seg = seg - proj
                            hidden[:, start:S, :] = seg
                    # ---- Additive steering (if any) ----
                    if alpha != 0.0:
                        hidden[:, start:S, :] += alpha * w
                return hidden

            # Dataclass-like ModelOutput
            if hasattr(out, "last_hidden_state"):
                hidden = out.last_hidden_state
                hidden = _apply_add(hidden)
                # Reconstruct a new ModelOutput preserving other fields
                try:
                    data = dict(out)
                except Exception:
                    # fallback for older versions
                    data = out.__dict__.copy()
                data["last_hidden_state"] = hidden
                return out.__class__(**data)

            # Tuple API
            if isinstance(out, tuple) and len(out) > 0:
                hidden, *rest = out
                hidden = _apply_add(hidden)
                return (hidden, *rest)

            # Tensor API
            if torch.is_tensor(out):
                hidden = _apply_add(out)
                return hidden

            # Unknown type: don’t alter
            return out

        hook_handle = target.register_forward_hook(fwd_hook)

    want_logits = bool(steer_cfg.get("capture_last_logits", False))
    # Determine sampling mode explicitly: prefer steer_cfg["do_sample"], else infer from temperature
    inferred_do_sample = bool(steer_cfg.get("temperature", 0.0) > 0.0)
    do_sample_flag = bool(steer_cfg.get("do_sample", inferred_do_sample))

    gen_kwargs = dict(
        max_new_tokens=steer_cfg.get("max_new_tokens", 16),
        do_sample=do_sample_flag,
    )
    if do_sample_flag:
        if "temperature" in steer_cfg:
            gen_kwargs["temperature"] = float(steer_cfg.get("temperature"))
        if "top_p" in steer_cfg:
            gen_kwargs["top_p"] = float(steer_cfg.get("top_p"))
        if "top_k" in steer_cfg:
            gen_kwargs["top_k"] = int(steer_cfg.get("top_k"))
    else:
        # Ensure no sampling-only flags leak into generate() when do_sample=False
        for _flag in ("temperature", "top_p", "top_k"):
            if _flag in gen_kwargs:
                gen_kwargs.pop(_flag, None)
    if tokenizer.eos_token_id is not None:
        gen_kwargs["eos_token_id"] = tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    if want_logits:
        gen_kwargs["output_scores"] = True
        gen_kwargs["return_dict_in_generate"] = True

    # ----- Optional/Auto: restrict next-token vocabulary to a set of letters (e.g., 'ABCD') -----
    processors = None
    # `letters_str` was resolved before tokenization (may be None)
    allowed_ids = []
    if letters_str:
        # Convert each letter into one or more plausible first-token IDs.
        for ch in list(str(letters_str)):
            cands = []
            # 1) direct token (rare but cheap)
            try:
                tid = tokenizer.convert_tokens_to_ids(ch)
                if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
                    cands.append(tid)
            except Exception:
                pass
            # 2) encode raw char
            try:
                toks = tokenizer.encode(ch, add_special_tokens=False)
                if toks:
                    cands.append(int(toks[0]))
            except Exception:
                pass
            # 3) encode with preceding space (common BPE/SentencePiece form)
            try:
                toks = tokenizer.encode(" " + ch, add_special_tokens=False)
                if toks:
                    cands.append(int(toks[0]))
            except Exception:
                pass
            # Dedup
            for t in cands:
                if isinstance(t, int):
                    allowed_ids.append(t)
        # Install logits processor if we found any usable IDs
        if allowed_ids:
            processors = LogitsProcessorList([OnlyAllowLetters(allowed_ids)])
            gen_kwargs["logits_processor"] = processors
        else:
            if steer_cfg.get("verbose", False):
                warn = f"[warn] allowed_letters inferred as {letters_str!r} but produced no token IDs; masking disabled."
                _LOG.warning(warn)
                print(warn)
    else:
        if steer_cfg.get("verbose", False):
            info = "[info] No allowed_letters provided/inferred; logits not masked."
            _LOG.debug(info)
            print(info)

    # Force exactly one new token if requested or if we set up letter masking
    if steer_cfg.get("force_single_token", False) or processors is not None:
        gen_kwargs["max_new_tokens"] = 1
        gen_kwargs["do_sample"] = False

    out = model.generate(input_ids=input_ids, attention_mask=attn_mask, **gen_kwargs)
    if want_logits:
        # `out` is a ModelOutput with .sequences and .scores (list length = generated steps)
        out_ids = out.sequences
        scores_list = out.scores if isinstance(out.scores, (list, tuple)) else []
    else:
        out_ids = out
        scores_list = []
    gen_ids = out_ids[0, input_ids.size(1) :]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    print("Response:", gen_text)

    # ---- Collect extra metadata for optional return ----
    extra = {
        "prompt_text": prompt_text,
        "letters_str": letters_str if "letters_str" in locals() else None,
        "allowed_ids": (
            sorted(set(allowed_ids))
            if "allowed_ids" in locals() and allowed_ids
            else None
        ),
    }

    # ----- Verbose logging of prompt and generated answer -----
    if steer_cfg.get("verbose", False):
        try:
            sample_idx = steer_cfg.get("sample_idx", None)
            hdr = f"[sample {sample_idx}]" if sample_idx is not None else "[gen]"
            msg_lines = [f"{hdr} Prompt:", str(prompt_text)]
            if letters_str:
                msg_lines.append(f"{hdr} Letters: {letters_str}")
                msg_lines.append(
                    f"{hdr} Allowed token IDs: {sorted(set(allowed_ids)) if allowed_ids else None}"
                )
            msg_lines.append(f"{hdr} Generated: {repr(gen_text)}")
            msg = "\n".join(msg_lines)
            _LOG.debug(msg)
            print(msg)
        except Exception:
            pass

    if hook_handle is not None:
        hook_handle.remove()

    # ---- Adjusted returns to optionally include extra dict ----
    if want_logits and len(scores_list) > 0:
        step0_logits = scores_list[0][0].detach().to("cpu")  # (vocab,)
        if steer_cfg.get("return_extra", False):
            return gen_text, gen_ids, step0_logits, extra
        return gen_text, gen_ids, step0_logits
    if want_logits:
        if steer_cfg.get("return_extra", False):
            return gen_text, gen_ids, None, extra
        return gen_text, gen_ids, None
    if steer_cfg.get("return_extra", False):
        return gen_text, gen_ids, extra
    return gen_text, gen_ids








    s = s.replace("−", "-").replace("—", "-").replace("–", "-")
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")

    return alphas


def _load_json_or_jsonl(path: str):
    """
    Load a file that may be JSON (single object/array) or JSONL (one JSON per line).
    Returns a Python object. For JSONL, returns a list of per-line objects.
    """
    with open(path, "r") as f:
        data = f.read().strip()
    if not data:
        return []
    # Try standard JSON first
    try:
        return json.loads(data)
    except Exception:
        pass
    # Fallback: JSONL
    rows = []
    for ln in data.splitlines():
        s = ln.strip()
        if not s:
            continue
        try:
            rows.append(json.loads(s))
        except Exception:
            # skip malformed lines silently
            continue
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Compute DiffMean vectors directly from a Transformer model."
    )
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument(
        "--eval_json",
        type=str,
        default=None,
        help="Optional JSON file for held-out evaluation. If provided, skip internal split.",
    )
    parser.add_argument(
        "--label_fields",
        type=str,
        nargs="+",
        default=["sycophantic_label"],
        help="Label(s) to predict.",
    )
    parser.add_argument(
        "--label_span",
        type=str,
        choices=["response", "response_eq", "prompt", "full", "statement_eos"],
        default="response",
        help="Which tokens contribute to DiffMean stats and scoring: response-only, response tokens after the substring '= ' in the response, prompt-only, full sequence (content tokens only), or statement_eos (group-b punctuation token). statement_eos = single punctuation token at end of claim (group-b)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace transformer model name (e.g., bert-base-uncased).",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Layer index to extract hidden states from (0 = embeddings, -1 = last layer).",
    )
    parser.add_argument(
        "--all_layers",
        action="store_true",
        help="If set, compute DiffMean for all transformer layers (1..num_hidden_layers). Overrides --layer.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="diffmean_results",
        help="Directory to save the DiffMean vectors and test AUROC.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for model inference (e.g., 'cuda:0' or 'cpu').",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging and progress bars."
    )
    # Removed arguments not needed for steering
    # ----------- Accelerate and sharding arguments -----------
    parser.add_argument(
        "--shard_model",
        action="store_true",
        help="Shard the base model across all visible GPUs via device_map='auto' (good for very large models).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Torch dtype for model weights. 'auto' picks bfloat16 on CUDA else float32.",
    )

    # --- Steering / generation args ---
    parser.add_argument(
        "--steer_eval",
        action="store_true",
        help="Run generation with optional steering and measure metrics.",
    )
    parser.add_argument(
        "--steer_layer",
        type=int,
        default=None,
        help="Layer index at which to add the steering vector (default: best IID layer).",
    )
    parser.add_argument(
        "--steer_alpha_grid",
        type=str,
        default="-2.0,-1.0,-0.5,-0.25,0.25,0.5,1.0,2.0",
        help="Comma-separated list of alphas to sweep; 0 is added automatically.",
    )
    parser.add_argument(
        "--steer_k_tokens",
        type=int,
        default=16,
        help="Number of most-recent tokens to steer at each step.",
    )
    parser.add_argument(
        "--steer_flip_sign",
        action="store_true",
        help="Flip the sign of the steering vector (w -> -w).",
    )
    parser.add_argument(
        "--gen_max_new_tokens",
        type=int,
        default=16,
        help="Max new tokens to generate for steering eval.",
    )
    parser.add_argument(
        "--gen_temperature",
        type=float,
        default=0.0,
        help="Generation temperature (0.0 = greedy).",
    )
    parser.add_argument(
        "--gen_top_p", type=float, default=1.0, help="Top-p sampling cutoff."
    )
    parser.add_argument(
        "--append_assistant_prefix",
        action="store_true",
        help="Append 'Assistant: ' before generation if not present.",
    )
    parser.add_argument(
        "--w_raw_file",
        type=str,
        default=None,
        help="Path to a .npy file containing a raw DiffMean steering vector (mu_pos - mu_neg). If provided, training is skipped.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of evaluation examples to use in steering eval.",
    )
    parser.add_argument(
        "--ci_bootstraps",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for confidence intervals (per alpha).",
    )
    parser.add_argument(
        "--ci_level",
        type=float,
        default=0.95,
        help="Confidence level for intervals (e.g., 0.95 for 95% CIs).",
    )
    parser.add_argument(
        "--prefill_until_equals",
        action="store_true",
        help="If set, prefill the Assistant response up to '= ' before generation. Default is OFF (regenerate full response).",
    )
    parser.add_argument(
        "--praise_detector_model",
        type=str,
        default=None,
        help="HuggingFace model name or local path for a RoBERTa (or compatible) classifier to detect praise in generated responses. If unset, praise detection is disabled.",
    )
    parser.add_argument(
        "--praise_threshold",
        type=float,
        default=0.5,
        help="Probability threshold for classifying a response as praise (only used when --praise_detector_model is set).",
    )
    parser.add_argument(
        "--null_ablate",
        action="store_true",
        help="Enable nullspace ablation (project hidden states onto the orthogonal complement of a behavior subspace).",
    )
    parser.add_argument(
        "--null_labels",
        type=str,
        nargs="*",
        default=[],
        help="Behavior labels whose DiffMean vectors define the ablated subspace (e.g., sycophantic_label genuine_label). If provided, vectors are loaded from output_dir.",
    )
    parser.add_argument(
        "--null_files",
        type=str,
        default=None,
        help="Comma-separated list or template of .npy files for basis vectors; supports {layer} and {label} placeholders.",
    )
    parser.add_argument(
        "--null_layer",
        type=int,
        default=None,
        help="Hidden-state layer index for ablation (defaults to --steer_layer if set, else the evaluation layer).",
    )
    parser.add_argument(
        "--null_k_tokens",
        type=int,
        default=16,
        help="Number of most-recent tokens to ablate per decode step (defaults to 16).",
    )
    parser.add_argument(
        "--save_outputs_json",
        type=str,
        default=None,
        help="Optional path to save generated outputs, classifications, and praise detector results as a JSON file.",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=log_level)
    logger = logging.getLogger(__name__)
    logger.info("label_span set to: %s", args.label_span)
    if args.verbose:
        _LOG.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize accelerator
    accelerator = Accelerator()
    if accelerator.is_main_process:
        logger.info(
            "Accelerate initialized: mixed_precision=%s, num_processes=%s",
            getattr(accelerator.state, "mixed_precision", None),
            accelerator.num_processes,
        )

    def _select_dtype(name):
        if name == "bfloat16":
            return torch.bfloat16
        if name == "float16":
            return torch.float16
        if name == "float32":
            return torch.float32
        # auto
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32

    import random, os

    DEFAULT_SEED = 42
    os.environ["PYTHONHASHSEED"] = str(DEFAULT_SEED)
    random.seed(DEFAULT_SEED)
    np.random.seed(DEFAULT_SEED)
    torch.manual_seed(DEFAULT_SEED)
    torch.cuda.manual_seed_all(DEFAULT_SEED)

    # Load training data
    train_data = _load_json_or_jsonl(args.input_json)

    # Support datasets that lack 'response' (e.g., multiple-choice eval-only corpora)
    def _norm_prompt(obj):
        """Accept either a raw prompt (str/list) or a full item dict. If a full item dict
        contains base.answers (multiline "(A) ..."), ensure the returned prompt has a filled
        "Answer Choices:" section with those answers, followed by a trailing blank line and
        an "Answer:" tag.
        """
        # Extract raw prompt text
        if isinstance(obj, dict) and "prompt" in obj:
            prompt_field = obj.get("prompt", "")
        else:
            prompt_field = obj

        if isinstance(prompt_field, list):
            prompt_text = " ".join(
                (m.get("content", "") for m in prompt_field if isinstance(m, dict))
            ).strip()
        else:
            prompt_text = (
                prompt_field if isinstance(prompt_field, str) else str(prompt_field)
            )
        prompt_text = prompt_text.rstrip()

        # If we were given the full item, try to attach full answer block
        answers_block = None
        if isinstance(obj, dict):
            base = obj.get("base", {}) if isinstance(obj.get("base", {}), dict) else {}
            answers_block = base.get("answers", None)
            # Some datasets store answers list; if so, render them with letter labels
            if (
                not answers_block
                and isinstance(base.get("answers_list"), list)
                and isinstance(base.get("letters"), str)
            ):
                letters = list(base.get("letters"))
                lines = []
                for i, ans in enumerate(base["answers_list"]):
                    if i < len(letters):
                        lines.append(f"({letters[i]}) {ans}")
                if lines:
                    answers_block = "\n".join(lines)

        # Only inject if we have a block and the current prompt lacks non-empty choices
        def _has_nonempty_choices(txt: str) -> bool:
            for ln in txt.splitlines():
                if ln.strip().lower().startswith("answer choices:"):
                    tail = ln.split(":", 1)[1].strip()
                    return bool(tail)
            return False

        if answers_block and not _has_nonempty_choices(prompt_text):
            # Normalize so we only have one Answer Choices block
            if not prompt_text.endswith("\n"):
                prompt_text += "\n"
            if not re.search(r"^Answer Choices:", prompt_text, flags=re.MULTILINE):
                prompt_text += "Answer Choices:\n"
            prompt_text += f"{answers_block}\n\n"
            # Ensure a single trailing "Answer:" line
            if not re.search(r"^Answer:\s*$", prompt_text, flags=re.MULTILINE):
                prompt_text += "Answer:"

        return prompt_text

    prompts_train = [_norm_prompt(item) for item in train_data]
    responses_train = [item.get("response", "") or "" for item in train_data]
    labels_train = {
        field: np.array(
            [int(item.get(field, 0) or 0) for item in train_data], dtype=int
        )
        for field in args.label_fields
    }

    if args.eval_json:
        # Load held-out eval set
        eval_data = _load_json_or_jsonl(args.eval_json)
        prompts_eval = [_norm_prompt(item) for item in eval_data]
        if args.verbose and eval_data:
            try:
                print(
                    "[debug] Example prompt with choices:\n" + prompts_eval[0][:600],
                    flush=True,
                )
            except Exception:
                pass
        responses_eval = [item.get("response", "") or "" for item in eval_data]
        labels_eval = {
            field: np.array(
                [int(item.get(field, 0) or 0) for item in eval_data], dtype=int
            )
            for field in args.label_fields
        }
    else:
        # Internal multi-label stratified split
        from sklearn.model_selection import train_test_split

        # stratify on tuple of all label fields
        stratify_vals = list(zip(*(labels_train[f] for f in args.label_fields)))
        indices = np.arange(len(train_data))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=DEFAULT_SEED, stratify=stratify_vals
        )
        # ----------- Class-balancing logic for test set -----------
        # Only balance the test set (do not move dropped to train), only for internal split
        y_primary = labels_train[args.label_fields[0]]
        pos = [i for i in test_idx if y_primary[i] == 1]
        neg = [i for i in test_idx if y_primary[i] == 0]
        n_pos_orig = len(pos)
        n_neg_orig = len(neg)
        n = min(n_pos_orig, n_neg_orig)
        if n >= 1:
            rng = np.random.RandomState(DEFAULT_SEED)
            pos_sub = rng.choice(pos, size=n, replace=False)
            neg_sub = rng.choice(neg, size=n, replace=False)
            balanced_test_idx = np.concatenate([pos_sub, neg_sub])
            rng.shuffle(balanced_test_idx)
            # For logging, get new counts
            n_pos_new = n_neg_new = n
            logger.info(
                f"Balanced test set: original pos={n_pos_orig}, neg={n_neg_orig}; after balancing pos={n_pos_new}, neg={n_neg_new}"
            )
            test_idx = balanced_test_idx
        else:
            logger.info(
                f"Test set not balanced: insufficient samples (pos={n_pos_orig}, neg={n_neg_orig})"
            )
        prompts_eval = [prompts_train[i] for i in test_idx]
        responses_eval = [responses_train[i] for i in test_idx]
        labels_eval = {f: labels_train[f][test_idx] for f in args.label_fields}
        # Build eval_data objects so downstream steer_eval can access full items
        try:
            eval_data = [train_data[i] for i in test_idx]
        except Exception:
            eval_data = []
        # reduce training set
        prompts_train = [prompts_train[i] for i in train_idx]
        responses_train = [responses_train[i] for i in train_idx]
        labels_train = {f: labels_train[f][train_idx] for f in args.label_fields}

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize model and tokenizer (Accelerate/sharding aware)
    # Device remains as a hint for input tensors when not sharding
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    pad_token_added = False
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            pad_token_added = True

    torch_dtype = _select_dtype(args.dtype)
    if args.shard_model:
        # Let HF shard the model across all visible GPUs. Keep single process.
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            output_hidden_states=True,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        if pad_token_added:
            model.resize_token_embeddings(len(tokenizer))
        model.eval()
        # For sharded models, we'll still move input tensors to cuda:0 (HF handles dispatch)
        runtime_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        # Single-device execution (optionally with Accelerate to select correct device)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            output_hidden_states=True,
            torch_dtype=torch_dtype,
        )
        if pad_token_added:
            model.resize_token_embeddings(len(tokenizer))
        model.eval()
        runtime_device = accelerator.device
        model.to(runtime_device)

    # Determine which layer(s) to process
    num_layers = model.config.num_hidden_layers  # transformer blocks
    # hidden_states has length num_layers+1 (embeddings at index 0, then 1..num_layers)
    if args.all_layers:
        layers_to_do = list(range(1, num_layers + 1))  # exclude embeddings by default
    else:
        # map possibly-negative layer index to [0, num_layers]
        eff = args.layer if args.layer >= 0 else (num_layers + 1 + args.layer)
        if eff < 0 or eff > num_layers:
            raise ValueError(
                f"Layer index {args.layer} resolves to {eff}, valid range is 0..{num_layers}"
            )
        layers_to_do = [eff]
    logger.info("Will compute DiffMean for layer indices: %s", layers_to_do)

    # --- Resolve steering vector path and load w (if provided) ---
    steer_w = None
    steer_layer_idx = (
        args.steer_layer
        if args.steer_layer is not None
        else (layers_to_do[0] if layers_to_do else None)
    )
    if args.w_raw_file:
        try:
            w_path = args.w_raw_file
            if "{layer}" in w_path and steer_layer_idx is not None:
                w_path = w_path.format(layer=steer_layer_idx)
            if os.path.exists(w_path):
                steer_w = np.load(w_path)
                logger.info(
                    f"[weights] Loaded steering vector from {w_path} with shape {getattr(steer_w, 'shape', None)}"
                )
            else:
                logger.warning(f"[weights] Steering vector path not found: {w_path}")
        except Exception as e:
            logger.warning(f"[weights] Failed to load steering vector: {e}")

    primary_label = args.label_fields[0] if args.label_fields else "sycophantic_label"
    # Only pre-compute a single default path when NOT using --all_layers
    default_w_path = None
    if (not args.all_layers) and len(layers_to_do) > 0:
        default_w_path = model_utils._default_w_path(
            args.output_dir, primary_label, layers_to_do[0]
        )

    # Determine whether to skip training. If a usable w exists (explicit or default), we skip training.
    skip_training = False
    if args.w_raw_file is not None:
        skip_training = True
        logger.info(
            f"[weights] Using explicitly provided --w_raw_file: {args.w_raw_file}"
        )
    elif default_w_path is not None and os.path.exists(default_w_path):
        skip_training = True
        logger.info(
            f"[weights] Found default steering vector at {default_w_path}; will skip training and load it for steering."
        )
    elif args.all_layers:
        # In all-layers mode, skip training if all per-layer weights exist (either via pattern or default paths).
        needed_layers = list(range(1, num_layers + 1))
        missing_layers = []
        if args.w_raw_file is not None and "{layer}" in args.w_raw_file:
            # Check templated file per layer
            for L in needed_layers:
                path_L = args.w_raw_file.format(layer=L)
                if not os.path.exists(path_L):
                    missing_layers.append(L)
        else:
            # Check default path per layer
            for L in needed_layers:
                path_L = model_utils._default_w_path(args.output_dir, primary_label, L)
                if path_L is None or not os.path.exists(path_L):
                    missing_layers.append(L)
        if len(missing_layers) == 0:
            skip_training = True
            logger.info(
                "[weights] --all_layers: found all per-layer weight files; skipping training."
            )
        else:
            logger.info(
                f"[weights] --all_layers: missing weight files for layers {missing_layers[:12]}{'...' if len(missing_layers) > 12 else ''}; training will run."
            )
    else:
        logger.info(
            "[weights] No precomputed w found/passed; training would be required to create one."
        )

    if not skip_training:
        from torch.utils.data import DataLoader

        train_ds = dataset.TextDataset(prompts_train, responses_train, tokenizer)
        test_ds = dataset.TextDataset(prompts_eval, responses_eval, tokenizer)

        from transformers import DataCollatorWithPadding

        collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")

        def custom_collate(batch):
            input_ids_list = [item[0] for item in batch]
            attention_list = [item[1] for item in batch]
            resp_mask_list = [item[2] for item in batch]
            resp_eq_mask_list = [item[3] for item in batch]
            prompt_mask_list = [item[4] for item in batch]
            stmt_eos_mask_list = [item[5] for item in batch]
            idx_list = [item[6] for item in batch]
            encodings = [
                {"input_ids": ids, "attention_mask": mask}
                for ids, mask in zip(input_ids_list, attention_list)
            ]
            batch_enc = collator(encodings)
            max_len = batch_enc["input_ids"].size(1)

            def _pad_stack(masks):
                padded = []
                for m in masks:
                    pad_len = max_len - m.size(0)
                    if pad_len > 0:
                        m = torch.nn.functional.pad(m, (0, pad_len), value=False)
                    padded.append(m)
                return torch.stack(padded, dim=0)

            resp_mask_batch = _pad_stack(resp_mask_list)
            resp_eq_mask_batch = _pad_stack(resp_eq_mask_list)
            prompt_mask_batch = _pad_stack(prompt_mask_list)
            stmt_eos_mask_batch = _pad_stack(stmt_eos_mask_list)
            return (
                batch_enc["input_ids"],
                batch_enc["attention_mask"],
                resp_mask_batch,
                resp_eq_mask_batch,
                prompt_mask_batch,
                stmt_eos_mask_batch,
                torch.tensor(idx_list, dtype=torch.long),
            )

    # (skip_training already set above)

    # ---------------------- Steering/Evaluation Section: TruthfulQA MC logic ----------------------
    if args.steer_eval:
        # Example: run steering on a subset of eval_data, for TruthfulQA-style MC
        # (Assume dataset_subset is eval_data, or a subset if max_examples is set)
        dataset_subset = (
            eval_data[: args.max_examples] if args.max_examples else eval_data
        )
        for i, item in enumerate(dataset_subset):
            # Build prompt (using _norm_prompt or item["prompt"]) and optional prefill
            prompt = _norm_prompt(item)
            used_prefill = False
            # After building `prompt` above
            try:
                prefilled, used_prefill = text_processing.build_prefilled_until_equals(
                    {
                        "prompt": prompt,
                        "response": item.get("response", "") or "",
                    }
                )
                prompt_to_use = prefilled
            except Exception:
                used_prefill = False
                prompt_to_use = prompt

            # --- Build MC fields and letters ---
            letters = None
            base = (
                item.get("base", {}) if isinstance(item.get("base", {}), dict) else {}
            )
            if isinstance(base.get("letters"), str):
                letters = base["letters"].strip()

            # --- Parse alpha grid robustly and report once for sample 0 ---
            raw_alphas = [0.0] + text_processing.parse_alpha_grid(args.steer_alpha_grid)
            alpha_list, seen = [], set()
            for a in raw_alphas:
                a_norm = 0.0 if abs(float(a)) < 1e-12 else float(a)
                if a_norm not in seen:
                    seen.add(a_norm)
                    alpha_list.append(a_norm)
            if i == 0:
                print(f"[steer_eval] parsed alphas={alpha_list}")
                _LOG.debug(f"[steer_eval] parsed alphas={alpha_list}")

            # --- Common steering config seed (alpha gets set inside the loop) ---
            steer_cfg_seed = {
                "enabled": (
                    steer_w is not None
                ),  # enable only if a valid vector is loaded
                "alpha": 0.0,
                "k": args.steer_k_tokens,
                "layer": (args.steer_layer if args.steer_layer is not None else -1),
                "w": steer_w,
                "max_new_tokens": args.gen_max_new_tokens,
                "temperature": args.gen_temperature,
                "top_p": args.gen_top_p,
                "append_assistant_prefix": args.append_assistant_prefix,
                "capture_last_logits": False,
                "nullspace": {"enabled": bool(args.null_ablate), "W": None},
                "return_extra": True,
                "mc_letters": letters,
                "letters_fallback": letters or "ABCD",
                "verbose": bool(args.verbose),
            }

            # --- Sweep alphas and log EACH (sample, alpha) ---
            for alpha in alpha_list:
                eff_alpha = -alpha if args.steer_flip_sign else alpha
                steer_cfg = dict(steer_cfg_seed)
                steer_cfg["alpha"] = eff_alpha
                steer_cfg["sample_idx"] = i

                # Do not enable hook if w is missing; avoids tensor(None) error
                if steer_cfg["enabled"] and steer_cfg.get("w", None) is None:
                    steer_cfg["enabled"] = False

                try:
                    out = generate_with_optional_steer(
                        prompt_to_use, tokenizer, model, runtime_device, steer_cfg
                    )
                    if isinstance(out, tuple) and len(out) >= 2:
                        gen_text = out[0]
                        extra = (
                            out[-1]
                            if isinstance(out[-1], dict)
                            else {"prompt_text": prompt_to_use, "letters_str": letters}
                        )
                    else:
                        gen_text = out
                        extra = {"prompt_text": prompt_to_use, "letters_str": letters}
                except Exception as e:
                    gen_text = f"<GENERATION_ERROR: {e}>"
                    extra = {"prompt_text": prompt_to_use, "letters_str": letters}

                msg = f"[steer_eval][sample {i}] alpha={eff_alpha} gen={gen_text!r}"
                print(msg)
                _LOG.debug(msg)

                if args.save_outputs_json:
                    rec = {
                        "sample_idx": i,
                        "alpha": eff_alpha,
                        "used_prefill": bool(used_prefill),
                        "prompt": extra.get("prompt_text", prompt_to_use),
                        "generated": gen_text,
                        "letters": extra.get("letters_str", letters),
                        "allowed_token_ids": extra.get("allowed_ids", None),
                        "metadata": {
                            "input_json": args.input_json,
                            "eval_json": args.eval_json,
                            "model_name": args.model_name,
                            "steer_layer": args.steer_layer,
                            "steer_k_tokens": args.steer_k_tokens,
                            "temperature": args.gen_temperature,
                            "top_p": args.gen_top_p,
                            "null_ablate": bool(args.null_ablate),
                        },
                    }
                    _append_jsonl(args.save_outputs_json, rec)


if __name__ == "__main__":
    main()
