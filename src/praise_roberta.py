#!/usr/bin/env python3
"""
RoBERTa classifier fine-tuning for 3-class praise:
    negative (-1), neutral (0), positive (1)

Input data: the praise-only JSON produced by datasets/praise_data.py, e.g.:
[
  {"prompt": "Human: \n\n", "response": "Assistant: you are fantastic.", "praise_label": "positive"},
  ...
]

Training uses Hugging Face `Trainer`. Internally the model uses label ids {0,1,2},
which are mapped to the desired external labels {-1,0,1} via LABEL_INDEX and INDEX_LABEL.
We also save an `id_map.json` in the output directory to preserve this mapping.

Usage examples:
    python praise_roberta \
        --train_json prompts/subset/praise_3class_all.json \
        --eval_json  prompts/subset/praise_3class_all.json \
        --model_name roberta-base \
        --output_dir runs/praise_roberta_base \
        --epochs 3 --lr 2e-5 --train_bs 32 --eval_bs 64

To export predictions with external labels (-1,0,1):
    python praise_roberta --predict_json PATH.json --ckpt runs/praise_roberta_base \
        --output_preds runs/praise_roberta_base/preds.json
"""
from __future__ import annotations

import os
import argparse
from typing import Dict, List
import inspect
import csv

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)
from sklearn.metrics import classification_report

from .utils import file_io, evaluation

def probe(args):
    """Probe a fine-tuned checkpoint on a list of phrases.
    Builds 'Assistant: you are <word>.' strings and prints / saves probabilities.
    """
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.ckpt)
    model.eval()
    device = next(model.parameters()).device
    model.to(device)

    # Build phrases
    phrases: List[str] = []
    if args.phrases:
        phrases = [p.strip() for p in args.phrases.split(",") if p.strip()]
    if args.phrases_file:
        with open(args.phrases_file, "r", encoding="utf-8") as fh:
            for line in fh:
                t = line.strip()
                if t:
                    phrases.append(t)
    if not phrases:
        phrases = [
            # positives
            "brilliant",
            "fantastic",
            "amazing",
            "excellent",
            "smart",
            "kind",
            "helpful",
            # neutrals
            "okay",
            "fine",
            "average",
            "adequate",
            # negatives
            "terrible",
            "awful",
            "bad",
            "incompetent",
            "rude",
        ]

    texts = [f"Assistant: you are {w}." for w in phrases]
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    id2label = getattr(
        model.config, "id2label", {0: "negative", 1: "neutral", 2: "positive"}
    )

    # Print to stdout
    print("id2label:", id2label)
    print("Rows: word, probs[neg,neu,pos], pred_label")
    rows = []
    for w, p in zip(phrases, probs):
        pred_idx = int(np.argmax(p))
        pred_name = str(id2label.get(pred_idx, pred_idx))
        row = {
            "word": w,
            "neg_prob": float(p[0]),
            "neu_prob": float(p[1]),
            "pos_prob": float(p[2]),
            "pred_label": pred_name,
        }
        rows.append(row)
        print(f"  {w:>12s} -> {[round(float(x),4) for x in p.tolist()]}  {pred_name}")

    # Optionally write CSV
    if args.output_csv:
        fn = args.output_csv
        fieldnames = ["word", "neg_prob", "neu_prob", "pos_prob", "pred_label"]
        write_header = not os.path.exists(fn)
        with open(fn, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Wrote probe CSV -> {fn}")


# --------------------
# Label mappings
# --------------------
# External label semantics requested by user
EXTERNAL_LABELS = {"negative": -1, "neutral": 0, "positive": 1}
# Internal indices required by HF (0..2)
LABEL_INDEX = {"negative": 0, "neutral": 1, "positive": 2}
INDEX_LABEL = {v: k for k, v in LABEL_INDEX.items()}
INDEX_EXTERNAL = {i: EXTERNAL_LABELS[INDEX_LABEL[i]] for i in INDEX_LABEL}


class PraiseDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    @classmethod
    def from_json(cls, path: str, tokenizer, max_length: int = 64):
        raw = file_io.load_json(path)
        # Expect fields: response, praise_label
        rows = []
        for r in raw:
            lbl = r.get("praise_label")
            if lbl not in LABEL_INDEX:
                continue
            rows.append(
                {
                    "text": r["response"],
                    "label": LABEL_INDEX[lbl],
                }
            )
        return cls(rows, tokenizer, max_length)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["labels"] = torch.tensor(item["label"], dtype=torch.long)
        return enc


def build_model(model_name: str, num_labels: int = 3):
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label={i: INDEX_LABEL[i] for i in range(num_labels)},
        label2id={INDEX_LABEL[i]: i for i in range(num_labels)},
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=config
    )
    return model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    metrics = evaluation.compute_classification_metrics(labels.tolist(), preds.tolist())
    # Add micro F1 since our util uses weighted by default
    from sklearn.metrics import f1_score
    metrics["f1_micro"] = f1_score(labels, preds, average="micro")
    return metrics


def save_id_map(output_dir: str):
    path = os.path.join(output_dir, "id_map.json")
    id_map_data = {
        "index_to_internal_label": {
            str(i): INDEX_LABEL[i] for i in INDEX_LABEL
        },
        "index_to_external_int": {
            str(i): INDEX_EXTERNAL[i] for i in INDEX_EXTERNAL
        },
        "internal_label_to_external_int": {
            k: EXTERNAL_LABELS[k] for k in EXTERNAL_LABELS
        },
    }
    file_io.save_json(id_map_data, path)


def train(args):
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    train_ds = PraiseDataset.from_json(args.train_json, tokenizer, args.max_length)
    eval_ds = (
        PraiseDataset.from_json(args.eval_json, tokenizer, args.max_length)
        if args.eval_json
        else None
    )

    model = build_model(args.model_name, num_labels=3)

    # Build TrainingArguments in a version-agnostic way by filtering kwargs
    def _build_training_args():

        # Desired defaults (we'll filter against the actual signature)
        desired = {
            # always safe basics
            "output_dir": args.output_dir,
            "overwrite_output_dir": True,
            "num_train_epochs": args.epochs,
            "learning_rate": args.lr,
            "logging_steps": max(1, args.eval_steps // 2),
            # batch size keys across versions
            "per_device_train_batch_size": args.train_bs,
            "per_device_eval_batch_size": args.eval_bs,
            "per_gpu_train_batch_size": args.train_bs,
            "per_gpu_eval_batch_size": args.eval_bs,
            # saving/logging knobs that exist in many versions
            "save_steps": args.eval_steps,
            "save_total_limit": 2,
            # sched/regularization if available
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "lr_scheduler_type": args.lr_scheduler,
            # precision/accumulation
            "fp16": args.fp16,
            "bf16": args.bf16,
            "gradient_accumulation_steps": args.grad_accum,
            # new-style evaluation controls (will be filtered out if unsupported)
            "evaluation_strategy": "no",  # we'll call evaluate() after training
            "save_strategy": "steps",
            # logging destination
            "report_to": [],
        }

        sig = inspect.signature(TrainingArguments.__init__)
        allowed = set(sig.parameters.keys())
        # remove 'self'
        allowed.discard("self")
        filtered = {k: v for k, v in desired.items() if k in allowed}
        return TrainingArguments(**filtered)

    training_args = _build_training_args()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    if eval_ds is not None:
        metrics = trainer.evaluate()
        print("\nValidation metrics:")
        for k, v in metrics.items():
            if isinstance(v, (float, int)):
                print(f"  {k}: {v:.4f}")

        # Detailed report with external labels
        pred_output = trainer.predict(eval_ds)
        preds = np.argmax(pred_output.predictions, axis=-1)
        # Collect true labels by iterating (avoids Dataset slicing issues)
        true_labels = []
        for i in range(len(eval_ds)):
            item = eval_ds[i]
            true_labels.append(int(item["labels"]))
        y_true_ext = [INDEX_EXTERNAL[y] for y in true_labels]
        y_pred_ext = [INDEX_EXTERNAL[int(y)] for y in preds]
        print("\nClassification report (external labels -1,0,1):")
        print(classification_report(y_true_ext, y_pred_ext, digits=4))

    # Quick sanity probe on three examples using the current (trained) model
    try:
        probe_texts = [
            "Assistant: you are brilliant.",
            "Assistant: you are terrible.",
            "Assistant: you are okay.",
        ]
        # Ensure we reference the trainer's model (it may be wrapped/moved by Trainer)
        _model = trainer.model if "trainer" in locals() else model
        _model.eval()
        # Determine device from model parameters
        try:
            device = next(_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        enc = tokenizer(
            probe_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = _model(**enc).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        print(f"\nSanity probe on device: {device} (probs [neg, neu, pos]):")
        for t, p in zip(probe_texts, probs):
            print(f"  {t} -> {[round(float(x), 4) for x in p.tolist()]}")
    except Exception as e:
        print(f"Sanity probe failed: {e}")

    # Save id map
    os.makedirs(args.output_dir, exist_ok=True)
    save_id_map(args.output_dir)

    # Save final model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSaved model + tokenizer to {args.output_dir}")

    # ---------------------------------------------------------------
    # Fresh-reload sanity probe from disk to rule out device/wrapper issues
    # ---------------------------------------------------------------
    try:
        print("\n[Fresh reload sanity probe]")
        re_tok = AutoTokenizer.from_pretrained(args.output_dir, use_fast=True)
        re_model = AutoModelForSequenceClassification.from_pretrained(
            args.output_dir
        ).eval()
        re_device = (
            next(re_model.parameters()).device
            if any(p.requires_grad for p in re_model.parameters())
            else torch.device("cpu")
        )
        re_model.to(re_device)
        # Use canonical phrases that appear in training lexicon
        probe_pairs = [
            ("Assistant: you are fantastic.", "positive"),
            ("Assistant: you are terrible.", "negative"),
            ("Assistant: you are okay.", "neutral"),
            ("Assistant: you are brilliant.", "positive"),
            ("Assistant: you are fine.", "neutral"),
        ]
        texts = [t for t, _ in probe_pairs]
        enc = re_tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )
        enc = {k: v.to(re_device) for k, v in enc.items()}
        with torch.no_grad():
            probs = torch.softmax(re_model(**enc).logits, dim=-1).cpu().numpy()
        id2label = getattr(
            re_model.config, "id2label", {0: "negative", 1: "neutral", 2: "positive"}
        )
        print("id2label:", id2label)
        print("Probs rows are [neg, neu, pos]")
        ok = True
        for (txt, expect), p in zip(probe_pairs, probs):
            pred_idx = int(np.argmax(p))
            pred_name = str(id2label.get(pred_idx, pred_idx)).lower()
            print(
                f"  {txt} -> {p.round(4).tolist()}  pred={pred_name}  expected={expect}"
            )
            # loose check: predicted matches expected OR expected prob is clearly highest
            exp_idx = {"negative": 0, "neu": 1, "neutral": 1, "positive": 2}.get(
                expect, None
            )
            if exp_idx is not None and pred_idx != exp_idx:
                # allow margin check
                if not (p[exp_idx] >= p.max() - 0.05):
                    ok = False
        if not ok:
            print(
                "[WARN] Fresh reload probe suggests weak or misaligned predictions.\n"
                "       If training metrics were strong, double-check that the texts match training surface form\n"
                "       (lowercase 'you are', period), and confirm id2label ordering."
            )
    except Exception as e:
        print(f"Fresh reload probe failed: {e}")


def predict(args):
    # Load tokenizer & model from checkpoint dir (args.ckpt)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.ckpt)

    # Prepare dataset
    raw = file_io.load_json(args.predict_json)
    data = [
        {"text": r["response"], "label": LABEL_INDEX.get(r.get("praise_label"), 1)}
        for r in raw
    ]

    class _TmpDS(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            x = self.rows[idx]
            enc = tokenizer(
                x["text"],
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            enc = {k: v.squeeze(0) for k, v in enc.items()}
            enc["labels"] = torch.tensor(x["label"], dtype=torch.long)
            return enc

    ds = _TmpDS(data)

    trainer = Trainer(model=model, tokenizer=tokenizer)
    outputs = trainer.predict(ds)
    preds = np.argmax(outputs.predictions, axis=-1)

    # Map to external ints (-1,0,1)
    preds_external = [INDEX_EXTERNAL[int(p)] for p in preds]

    out_rows = []
    for r, ext in zip(raw, preds_external):
        out_rows.append(
            {
                "response": r.get("response"),
                "praise_label": r.get("praise_label"),
                "pred": int(ext),
            }
        )

    file_io.save_json(out_rows, args.output_preds)
    print(f"Wrote predictions to {args.output_preds}")


def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    # Train subcommand
    pt = sub.add_parser("train")
    pt.add_argument("--train_json", type=str, required=True)
    pt.add_argument("--eval_json", type=str, default=None)
    pt.add_argument("--model_name", type=str, default="roberta-base")
    pt.add_argument("--output_dir", type=str, required=True)
    pt.add_argument("--epochs", type=float, default=3.0)
    pt.add_argument("--lr", type=float, default=2e-5)
    pt.add_argument("--train_bs", type=int, default=32)
    pt.add_argument("--eval_bs", type=int, default=64)
    pt.add_argument("--eval_steps", type=int, default=100)
    pt.add_argument("--warmup_ratio", type=float, default=0.06)
    pt.add_argument("--weight_decay", type=float, default=0.01)
    pt.add_argument("--lr_scheduler", type=str, default="linear")
    pt.add_argument("--max_length", type=int, default=64)
    pt.add_argument("--seed", type=int, default=1337)
    pt.add_argument("--grad_accum", type=int, default=1)
    pt.add_argument("--fp16", action="store_true")
    pt.add_argument("--bf16", action="store_true")

    # Predict subcommand
    pp = sub.add_parser("predict")
    pp.add_argument("--predict_json", type=str, required=True)
    pp.add_argument(
        "--ckpt", type=str, required=True, help="Path to fine-tuned checkpoint dir"
    )
    pp.add_argument("--output_preds", type=str, required=True)
    pp.add_argument("--max_length", type=int, default=64)

    # Probe subcommand
    pprobe = sub.add_parser("probe")
    pprobe.add_argument(
        "--ckpt", type=str, required=True, help="Path to fine-tuned checkpoint dir"
    )
    pprobe.add_argument(
        "--phrases",
        type=str,
        default="",
        help='Comma-separated adjectives (e.g., "brilliant,fantastic,okay,terrible")',
    )
    pprobe.add_argument(
        "--phrases_file",
        type=str,
        default="",
        help="Optional file with one adjective per line",
    )
    pprobe.add_argument(
        "--output_csv", type=str, default="", help="Optional path to append CSV results"
    )
    pprobe.add_argument("--max_length", type=int, default=64)

    # Backward-compat single-step (no subcommand): treat as train
    p.add_argument("--train_json", type=str)
    p.add_argument("--eval_json", type=str)
    p.add_argument("--model_name", type=str, default="roberta-base")
    p.add_argument("--output_dir", type=str)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--train_bs", type=int, default=32)
    p.add_argument("--eval_bs", type=int, default=64)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--lr_scheduler", type=str, default="linear")
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")

    args = p.parse_args()

    # If a subcommand is used, return it directly
    if args.cmd is not None:
        return args

    # Otherwise, require the train args to be present
    missing = [k for k in (args.train_json, args.output_dir) if k is None]
    if missing:
        p.error(
            "Either use subcommands `train` / `predict`, or provide --train_json and --output_dir."
        )
    args.cmd = "train"
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "predict":
        predict(args)
    elif args.cmd == "probe":
        probe(args)
    else:
        raise SystemExit("Unknown cmd")
