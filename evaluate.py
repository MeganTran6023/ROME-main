import json
import os
import shutil
from pathlib import Path
from time import time
from typing import Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines.ft import FTHyperParams, apply_ft_to_model
from baselines.kn import KNHyperParams, apply_kn_to_model
from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    get_tfidf_vectorizer,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from rome import ROMEHyperParams, apply_rome_to_model
from util.globals import *

ALG_DICT = {
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "KN": (KNHyperParams, apply_kn_to_model),
    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
}

DS_DICT = {
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
}

# ----------------------------------------------------------------------
# Fix target strings for ROME
# ----------------------------------------------------------------------
def fix_target_string(s: str) -> str:
    """Ensures ROME receives a non-empty string starting with a space."""
    if s is None:
        return " UNKNOWN"
    s = str(s).strip()
    if len(s) == 0:
        return " UNKNOWN"
    if not s.startswith(" "):
        s = " " + s
    return s


def build_requested_rewrite_if_missing(record):
    """Ensure record has a valid ROME-style 'requested_rewrite'."""
    if "requested_rewrite" not in record:
        subject = record.get("subject", "ENTITY")
        answers = record.get("answers", ["UNKNOWN"])
        target = fix_target_string(answers[0])

        record["requested_rewrite"] = {
            "prompt": record["src"].replace(subject, "{}"),
            "subject": subject,
            "target_new": {"str": target},
            "target_true": {"str": "<|endoftext|>"},
        }
    else:
        # Ensure valid target string
        record["requested_rewrite"]["target_new"]["str"] = fix_target_string(
            record["requested_rewrite"]["target_new"]["str"]
        )

    return record


def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    conserve_memory: bool,
    dir_name: str,
    train_path: str,
    test_path: str,
    save_model_dir: str,
):

    # ==========================================================
    # Create run directory
    # ==========================================================
    params_class, apply_algo = ALG_DICT[alg_name]

    run_dir = Path("ROME_test_results/results")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Results stored at: {run_dir}")

    # ==========================================================
    # Load hyperparameters
    # ==========================================================
    params_path = HPARAMS_DIR / alg_name / hparams_fname
    hparams = params_class.from_json(params_path)
    print(f"[INFO] Loaded hparams: {params_path}")

    # ==========================================================
    # Load model
    # ==========================================================
    print("[INFO] Loading model...")

    if isinstance(model_name, str):
        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name

    # ==========================================================
    # Load datasets
    # ==========================================================
    print("[INFO] Loading datasets...")

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Limit train set = exactly N samples
    train_data = train_data[:dataset_size_limit]

    print(f"[INFO] Loaded {len(train_data)} training samples")
    print(f"[INFO] Loaded {len(test_data)} test samples")

    # ==========================================================
    # TRAINING PHASE
    # ==========================================================
    print("\n===== TRAINING =====\n")

    for idx, record in enumerate(train_data):
        print(f"[TRAIN] Editing case {idx}")

        record = build_requested_rewrite_if_missing(record)

        start = time()

        edited_model, weights_copy = apply_algo(
            model,
            tok,
            [record["requested_rewrite"]],
            hparams,
            copy=False,
            return_orig_weights=True,
        )

        print(f"[TRAIN] Edit time: {time() - start:.2f}s")
        model = edited_model

    # ==========================================================
    # SAVE TRAINED MODEL
    # ==========================================================
    if save_model_dir:
        print(f"[INFO] Saving edited model → {save_model_dir}")
        os.makedirs(save_model_dir, exist_ok=True)
        model.save_pretrained(save_model_dir)
        tok.save_pretrained(save_model_dir)
        print("[INFO] Model saved.\n")

    # ==========================================================
    # TESTING PHASE
    # ==========================================================
    print("\n===== TESTING =====\n")

    _, ds_eval_method = DS_DICT[ds_name]

    for record in test_data:
        record = build_requested_rewrite_if_missing(record)
        case_id = record.get("case_id", "unknown")

        print(f"[TEST] Case {case_id}")

        metrics = {
            "case_id": case_id,
            "requested_rewrite": record["requested_rewrite"],
            "post": ds_eval_method(model, tok, record, None, None),
        }

        with open(run_dir / f"case_{case_id}.json", "w") as f:
            json.dump(metrics, f, indent=1)

    print("\n===== DONE — ALL TEST RESULTS SAVED =====\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--alg_name", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--hparams_fname", required=True)
    parser.add_argument("--ds_name", required=True)
    parser.add_argument("--dataset_size_limit", type=int, default=20)
    parser.add_argument("--continue_from_run", default=None)
    parser.add_argument("--skip_generation_tests", action="store_true")
    parser.add_argument("--conserve_memory", action="store_true")

    parser.add_argument("--train_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--save_model_dir", required=True)

    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.conserve_memory,
        dir_name=args.alg_name,
        train_path=args.train_path,
        test_path=args.test_path,
        save_model_dir=args.save_model_dir,
    )
