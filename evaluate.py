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
# Functions to guarantee ZSRE fields exist
# ----------------------------------------------------------------------

def ensure_target_has_space(target_str):
    """ROME requires the target string to begin with a space."""
    if not isinstance(target_str, str) or len(target_str.strip()) == 0:
        return " <empty>"
    if not target_str.startswith(" "):
        return " " + target_str
    return target_str


def build_requested_rewrite(record):
    """Ensure ZSRE fields exist even for custom multilingual data."""
    if "src" not in record or "answers" not in record:
        raise ValueError("ZSRE record missing src or answers fields")

    subject = record.get("subject", "ENTITY")
    answer = ensure_target_has_space(record["answers"][0])

    return {
        "prompt": record["src"].replace(subject, "{}"),
        "subject": subject,
        "target_new": {"str": answer},
        "target_true": {"str": "<|endoftext|>"},
    }


def ensure_zsre_record(record):
    """Ensures paraphrase_prompts and neighborhood_prompts exist."""
    # Requested rewrite
    if "requested_rewrite" not in record:
        record["requested_rewrite"] = build_requested_rewrite(record)

    # Paraphrase fallback
    if "paraphrase_prompts" not in record:
        record["paraphrase_prompts"] = [record["src"]]

    # Neighborhood fallback
    if "neighborhood_prompts" not in record:
        record["neighborhood_prompts"] = [{
            "prompt": record["src"],
            "target": record["answers"][0],
        }]

    return record


# ======================================================================
# MAIN
# ======================================================================

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
    # RUN DIRECTORY
    # ==========================================================
    run_dir = Path("ROME_test_results/results")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Results will be stored at: {run_dir}")

    # ==========================================================
    # Load hparams
    # ==========================================================
    params_class, apply_algo = ALG_DICT[alg_name]
    params_path = HPARAMS_DIR / alg_name / hparams_fname
    hparams = params_class.from_json(params_path)

    print(f"[INFO] Using hyperparameters from: {params_path}")

    # ==========================================================
    # Load model
    # ==========================================================
    print("[INFO] Loading model...")

    if os.path.exists(save_model_dir):
        print(f"[INFO] Loading SAVED English-edited model from: {save_model_dir}")
        model = AutoModelForCausalLM.from_pretrained(save_model_dir).cuda()
        tok = AutoTokenizer.from_pretrained(save_model_dir)
    else:
        print(f"[INFO] Loading fresh model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        tok = AutoTokenizer.from_pretrained(model_name)

    tok.pad_token = tok.eos_token

    # ==========================================================
    # Load datasets
    # ==========================================================
    print("[INFO] Loading datasets...")

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    train_data = train_data[:dataset_size_limit]

    print(f"[INFO] Loaded {len(train_data)} training samples")
    print(f"[INFO] Loaded {len(test_data)} test samples")

    # ==========================================================
    # TRAINING PHASE (only if model not already saved)
    # ==========================================================
    if not os.path.exists(save_model_dir):
        print("\n===== TRAINING (English) =====\n")

        for idx, record in enumerate(train_data):
            print(f"[TRAIN] Editing case {idx}")

            record = ensure_zsre_record(record)
            request = record["requested_rewrite"]

            start = time()

            edited_model, weights_copy = apply_algo(
                model,
                tok,
                [request],
                hparams,
                copy=False,
                return_orig_weights=True
            )

            model = edited_model
            print(f"[TRAIN] Done in {time() - start:.2f}s")

        print(f"[INFO] Saving edited English model to: {save_model_dir}")
        Path(save_model_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_model_dir)
        tok.save_pretrained(save_model_dir)
    else:
        print("[INFO] Skipping training — pre-edited English model already exists.")

    # ==========================================================
    # TESTING PHASE (French)
    # ==========================================================
    print("\n===== TESTING (French) =====\n")

    _, ds_eval_method = DS_DICT[ds_name]

    for record in test_data:
        record = ensure_zsre_record(record)
        case_id = record.get("case_id", "unknown")

        print(f"[TEST] Case {case_id}")

        result = ds_eval_method(model, tok, record, None, None)

        # Save results
        out_file = run_dir / f"case_{case_id}.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=1)

    print("\n===== ALL TEST RESULTS SAVED =====\n")


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
