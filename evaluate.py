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
from util import nethook
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
    """
    Train model edits using the dataset at `train_path`,
    then evaluate those edits using the dataset at `test_path`.
    Optionally save the edited model.
    """

    # ==========================================================
    # Determine run directory
    # ==========================================================
    params_class, apply_algo = ALG_DICT[alg_name]

    if continue_from_run is not None:
        run_dir = RESULTS_DIR / dir_name / continue_from_run
        assert run_dir.exists(), f"{continue_from_run} must exist!"
    else:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0

        run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results will be stored at {run_dir}")

    # ==========================================================
    # Load hyperparameters
    # ==========================================================
    params_path = (
        run_dir / "params.json"
        if continue_from_run is not None
        else HPARAMS_DIR / alg_name / hparams_fname
    )
    hparams = params_class.from_json(params_path)

    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    print(f"Executing {alg_name} with parameters {hparams}")

    # ==========================================================
    # Load model
    # ==========================================================
    print("Instantiating model")

    if type(model_name) is str:
        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name

    # ==========================================================
    # Load datasets from absolute paths
    # ==========================================================
    print("Loading dataset, attribute snippets, tf-idf data")

    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    ds_class, ds_eval_method = DS_DICT[ds_name]

    print(f"[INFO] Training dataset path: {train_path}")
    print(f"[INFO] Testing  dataset path: {test_path}")

    train_ds = ds_class(train_path, size=dataset_size_limit, tok=tok)
    test_ds = ds_class(test_path, size=dataset_size_limit, tok=tok)

    # ==========================================================
    # TRAINING PHASE
    # ==========================================================
    print("\n===== TRAINING PHASE =====\n")

    for record in train_ds:
        case_id = record["case_id"]
        print(f"[TRAIN] Editing case {case_id}")

        start = time()
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )

        edited_model, weights_copy = apply_algo(
            model,
            tok,
            [record["requested_rewrite"]],
            hparams,
            copy=False,
            return_orig_weights=True,
            **args_conserve_memory,
        )

        exec_time = time() - start
        print(f"[TRAIN] Execution took {exec_time}")

        # apply edit to the working model
        model = edited_model

    # ==========================================================
    # SAVE TRAINED MODEL
    # ==========================================================
    if save_model_dir is not None:
        print(f"[INFO] Saving fully edited model to: {save_model_dir}")
        os.makedirs(save_model_dir, exist_ok=True)
        model.save_pretrained(save_model_dir)
        tok.save_pretrained(save_model_dir)
        print("[INFO] Model saved successfully.\n")

    # ==========================================================
    # TESTING PHASE
    # ==========================================================
    print("\n===== TESTING PHASE =====\n")

    for record in test_ds:
        case_id = record["case_id"]
        case_result_path = run_dir / f"case_{case_id}.json"

        print(f"[TEST] Evaluating case {case_id}")

        metrics = {
            "case_id": case_id,
            "requested_rewrite": record["requested_rewrite"],
            "post": ds_eval_method(model, tok, record, snips, vec),
        }

        with open(case_result_path, "w") as f:
            json.dump(metrics, f, indent=1)

        print(f"[TEST] Saved results for case {case_id}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--alg_name", choices=["ROME", "FT", "KN", "MEND"], required=True)
    parser.add_argument("--model_name", choices=["gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/gpt-j-6B"], required=True)
    parser.add_argument("--hparams_fname", type=str, required=True)
    parser.add_argument("--ds_name", choices=["cf", "zsre"], default="cf")
    parser.add_argument("--continue_from_run", type=str, default=None)
    parser.add_argument("--dataset_size_limit", type=int, default=10000)
    parser.add_argument("--skip_generation_tests", action="store_true")
    parser.add_argument("--conserve_memory", action="store_true")

    # NEW: absolute dataset paths
    parser.add_argument("--train_path", type=str, required=True, help="Full path to training dataset folder")
    parser.add_argument("--test_path", type=str, required=True, help="Full path to testing dataset folder")

    # NEW: save edited model
    parser.add_argument("--save_model_dir", type=str, default=None, help="Directory to save edited model")

    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
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
