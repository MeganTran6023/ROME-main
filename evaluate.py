import json
import os
import shutil
from pathlib import Path
from time import time
from typing import Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util.globals import DATA_DIR, RESULTS_DIR, HPARAMS_DIR

#from baselines.efk import EFKHyperParams, EfkRewriteExecutor
from baselines.ft import FTHyperParams, apply_ft_to_model
from baselines.kn import KNHyperParams, apply_kn_to_model
from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    get_tfidf_vectorizer,
    WikibioDataset, # Your custom class
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
# 	"KE": (EFKHyperParams, EfkRewriteExecutor().apply_to_model),
}

DS_DICT = {
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
    "wikibio": (WikibioDataset, compute_rewrite_quality_zsre),
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
    # --- ADDED ARGUMENTS FOR CUSTOM PATHS/CONFIG ---
    config_path: str, # Retain config_path for reference, although it's not the primary hparams source here
    train_path: str,
    test_path: str,
    save_model_dir: str,
    # -----------------------------------------------
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Determine run directory
    if continue_from_run is not None:
        run_dir = RESULTS_DIR / dir_name / continue_from_run
        assert (
            run_dir.exists()
        ), f"If continuing from run, {continue_from_run} must exist!"
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

    # --- START OF HYPERPARAMETER LOADING FIX (Type Error Resolution) ---
    params_path = (
        run_dir / "params.json"
        if continue_from_run is not None
        else HPARAMS_DIR / alg_name / hparams_fname
    )

    # 1. Manually load data from the JSON file
    with open(params_path, "r") as f:
        raw_hparams_data = json.load(f)

    # 2. Filter out all known conflicting run metadata
    # This addresses all TypeErrors where evaluation/run metadata leaks into ROMEHyperParams
    for key in [
        'alg_name', 'model_name', 'device', 'results_dir', 'seed', 'archive', 
        'n_edits', 'model_parallel', 'v_num', 'config_path', 'train_path', 
        'test_path', 'save_model_dir', 'dataset_size_limit', 'continue_from_run',
        'skip_generation_tests', 'conserve_memory', 'hparams_fname', 'ds_name', 'dir_name'
    ]:
        if key in raw_hparams_data:
            raw_hparams_data.pop(key)

    # 3. Instantiate the clean ROMEHyperParams object (resolves TypeErrors)
    hparams = params_class(**raw_hparams_data)

    # 4. Save params.json for the run, as in original logic
    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    # --- END OF HYPERPARAMETER LOADING FIX ---

    print(f"Executing {alg_name} with parameters {hparams}")

    # Instantiate vanilla model
    print("Instantiating model")
    if type(model_name) is str:
        # --- START OF MODEL LOADING OPTIMIZATION FIX ---
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True, 
            torch_dtype=torch.float16, 
            device_map="auto"
        ).cuda()
        # --- END OF MODEL LOADING OPTIMIZATION FIX ---
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name

    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    # Load dataset using the class selected by ds_name
    ds_class, ds_eval_method = DS_DICT[ds_name]
    
    # Instantiate the dataset class
    ds = ds_class(DATA_DIR, size=dataset_size_limit, tok=tok)

    # Iterate through dataset
    for record in ds:
        case_id = record["case_id"]
        case_result_path = run_dir / f"case_{case_id}.json"
        if not case_result_path.exists():
            # Compute weight changes + record weights that changed
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
            print("Execution took", exec_time)

            # Execute evaluation suite
            start = time()
            metrics = {
                "case_id": case_id,
                "requested_rewrite": record["requested_rewrite"],
                "time": exec_time,
                "post": ds_eval_method(edited_model, tok, record, snips, vec),
            }

            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(model, k)[...] = v.to("cuda")
            metrics["pre"] = ds_eval_method(model, tok, record, snips, vec)

            print("Evaluation took", time() - start)

            # Dump metrics in .json
            with open(case_result_path, "w") as f:
                json.dump(metrics, f, indent=1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    
    # --- ROME CORE ARGS ---
    parser.add_argument(
        "--alg_name",
        choices=["ROME", "FT", "KN", "MEND", "KE"],
        default="ROME",
        help="Editing algorithm to use.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        choices=["gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/gpt-j-6B"],
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["cf", "zsre", "wikibio"],
        default="cf",
        help="Dataset to perform evaluations on.",
    )
    
    # --- OPTIONAL RUN ARGS ---
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id.",
    )
    parser.add_argument(
        # Note: This is dynamically set by the command, overriding the default
        "--dataset_size_limit",
        type=int,
        default=10000,
        help="Truncate dataset to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation.",
    )
    
    # --- CUSTOM/UNRECOGNIZED ARGS ---
    # These are added back to stop the 'unrecognized arguments' error.
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to a full configuration JSON file (if needed).",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default=None,
        help="Path to the training data file.",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default=None,
        help="Path to the testing data file.",
    )
    parser.add_argument(
        "--save_model_dir",
        type=str,
        default=None,
        help="Directory to save the edited model checkpoint.",
    )
    
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
        # --- PASSING ALL CUSTOM/UNRECOGNIZED ARGS TO MAIN ---
        config_path=args.config_path,
        train_path=args.train_path,
        test_path=args.test_path,
        save_model_dir=args.save_model_dir,
        # ----------------------------------------------------
    )
