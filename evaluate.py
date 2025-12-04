import json
import os
import shutil
from pathlib import Path
from time import time
from typing import Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util.globals import DATA_DIR, RESULTS_DIR, HPARAMS_DIR

from baselines.ft import FTHyperParams, apply_ft_to_model
from baselines.kn import KNHyperParams, apply_kn_to_model
from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    get_tfidf_vectorizer,
    WikibioDataset,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from rome import ROMEHyperParams, apply_rome_to_model
from util import nethook


ALG_DICT = {
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "KN": (KNHyperParams, apply_kn_to_model),
    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
}

DS_DICT = {
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
    "wikibio": (WikibioDataset, compute_rewrite_quality_zsre),
}


def load_dataset(ds_name, train_path, test_path, tok, size):
    ds_class, ds_eval_method = DS_DICT[ds_name]

    # Special handling for WikiBio
    if ds_name == "wikibio":
        if test_path:
            print(f"🔹 Using test dataset: {test_path}")
            ds = ds_class(Path(test_path), size=size, tok=tok)
        else:
            print(f"⚠️ No test_path provided, falling back to DATA_DIR")
            ds = ds_class(DATA_DIR, size=size, tok=tok)
    else:
        ds = ds_class(DATA_DIR, size=size, tok=tok)

    return ds, ds_eval_method


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
    config_path: str,
    train_path: str,
    test_path: str,
    save_model_dir: str,
):
    params_class, apply_algo = ALG_DICT[alg_name]

    # Run folder setup
    if continue_from_run:
        run_dir = RESULTS_DIR / dir_name / continue_from_run
        assert run_dir.exists(), "Continue run does not exist!"
    else:
        alg_dir = RESULTS_DIR / dir_name
        ids = [
            int(str(x).split("_")[-1])
            for x in alg_dir.iterdir()
            if str(x).split("_")[-1].isnumeric()
        ] if alg_dir.exists() else []
        run_id = max(ids) + 1 if ids else 0
        run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")

    if config_path:
        shutil.copyfile(config_path, run_dir / "config.json")

    # Load hyperparams
    params_path = (
        run_dir / "params.json" if continue_from_run
        else HPARAMS_DIR / alg_name / hparams_fname
    )
    with open(params_path) as f:
        raw = json.load(f)

    # Clean metadata keys
    for k in list(raw.keys()):
        if k not in params_class.__dataclass_fields__:
            raw.pop(k)
    hparams = params_class(**raw)
    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")

    # Load model
    print("Instantiating model")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
    ).cuda()
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token

    # Load data
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None
    ds, ds_eval_method = load_dataset(ds_name, train_path, test_path, tok, dataset_size_limit)

    for record in ds:
        case_id = record["case_id"]
        out_path = run_dir / f"case_{case_id}.json"
        if out_path.exists():
            continue

        # Build rewrite request
        requested_rewrite = {
            "prompt": record["text"],
            "target_new": record["labels"],
            "subject": record["concept"],
        }
        print(f"\n➡️ Editing case {case_id} — {record['concept']}")
        start = time()
        mem_args = dict(return_orig_weights_device="cpu") if conserve_memory else {}
        edited_model, weights_copy = apply_algo(
            model, tok, [requested_rewrite], hparams,
            copy=False, return_orig_weights=True, **mem_args
        )
        exec_time = time() - start
        print(f"Edit time: {exec_time:.1f}s")

        start = time()
        metrics = {
            "case_id": case_id,
            "requested_rewrite": requested_rewrite,
            "time": exec_time,
            "post": ds_eval_method(edited_model, tok, record, snips, vec),
        }

        with torch.no_grad():
            for k, v in weights_copy.items():
                nethook.get_parameter(model, k)[...] = v.to("cuda")

        metrics["pre"] = ds_eval_method(model, tok, record, snips, vec)
        print(f"Eval time: {time() - start:.1f}s")

        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)

    # Save edited model
    if save_model_dir:
        os.makedirs(save_model_dir, exist_ok=True)
        model.save_pretrained(save_model_dir)
        tok.save_pretrained(save_model_dir)
        print(f"\n💾 Model saved → {save_model_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--alg_name", choices=["ROME", "FT", "KN", "MEND"], required=True)
    parser.add_argument("--model_name", default="gpt2-xl", required=True)
    parser.add_argument("--hparams_fname", default="gpt2-xl.json", required=True)
    parser.add_argument("--ds_name", choices=["cf", "zsre", "wikibio"], default="wikibio")
    parser.add_argument("--continue_from_run", default=None)
    parser.add_argument("--dataset_size_limit", type=int, default=10000)
    parser.add_argument("--skip_generation_tests", action="store_true")
    parser.add_argument("--conserve_memory", action="store_true")

    parser.add_argument("--config_path", default=None)
    parser.add_argument("--train_path", default=None)
    parser.add_argument("--test_path", default=None)
    parser.add_argument("--save_model_dir", default=None)

    args = parser.parse_args()

    main(
        args.alg_name, args.model_name, args.hparams_fname, args.ds_name,
        args.dataset_size_limit, args.continue_from_run,
        args.skip_generation_tests, args.conserve_memory,
        dir_name=args.alg_name,
        config_path=args.config_path,
        train_path=args.train_path,
        test_path=args.test_path,
        save_model_dir=args.save_model_dir,
    )
