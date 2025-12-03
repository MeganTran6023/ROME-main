import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from util.globals import *

REMOTE_URL = f"{REMOTE_ROOT_URL}/data/dsets/zsre_mend_eval.json"


class MENDQADataset:
    def __init__(self, data_dir: str, size: int, tok: AutoTokenizer, *args, **kwargs):
        data_dir = Path(data_dir)
        zsre_loc = data_dir / "zsre_mend_eval.json"

        if not zsre_loc.exists():
            print(f"{zsre_loc} does not exist. Downloading...")
            data_dir.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(REMOTE_URL, zsre_loc)

        with open(zsre_loc, "r") as f:
            raw = json.load(f)

        # 👉 LIMIT THE DATASET HERE
        if size is not None and size > 0:
            raw = raw[:size]


        # ---------------------------------------------------------
        # Spanish + English nq prefix support
        # ---------------------------------------------------------
        valid_prefixes = [
            "nq question:",
            "pregunta nq:",
            "pregunta nueva:",
            "nueva pregunta:",
            "nq pregunta:",
        ]

        def normalize_loc(loc: str) -> str:
            """
            Convert Spanish variants into:  "nq question:"
            """
            loc_lower = loc.lower()
            for prefix in valid_prefixes[1:]:
                if loc_lower.startswith(prefix):
                    # Replace prefix with English normalized prefix
                    return "nq question:" + loc[len(prefix):].strip()
            return loc

        data = []
        for i, record in enumerate(raw):

            loc_raw = record["loc"]
            loc_lower = loc_raw.lower()

            # Validate prefix
            if not any(prefix in loc_lower for prefix in valid_prefixes):
                raise AssertionError(
                    f"Neighborhood prompt missing valid nq prefix.\n"
                    f"Expected one of: {valid_prefixes}\n"
                    f"Found: {record['loc']}"
                )

            # Normalize Spanish → English nq format
            record["loc"] = normalize_loc(record["loc"])

            ans_toks = tok(" " + record["loc_ans"])["input_ids"]

            data.append(
                {
                    "case_id": i,
                    "requested_rewrite": {
                        "prompt": record["src"].replace(record["subject"], "{}"),
                        "subject": record["subject"],
                        "target_new": {"str": record["answers"][0]},
                        "target_true": {"str": "<|endoftext|>"},
                    },
                    "paraphrase_prompts": [record["rephrase"]],
                    "neighborhood_prompts": [
                        {
                            "prompt": record["loc"] + "?" + tok.decode(ans_toks[:j]),
                            "target": tok.decode(ans_toks[j]),
                        }
                        for j in range(len(ans_toks))
                    ],
                    "attribute_prompts": [],
                    "generation_prompts": [],
                }
            )

        self._data = data

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
