import json
from dataclasses import dataclass
from pathlib import Path  # <--- ADD THIS LINE

@dataclass
class HyperParams:
    """
    Simple wrapper to store hyperparameters for Python-based rewriting methods.
    """

    @classmethod
    def from_json(cls, path: Path):
        with open(path, "r") as f:
            data = json.load(f)
        
        # --- FIX: Remove all known conflicting metadata keys ---
        # These keys are run parameters handled by evaluate.py, not ROMEHyperParams.
        for key in ['alg_name', 'model_name', 'device', 'results_dir', 'seed', 'archive', 'n_edits', 'model_parallel', 'v_num']:
            if key in data:
                data.pop(key)
        # -----------------------------------------------------

        return cls(**data)
