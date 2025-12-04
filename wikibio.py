import json
from pathlib import Path
from typing import Dict, List
#from util.globals import DATA_DIR

class WikibioDataset:
    # --- CORRECTED: Added size=None and tok=None to signature ---
    def __init__(self, data_dir: str, mode: str = 'train' , size: int = None, tok = None): 
        # Determine the file path
        file_name = f"wikibio-{mode}-all.json"
        
        # NOTE: Using DATA_DIR from util.globals, which we hardcoded previously.
        file_path = Path(data_dir) / file_name
        self.data = self._load_data(file_path)
        
        # --- ADDED: Apply size limit for compatibility with evaluate.py ---
        if size is not None:
            self.data = self.data[:size]
        # ----------------------------------------------------------------

    def _load_data(self, file_path: Path) -> List[Dict]:
        """Loads and processes the Wikibio JSON dataset for ZSRE compatibility."""
        print(f"[WIKIBIO] Loading data from {file_path}")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Wikibio data not found at: {file_path}. Check your globals.yml DATA_DIR.")
        
        processed_data = []
        for idx, item in enumerate(raw_data):
            # 1. Safely extract nested data
            locality = item.get("locality", {})
            relation_info_list = locality.get("Relation_Specificity", [])
            
            if not relation_info_list:
                continue
                
            relation_info = relation_info_list[0]
            ground_truth_list = relation_info.get("ground_truth", [])
            
            if not ground_truth_list:
                continue

            # 2. Map Wikibio fields to the required flat ZSRE format
            new_record = {
                # This is the expected format for 'ensure_zsre_record' helpers
                "src": relation_info.get("prompt", ""),
                "answers": ground_truth_list,
                "subject": item.get("concept", "ENTITY"),
                
                # Keep original fields for context/debugging (not strictly used by ROME)
                "context": item.get("text", ""),
                "target_new_labels": item.get("labels", ""),
                "case_id": idx
            }

            processed_data.append(new_record)
            
        print(f"[WIKIBIO] Processed {len(processed_data)} records.")
        return processed_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
