import json
from pathlib import Path
from typing import Dict, List

class WikibioDataset:
    def __init__(self, data_dir: Path, mode: str = "test", size: int = None, tok=None):
        data_dir = Path(data_dir)

        # If user gave a JSON file directly → load it
        if data_dir.is_file() and data_dir.suffix == ".json":
            self.file_path = data_dir
            print(f"[WIKIBIO] Using provided file: {self.file_path}")

        # Otherwise → assume folder and build a filename
        else:
            self.file_path = data_dir / f"wikibio-{mode}-all.json"
            print(f"[WIKIBIO] Using default filename: {self.file_path}")

        # Load + normalize fields
        self.data = self._load_data(self.file_path)

        # Optional dataset truncation
        if size is not None:
            self.data = self.data[:size]

    def _load_data(self, file_path: Path) -> List[Dict]:
        print(f"[WIKIBIO] Loading data from {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"Wikibio JSON not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            raw_json = json.load(f)

        processed = []
        for idx, item in enumerate(raw_json):
            # Use your native fields for editing
            rec = {
                "text": item.get("text", ""),
                "labels": item.get("labels", ""),
                "concept": item.get("concept", ""),
                "case_id": idx,
            }
            processed.append(rec)

        print(f"[WIKIBIO] Loaded {len(processed)} records.")
        return processed

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
