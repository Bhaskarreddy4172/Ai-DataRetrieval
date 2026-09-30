"""Dataset manager: abstract schemas, train/val/test splits, and unseen test set generation."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


class TrainingDatasetManager:
    """Manages train/val/test splits and zero-shot cross-dataset evaluation sets."""

    @staticmethod
    def create_splits(
        examples: List[Dict[str, Any]],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        total = len(examples)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))

        train_set = examples[:train_end]
        val_set = examples[train_end:val_end]
        test_set = examples[val_end:]

        return train_set, val_set, test_set

    @staticmethod
    def save_dataset_splits(
        train_set: List[Dict[str, Any]],
        val_set: List[Dict[str, Any]],
        test_set: List[Dict[str, Any]],
        output_dir: Path = Path("models/training_data")
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "train.json", "w", encoding="utf-8") as f:
            json.dump(train_set, f, indent=2)
        with open(output_dir / "validation.json", "w", encoding="utf-8") as f:
            json.dump(val_set, f, indent=2)
        with open(output_dir / "test.json", "w", encoding="utf-8") as f:
            json.dump(test_set, f, indent=2)


dataset_manager = TrainingDatasetManager()
