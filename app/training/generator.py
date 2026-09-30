"""Synthetic training instance generator based on intents, operations, schemas, and columns."""

import random
from typing import Any, Dict, List
import pandas as pd
from app.training.augmenter import linguistic_augmenter


class TrainingDataGenerator:
    """Generates synthetic questions paired with ground-truth semantic query plans."""

    INTENT_TEMPLATES = {
        "MAX": [
            "Who has the highest {metric}?",
            "Top {metric} in the company",
            "Show person with maximum {metric}",
            "Highest {metric} record",
            "Who makes the most {metric}?"
        ],
        "MIN": [
            "Who has the lowest {metric}?",
            "Bottom {metric}",
            "Show record with minimum {metric}",
            "Who has lowest {metric}?"
        ],
        "AVERAGE": [
            "What is the average {metric}?",
            "Mean {metric} across all records",
            "What is the avg {metric}?"
        ],
        "SUM": [
            "What is the total {metric}?",
            "Sum of {metric} for everyone",
            "Overall {metric} total"
        ],
        "COUNT": [
            "How many records in {dimension} {val}?",
            "Total count of {val} {dimension}",
            "Number of entries where {dimension} is {val}"
        ],
        "FILTER": [
            "Show all records where {dimension} is {val}",
            "List entries in {val}",
            "Find {dimension} matching {val}"
        ]
    }

    def generate_from_dataframe(self, df: pd.DataFrame, dataset_name: str = "dataset", max_samples: int = 200) -> List[Dict[str, Any]]:
        """Synthesize training pairs directly from an active dataset."""
        examples = []
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        num_cols = [c for c in num_cols if not c.startswith("_")]
        cat_cols = [c for c in cat_cols if not c.startswith("_")]

        for metric in num_cols:
            for op in ["MAX", "MIN", "AVERAGE", "SUM"]:
                templates = self.INTENT_TEMPLATES.get(op, [])
                for tmpl in templates:
                    base_q = tmpl.format(metric=metric)
                    variations = linguistic_augmenter.augment(base_q)
                    for v in variations:
                        examples.append({
                            "dataset": dataset_name,
                            "question": v,
                            "intent": op,
                            "target_column": metric,
                            "conditions": []
                        })

        for cat in cat_cols[:4]:
            top_vals = [str(x) for x in df[cat].dropna().unique()[:4]]
            for val in top_vals:
                for op in ["COUNT", "FILTER"]:
                    templates = self.INTENT_TEMPLATES.get(op, [])
                    for tmpl in templates:
                        base_q = tmpl.format(dimension=cat, val=val)
                        variations = linguistic_augmenter.augment(base_q)
                        for v in variations:
                            examples.append({
                                "dataset": dataset_name,
                                "question": v,
                                "intent": op,
                                "target_column": None,
                                "conditions": [{"column": cat, "operator": "=", "value": val}]
                            })

        random.shuffle(examples)
        return examples[:max_samples]


training_generator = TrainingDataGenerator()
