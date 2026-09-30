"""Automatic Test Generation System (Section 40, 41, 42).

Generates thousands of diverse, realistic, unseen query variants for any base dataset fact,
computing programmatic expected results using Pandas and DuckDB.
"""

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import duckdb
import pandas as pd

from app.dataset.alias_resolver import entity_alias_resolver, INDIAN_STATE_CODES, CITY_AIRPORT_CODES
from app.utils.fuzzy_match import LOCATION_ABBREVIATIONS



class DatasetTestGenerator:
    """Generates synthetic, highly varied test cases with verified programmatic ground truth."""

    def __init__(self, df: Optional[pd.DataFrame] = None, dataset_name: str = "dataset"):
        self.df = df
        self.dataset_name = dataset_name
        self.con = duckdb.connect(database=":memory:")
        if df is not None:
            self.con.register("dataset_table", df)

    def load_dataset(self, df: pd.DataFrame, dataset_name: str = "dataset"):
        self.df = df
        self.dataset_name = dataset_name
        self.con.register("dataset_table", df)

    def generate_large_scale_benchmark(
        self,
        total_target: int = 10000
    ) -> List[Dict[str, Any]]:
        """Generate 10,000+ test cases across all required categories programmatically."""
        if self.df is None or self.df.empty:
            raise ValueError("Dataset not loaded.")
        df = self.df

        suite: List[Dict[str, Any]] = []
        counter = 1

        states = df["state"].tolist()
        capitals = df["capital"].tolist()
        pairs = list(zip(states, capitals))

        cases_per_cat = total_target // 10

        # Category 1: Standard Normal Questions (1,000)
        cat1_templates = [
            "What is the capital of {state}?",
            "What is {state}'s capital?",
            "Tell me the capital of {state}.",
            "Which city is the capital of {state}?",
            "Find the capital city of {state}.",
            "{state} capital?",
            "What is the capital city associated with {state}?",
            "Show me the capital of {state}.",
            "Capital of {state} please.",
            "Can you tell me {state}'s capital?"
        ]
        for i in range(cases_per_cat):
            st, cap = pairs[i % len(pairs)]
            tmpl = cat1_templates[i % len(cat1_templates)]
            q = tmpl.format(state=st)
            suite.append({
                "id": f"TEST_{counter:05d}",
                "category": "1_standard_normal",
                "question": q,
                "expected_intent": "LOOKUP",
                "expected_entity": st,
                "expected_column": "capital",
                "expected_answer": cap,
                "source_type": "DATASET"
            })
            counter += 1

        # Category 2: Informal / Slang Questions (1,000)
        cat2_templates = [
            "yo what is {state} capital",
            "tell me {state} capital bro",
            "capital {state}?",
            "hey {state} capital please",
            "{state} capital dude",
            "whats {state} capital bro",
            "capital of {state} yo",
            "gimme {state} capital",
            "tell {state} capital now",
            "{state} capital fast"
        ]
        for i in range(cases_per_cat):
            st, cap = pairs[i % len(pairs)]
            tmpl = cat2_templates[i % len(cat2_templates)]
            q = tmpl.format(state=st)
            suite.append({
                "id": f"TEST_{counter:05d}",
                "category": "2_informal_slang",
                "question": q,
                "expected_intent": "LOOKUP",
                "expected_entity": st,
                "expected_column": "capital",
                "expected_answer": cap,
                "source_type": "DATASET"
            })
            counter += 1

        # Category 3: Typo / Spelling Variation Questions (1,000)
        typo_pairs = [
            ("Telangana", "telengana"), ("Gujarat", "gujrat"), ("Bihar", "bihaar"),
            ("Sikkim", "sikim"), ("Sikkim", "sikkimm"), ("Chhattisgarh", "chattisgarh"),
            ("Karnataka", "karnatak"), ("West Bengal", "westbengal"), ("West Bengal", "west bengol"),
            ("Andhra Pradesh", "andhrapradesh"), ("Andhra Pradesh", "andra pradesh"),
            ("Himachal Pradesh", "himachalpradesh"), ("Tamil Nadu", "tamilnadu"),
            ("Arunachal Pradesh", "arunachalpradesh"), ("Madhya Pradesh", "madhyapradesh"),
            ("Uttar Pradesh", "uttarpradesh"), ("Uttarakhand", "uttrakhand"),
            ("Maharashtra", "maharastra"), ("Odisha", "odisa"), ("Punjab", "punjaab")
        ]
        cat3_templates = [
            "{typo} capital?",
            "capital of {typo}?",
            "what is {typo}'s capital?",
            "{typo} captal?",
            "wat is {typo} capital?",
            "tell me {typo} capital",
            "what is the captal of {typo}?",
            "{typo} capital city?",
            "which city is {typo} capital?",
            "find capital of {typo}"
        ]
        for i in range(cases_per_cat):
            st, typo_st = typo_pairs[i % len(typo_pairs)]
            cap_vals = df.loc[df["state"] == st, "capital"].tolist()
            cap = str(cap_vals[0]) if cap_vals else ""
            tmpl = cat3_templates[i % len(cat3_templates)]
            q = tmpl.format(typo=typo_st)
            suite.append({
                "id": f"TEST_{counter:05d}",
                "category": "3_typo_variations",
                "question": q,
                "expected_intent": "LOOKUP",
                "expected_entity": st,
                "expected_column": "capital",
                "expected_answer": cap,
                "source_type": "DATASET"
            })
            counter += 1

        # Category 4: Phonetic / Sound-alike Questions (1,000)
        phonetic_pairs = [
            ("Sikkim", "cikkim"), ("Telangana", "telengana"), ("Gujarat", "gujraat"),
            ("Bihar", "beehar"), ("Kerala", "keralam"), ("Assam", "asom"),
            ("Goa", "goaa"), ("Punjab", "panjab"), ("Odisha", "orisa"),
            ("Dehradun", "dehradoon"), ("Bengaluru", "bangalore"), ("Kolkata", "calcutta")
        ]
        cat4_templates = [
            "capital of {phonetic}?",
            "{phonetic} capital?",
            "what is {phonetic}'s capital?",
            "which state has {phonetic}?",
            "state of {phonetic}?",
            "where is {phonetic} situated?",
            "which state is associated with {phonetic}?"
        ]
        for i in range(cases_per_cat):
            entity, phon = phonetic_pairs[i % len(phonetic_pairs)]
            is_state = entity in states
            target_col = "capital" if is_state else "state"
            if is_state:
                ans_vals = df.loc[df["state"] == entity, "capital"].tolist()
            else:
                ans_vals = df.loc[df["capital"] == entity, "state"].tolist()
            expected_ans = str(ans_vals[0]) if ans_vals else ""
            tmpl = cat4_templates[i % len(cat4_templates)]
            q = tmpl.format(phonetic=phon)
            suite.append({
                "id": f"TEST_{counter:05d}",
                "category": "4_phonetic_soundalike",
                "question": q,
                "expected_intent": "LOOKUP",
                "expected_entity": entity,
                "expected_column": target_col,
                "expected_answer": expected_ans,
                "source_type": "DATASET"
            })
            counter += 1

        # Category 5: Abbreviation & Shortcut Questions (1,000)
        abbr_list = [
            ("Telangana", "TS"), ("Telangana", "TG"), ("Andhra Pradesh", "AP"),
            ("West Bengal", "WB"), ("Karnataka", "KA"), ("Tamil Nadu", "TN"),
            ("Maharashtra", "MH"), ("Gujarat", "GJ"), ("Rajasthan", "RJ"),
            ("Punjab", "PB"), ("Uttar Pradesh", "UP"), ("Himachal Pradesh", "HP"),
            ("Madhya Pradesh", "MP"), ("Bihar", "BR"), ("Goa", "GA"),
            ("Kerala", "KL"), ("Odisha", "OD"), ("Sikkim", "SK"),
            ("Hyderabad", "HYD"), ("Bengaluru", "BLR"), ("Kolkata", "CCU"),
            ("Mumbai", "BOM"), ("Chennai", "MAA"), ("Jaipur", "JAI")
        ]
        cat5_templates_state = [
            "capital of {code}?",
            "{code} capital?",
            "{code} ka capital?",
            "what is {code}'s capital?",
            "which city is {code} capital?",
            "tell me {code} capital"
        ]
        cat5_templates_city = [
            "which state has {code}?",
            "{code} which state comes?",
            "{code} is the capital of which state?",
            "which state is {code} in?",
            "which state has {code} as capital?"
        ]
        for i in range(cases_per_cat):
            entity, code = abbr_list[i % len(abbr_list)]
            is_state = entity in states
            target_col = "capital" if is_state else "state"
            if is_state:
                ans_vals = df.loc[df["state"] == entity, "capital"].tolist()
            else:
                ans_vals = df.loc[df["capital"] == entity, "state"].tolist()
            expected_ans = str(ans_vals[0]) if ans_vals else ""
            tmpl_list = cat5_templates_state if is_state else cat5_templates_city
            tmpl = tmpl_list[i % len(tmpl_list)]
            q = tmpl.format(code=code)
            suite.append({
                "id": f"TEST_{counter:05d}",
                "category": "5_abbreviation_shortcuts",
                "question": q,
                "expected_intent": "LOOKUP",
                "expected_entity": entity,
                "expected_column": target_col,
                "expected_answer": expected_ans,
                "source_type": "DATASET"
            })
            counter += 1

        # Category 6: Reverse Lookup / Relationship Questions (1,000)
        cat6_templates = [
            "Which state has {capital} as capital?",
            "{capital} is the capital of which state?",
            "Which state has {capital}?",
            "Which state is {capital} in?",
            "{capital} which state comes?",
            "Which state is associated with {capital}?",
            "{capital} belongs to which state?",
            "What state is {capital} in?",
            "Tell me the state having {capital} as capital",
            "In which state is {capital} located?"
        ]
        for i in range(cases_per_cat):
            st, cap = pairs[i % len(pairs)]
            tmpl = cat6_templates[i % len(cat6_templates)]
            q = tmpl.format(capital=cap)
            suite.append({
                "id": f"TEST_{counter:05d}",
                "category": "6_reverse_relationships",
                "question": q,
                "expected_intent": "LOOKUP",
                "expected_entity": cap,
                "expected_column": "state",
                "expected_answer": st,
                "source_type": "DATASET"
            })
            counter += 1

        # Category 7: Hinglish / Multilingual Questions (1,000)
        cat7_templates = [
            "{state} ka capital?",
            "{state} ka capital kya hai?",
            "{state} ka capital batao",
            "{state} ki capital kya hai?",
            "{state} ke capital bataiye",
            "{state} capital kya hai bhai",
            "{state} ka capital batao zara",
            "mujhe {state} ka capital chahiye",
            "{state} ka rajdhani kya hai?",
            "{state} ki rajdhani batao"
        ]
        for i in range(cases_per_cat):
            st, cap = pairs[i % len(pairs)]
            tmpl = cat7_templates[i % len(cat7_templates)]
            q = tmpl.format(state=st)
            suite.append({
                "id": f"TEST_{counter:05d}",
                "category": "7_hinglish_colloquial",
                "question": q,
                "expected_intent": "LOOKUP",
                "expected_entity": st,
                "expected_column": "capital",
                "expected_answer": cap,
                "source_type": "DATASET"
            })
            counter += 1

        # Category 8: Boolean / Fact-Checking Questions (1,000)
        for i in range(cases_per_cat):
            st, cap = pairs[i % len(pairs)]
            is_true = (i % 2 == 0)
            if is_true:
                q = f"Is {cap} the capital of {st}?"
                exp_ans = "True"
            else:
                wrong_cap = capitals[(i + 1) % len(capitals)]
                q = f"Is {wrong_cap} the capital of {st}?"
                exp_ans = "False"
            suite.append({
                "id": f"TEST_{counter:05d}",
                "category": "8_boolean_fact_checking",
                "question": q,
                "expected_intent": "BOOLEAN_CHECK",
                "expected_entity": st,
                "expected_column": "capital",
                "expected_answer": exp_ans,
                "source_type": "DATASET"
            })
            counter += 1

        # Category 9: Analytical / Aggregations (1,000)
        cat9_templates = [
            ("How many states are in the dataset?", "COUNT", len(states), "count"),
            ("Count the total states", "COUNT", len(states), "count"),
            ("Total count of states in data", "COUNT", len(states), "count"),
            ("How many records are in this dataset?", "COUNT", len(states), "count"),
            ("List all distinct states", "DISTINCT", len(states), "distinct"),
            ("Distinct capitals in the dataset", "DISTINCT", len(set(capitals)), "distinct")
        ]
        for i in range(cases_per_cat):
            tmpl_q, op, exp_val, metric_type = cat9_templates[i % len(cat9_templates)]
            suite.append({
                "id": f"TEST_{counter:05d}",
                "category": "9_analytical_aggregations",
                "question": tmpl_q,
                "expected_intent": op,
                "expected_entity": None,
                "expected_column": "state",
                "expected_answer": str(exp_val),
                "source_type": "DATASET"
            })
            counter += 1

        # Category 10: Anti-Hallucination & Unsupported Guard (1,000)
        cat10_cases = [
            ("What is the population of Telangana?", "UNSUPPORTED_QUERY", "population", "COLUMN_NOT_FOUND"),
            ("Who is the chief minister of West Bengal?", "UNSUPPORTED_QUERY", "chief minister", "COLUMN_NOT_FOUND"),
            ("What is the GDP of Maharashtra?", "UNSUPPORTED_QUERY", "gdp", "COLUMN_NOT_FOUND"),
            ("What is the literacy rate of Kerala?", "UNSUPPORTED_QUERY", "literacy rate", "COLUMN_NOT_FOUND"),
            ("What is the capital of Wakanda?", "NO_MATCH", "Wakanda", "ENTITY_NOT_FOUND"),
            ("What is the capital of France?", "NO_MATCH", "France", "ENTITY_NOT_FOUND"),
            ("Tell me the capital of Narnia", "NO_MATCH", "Narnia", "ENTITY_NOT_FOUND"),
            ("What is the crime rate in Delhi?", "UNSUPPORTED_QUERY", "crime rate", "COLUMN_NOT_FOUND"),
            ("What is the temperature in Hyderabad?", "UNSUPPORTED_QUERY", "temperature", "COLUMN_NOT_FOUND"),
            ("Who has the highest salary?", "UNSUPPORTED_QUERY", "salary", "COLUMN_NOT_FOUND")
        ]
        for i in range(cases_per_cat):
            q, op, missing, nm_type = cat10_cases[i % len(cat10_cases)]
            suite.append({
                "id": f"TEST_{counter:05d}",
                "category": "10_unsupported_antihallucination",
                "question": q,
                "expected_intent": op,
                "expected_entity": missing if nm_type == "ENTITY_NOT_FOUND" else None,
                "expected_column": missing if nm_type == "COLUMN_NOT_FOUND" else None,
                "expected_answer": f"The uploaded dataset does not contain {missing}" if nm_type == "COLUMN_NOT_FOUND" else f'No matching state or entity was found in the uploaded dataset for "{missing}"',
                "source_type": "DATASET",
                "no_match_type": nm_type
            })
            counter += 1

        return suite


dataset_test_generator = DatasetTestGenerator()


if __name__ == "__main__":
    csv_path = Path(__file__).resolve().parent.parent / "data" / "indian_states_capitals.csv"
    if csv_path.exists():
        df_states = pd.read_csv(csv_path)
        generator = DatasetTestGenerator(df_states, "indian_states_capitals.csv")

        # 1. Generate 1,000 golden suite
        golden_suite = generator.generate_large_scale_benchmark(total_target=1000)
        golden_path = Path(__file__).resolve().parent / "golden_indian_states_suite.json"
        with open(golden_path, "w", encoding="utf-8") as f:
            json.dump(golden_suite, f, indent=2)
        print(f"Generated {len(golden_suite)} test cases at {golden_path}")

        # 2. Generate 10,000 benchmark suite
        benchmark_10k = generator.generate_large_scale_benchmark(total_target=10000)
        benchmark_dir = Path(__file__).resolve().parent.parent / "benchmark"
        benchmark_dir.mkdir(exist_ok=True)
        benchmark_path = benchmark_dir / "synthetic_10k_suite.json"
        with open(benchmark_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_10k, f, indent=2)
        print(f"Generated {len(benchmark_10k)} test cases at {benchmark_path}")
    else:
        print(f"Dataset not found at {csv_path}")
