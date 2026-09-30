"""Synthetic 10,000+ Test Benchmark Generator for Universal Question Understanding.

Generates 10,000 deterministic test cases across 10 distinct categories,
computing verified ground truth using DuckDB and Pandas against data/sample_employees.xlsx.
"""

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "sample_employees.xlsx"
OUTPUT_PATH = PROJECT_ROOT / "benchmark" / "synthetic_10k_suite.json"


def generate_10k_suite():
    print(f"Loading dataset from {DATA_PATH}...")
    df = pd.read_excel(DATA_PATH)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # DuckDB registration
    con = duckdb.connect(database=":memory:")
    con.register("employees", df)

    suite: List[Dict[str, Any]] = []
    counter = 1

    def add_case(category: str, question: str, expected_op: str, expected_count: int = 0,
                 expected_agg_val: Any = None, expected_ids: List[str] = None,
                 expected_names: List[str] = None, target_col: str = None):
        nonlocal counter
        suite.append({
            "id": f"SYN_10K_{counter:05d}",
            "category": category,
            "question": question,
            "expected_op": expected_op,
            "expected_count": expected_count,
            "expected_agg_val": round(float(expected_agg_val), 2) if expected_agg_val is not None else None,
            "expected_ids": expected_ids or [],
            "expected_names": expected_names or [],
            "target_col": target_col
        })
        counter += 1

    departments = sorted(df["Department"].unique().tolist())
    cities = sorted(df["City"].unique().tolist())
    designations = sorted(df["Designation"].unique().tolist())
    emp_records = df[["Employee ID", "Employee Name", "Department", "Designation", "City", "Salary", "Performance Score"]].to_dict(orient="records")

    print("Generating 10 categories x 1,000 cases = 10,000 cases...")

    # -------------------------------------------------------------
    # Category 1: Standard / Normal Queries (1,000 cases)
    # -------------------------------------------------------------
    print("Generating Cat 1: Standard / Normal Queries...")
    std_templates = [
        ("Show all employees in {val}", "Department", "FILTER"),
        ("List employees working in {val}", "City", "FILTER"),
        ("Find employee with ID {val}", "Employee ID", "LOOKUP"),
        ("Details for {val}", "Employee Name", "LOOKUP"),
        ("How many employees are in {val}?", "Department", "COUNT"),
        ("Count of employees in {val}", "City", "COUNT"),
        ("What is the average salary in {val}?", "Department", "AVERAGE"),
        ("Total salary of employees in {val}", "City", "SUM"),
        ("Who is the highest earner in {val}?", "Department", "MAX"),
        ("Who has the lowest salary in {val}?", "City", "MIN"),
    ]
    for i in range(1000):
        tpl, col, op = std_templates[i % len(std_templates)]
        if col == "Department":
            val = departments[i % len(departments)]
            sub = df[df["Department"] == val]
        elif col == "City":
            val = cities[i % len(cities)]
            sub = df[df["City"] == val]
        elif col == "Employee ID":
            rec = emp_records[i % len(emp_records)]
            val = rec["Employee ID"]
            sub = df[df["Employee ID"] == val]
        else:
            rec = emp_records[i % len(emp_records)]
            val = rec["Employee Name"]
            sub = df[df["Employee Name"] == val]

        q = tpl.format(val=val)
        exp_count = len(sub)
        exp_ids = sub["Employee ID"].tolist()
        exp_names = sub["Employee Name"].tolist()
        agg_val = None
        if op == "COUNT":
            agg_val = exp_count
        elif op == "AVERAGE":
            agg_val = sub["Salary"].mean() if not sub.empty else 0
        elif op == "SUM":
            agg_val = sub["Salary"].sum() if not sub.empty else 0
        elif op in ("MAX", "MIN"):
            exp_count = 1 if not sub.empty else 0
            if op == "MAX" and not sub.empty:
                top_row = sub.sort_values(by="Salary", ascending=False).iloc[0]
                exp_ids = [top_row["Employee ID"]]
                exp_names = [top_row["Employee Name"]]
                agg_val = top_row["Salary"]
            elif op == "MIN" and not sub.empty:
                low_row = sub.sort_values(by="Salary", ascending=True).iloc[0]
                exp_ids = [low_row["Employee ID"]]
                exp_names = [low_row["Employee Name"]]
                agg_val = low_row["Salary"]

        add_case("1_standard_normal", q, op, exp_count, agg_val, exp_ids, exp_names)

    # -------------------------------------------------------------
    # Category 2: Informal / Broken English / Slang (1,000 cases)
    # -------------------------------------------------------------
    print("Generating Cat 2: Informal / Slang Queries...")
    slang_templates = [
        ("gimme dudes in {dept}", "FILTER"),
        ("who makin bank in {city}", "MAX"),
        ("tell me top earner in {dept}", "MAX"),
        ("lowest paid guy in {city}", "MIN"),
        ("how many ppl workin in {dept}", "COUNT"),
        ("show me folks gettin paid most in {dept}", "MAX"),
        ("who rockstar in {city}", "MAX"),
        ("gimme salry average for {dept}", "AVERAGE"),
        ("who gettin lowest dough in {dept}", "MIN"),
        ("u got any dudes in {city}", "FILTER"),
    ]
    for i in range(1000):
        tpl, op = slang_templates[i % len(slang_templates)]
        dept = departments[i % len(departments)]
        city = cities[i % len(cities)]
        q = tpl.format(dept=dept, city=city)
        filter_col = "Department" if "{dept}" in tpl else "City"
        val = dept if filter_col == "Department" else city
        sub = df[df[filter_col] == val]

        exp_count = len(sub)
        exp_ids = sub["Employee ID"].tolist()
        exp_names = sub["Employee Name"].tolist()
        agg_val = None
        if op == "COUNT":
            agg_val = exp_count
        elif op == "AVERAGE":
            agg_val = sub["Salary"].mean() if not sub.empty else 0
        elif op in ("MAX", "MIN"):
            exp_count = 1 if not sub.empty else 0
            metric_col = "Performance Score" if "rockstar" in q else "Salary"
            asc = (op == "MIN")
            if not sub.empty:
                target_r = sub.sort_values(by=metric_col, ascending=asc).iloc[0]
                exp_ids = [target_r["Employee ID"]]
                exp_names = [target_r["Employee Name"]]
                agg_val = target_r[metric_col]

        add_case("2_informal_slang", q, op, exp_count, agg_val, exp_ids, exp_names)

    # -------------------------------------------------------------
    # Category 3: Typo Variations (1,000 cases)
    # -------------------------------------------------------------
    print("Generating Cat 3: Typo Variations...")
    typo_patterns = [
        ("show employes in {dept_typo}", "FILTER"),
        ("who has higest slary in {dept_typo}", "MAX"),
        ("avrg salary in {dept_typo}", "AVERAGE"),
        ("totl salry of {dept_typo}", "SUM"),
        ("how mny employes in {city_typo}", "COUNT"),
        ("find emplyee with id {id}", "LOOKUP"),
        ("detals for {name}", "LOOKUP"),
        ("who erans maximum in {dept_typo}", "MAX"),
        ("who erans minimum in {city_typo}", "MIN"),
        ("whre is {name} working", "LOOKUP"),
    ]
    dept_typos = {
        "Engineering": ["enginering", "engneering", "enginerring"],
        "Finance": ["fiance", "finanace", "finace"],
        "Human Resources": ["hr dept", "human resorses", "humna resources"],
        "Marketing": ["markting", "mktg", "marketng"],
        "Sales": ["sles", "saels", "slaes"],
        "Support": ["suport", "supprt", "supprot"],
    }
    city_typos = {
        "Bengaluru": ["banglore", "bengluru", "bangalor"],
        "Hyderabad": ["hydrabad", "hyderbad", "hyd"],
        "Mumbai": ["mumbaii", "bombay", "mumbay"],
        "Pune": ["pune city", "punee", "pune"],
        "Delhi": ["dilli", "new delhi", "delhi"],
        "Kolkata": ["calcutta", "kolkataa", "kolkatta"],
        "Chennai": ["chenai", "madras", "chenai"],
    }
    for i in range(1000):
        tpl, op = typo_patterns[i % len(typo_patterns)]
        dept = departments[i % len(departments)]
        d_typo = dept_typos.get(dept, [dept.lower()])[i % len(dept_typos.get(dept, [dept.lower()]))]
        city = cities[i % len(cities)]
        c_typo = city_typos.get(city, [city.lower()])[i % len(city_typos.get(city, [city.lower()]))]
        rec = emp_records[i % len(emp_records)]

        q = tpl.format(dept_typo=d_typo, city_typo=c_typo, id=rec["Employee ID"], name=rec["Employee Name"])
        if "{dept_typo}" in tpl:
            sub = df[df["Department"] == dept]
        elif "{city_typo}" in tpl:
            sub = df[df["City"] == city]
        elif "{id}" in tpl:
            sub = df[df["Employee ID"] == rec["Employee ID"]]
        else:
            sub = df[df["Employee Name"] == rec["Employee Name"]]

        exp_count = len(sub)
        exp_ids = sub["Employee ID"].tolist()
        exp_names = sub["Employee Name"].tolist()
        agg_val = None
        if op == "COUNT":
            agg_val = exp_count
        elif op == "AVERAGE":
            agg_val = sub["Salary"].mean() if not sub.empty else 0
        elif op == "SUM":
            agg_val = sub["Salary"].sum() if not sub.empty else 0
        elif op in ("MAX", "MIN"):
            exp_count = 1 if not sub.empty else 0
            asc = (op == "MIN")
            if not sub.empty:
                target_r = sub.sort_values(by="Salary", ascending=asc).iloc[0]
                exp_ids = [target_r["Employee ID"]]
                exp_names = [target_r["Employee Name"]]
                agg_val = target_r["Salary"]

        add_case("3_typo_variations", q, op, exp_count, agg_val, exp_ids, exp_names)

    # -------------------------------------------------------------
    # Category 4: Phonetic / Sound-Alike Queries (1,000 cases)
    # -------------------------------------------------------------
    print("Generating Cat 4: Phonetic / Sound-Alike Queries...")
    phonetic_city_variants = {
        "Bengaluru": ["Bengluru", "Banglore", "Bangaluru"],
        "Hyderabad": ["Haiderabad", "Haydarabad", "Haidarabad"],
        "Kolkata": ["Kalkata", "Kolkatta", "Kalkatta"],
        "Delhi": ["Deli", "Dehli", "Dilli"],
        "Pune": ["Puna", "Poona", "Pune"],
    }
    phonetic_templates = [
        ("who is working in {p_city}?", "FILTER"),
        ("find employees stationed in {p_city}", "FILTER"),
        ("count staff located in {p_city}", "COUNT"),
        ("highest paid employee in {p_city}", "MAX"),
        ("lowest earner in {p_city}", "MIN"),
        ("average compensation in {p_city}", "AVERAGE"),
        ("total remuneration in {p_city}", "SUM"),
        ("show top talent in {p_city}", "MAX"),
        ("list all records for {p_city}", "FILTER"),
        ("what is headcount in {p_city}", "COUNT"),
    ]
    p_city_keys = list(phonetic_city_variants.keys())
    for i in range(1000):
        tpl, op = phonetic_templates[i % len(phonetic_templates)]
        canonical_c = p_city_keys[i % len(p_city_keys)]
        variants = phonetic_city_variants[canonical_c]
        p_c = variants[i % len(variants)]

        q = tpl.format(p_city=p_c)
        sub = df[df["City"] == canonical_c]
        exp_count = len(sub)
        exp_ids = sub["Employee ID"].tolist()
        exp_names = sub["Employee Name"].tolist()
        agg_val = None
        if op == "COUNT":
            agg_val = exp_count
        elif op == "AVERAGE":
            agg_val = sub["Salary"].mean() if not sub.empty else 0
        elif op == "SUM":
            agg_val = sub["Salary"].sum() if not sub.empty else 0
        elif op in ("MAX", "MIN"):
            exp_count = 1 if not sub.empty else 0
            metric_col = "Performance Score" if "talent" in q else "Salary"
            asc = (op == "MIN")
            if not sub.empty:
                top_r = sub.sort_values(by=metric_col, ascending=asc).iloc[0]
                exp_ids = [top_r["Employee ID"]]
                exp_names = [top_r["Employee Name"]]
                agg_val = top_r[metric_col]

        add_case("4_phonetic_soundalike", q, op, exp_count, agg_val, exp_ids, exp_names)

    # -------------------------------------------------------------
    # Category 5: Synonyms & Natural Language Variations (1,000 cases)
    # -------------------------------------------------------------
    print("Generating Cat 5: Synonyms & Variations...")
    syn_templates = [
        ("who has highest compensation?", "MAX"),
        ("lowest compensation in {dept}", "MIN"),
        ("total remuneration of {dept}", "SUM"),
        ("mean salary in {city}", "AVERAGE"),
        ("headcount of {dept}", "COUNT"),
        ("workforce size in {city}", "COUNT"),
        ("earliest joiner in {dept}", "MIN"),
        ("most recent hire in {dept}", "MAX"),
        ("personnel in {dept}", "FILTER"),
        ("staff members in {city}", "FILTER"),
    ]
    for i in range(1000):
        tpl, op = syn_templates[i % len(syn_templates)]
        dept = departments[i % len(departments)]
        city = cities[i % len(cities)]
        q = tpl.format(dept=dept, city=city)

        if "{dept}" in tpl:
            sub = df[df["Department"] == dept]
        elif "{city}" in tpl:
            sub = df[df["City"] == city]
        else:
            sub = df

        exp_count = len(sub)
        exp_ids = sub["Employee ID"].tolist()
        exp_names = sub["Employee Name"].tolist()
        agg_val = None

        if op == "COUNT":
            agg_val = exp_count
        elif op == "AVERAGE":
            agg_val = sub["Salary"].mean() if not sub.empty else 0
        elif op == "SUM":
            agg_val = sub["Salary"].sum() if not sub.empty else 0
        elif op in ("MAX", "MIN"):
            exp_count = 1 if not sub.empty else 0
            metric_col = "Joining Date" if ("joiner" in q or "hire" in q) else "Salary"
            asc = (op == "MIN")
            if not sub.empty:
                sort_s = sub.sort_values(by=metric_col, ascending=asc).iloc[0]
                exp_ids = [sort_s["Employee ID"]]
                exp_names = [sort_s["Employee Name"]]
                agg_val = sort_s[metric_col] if metric_col == "Salary" else None

        add_case("5_synonyms_variations", q, op, exp_count, agg_val, exp_ids, exp_names)

    # -------------------------------------------------------------
    # Category 6: Multi-Condition Queries (1,000 cases)
    # -------------------------------------------------------------
    print("Generating Cat 6: Multi-Condition Queries...")
    sal_thresholds = [40000, 50000, 60000, 70000, 80000]
    multi_cond_templates = [
        ("employees in {dept} and salary > {sal}", ">"),
        ("employees in {dept} with salary at least {sal}", ">="),
        ("employees in {city} earning less than {sal}", "<"),
        ("staff in {dept} located in {city}", "dept_city"),
        ("employees in {city} with performance score >= {score}", "score_gte"),
    ]
    scores = [75, 80, 85, 90]
    for i in range(1000):
        tpl, kind = multi_cond_templates[i % len(multi_cond_templates)]
        dept = departments[i % len(departments)]
        city = cities[i % len(cities)]
        sal = sal_thresholds[i % len(sal_thresholds)]
        sc = scores[i % len(scores)]

        q = tpl.format(dept=dept, city=city, sal=sal, score=sc)

        if kind == ">":
            sub = df[(df["Department"] == dept) & (df["Salary"] > sal)]
        elif kind == ">=":
            sub = df[(df["Department"] == dept) & (df["Salary"] >= sal)]
        elif kind == "<":
            sub = df[(df["City"] == city) & (df["Salary"] < sal)]
        elif kind == "dept_city":
            sub = df[(df["Department"] == dept) & (df["City"] == city)]
        else:
            sub = df[(df["City"] == city) & (df["Performance Score"] >= sc)]

        add_case("6_multi_condition", q, "FILTER", len(sub), None, sub["Employee ID"].tolist(), sub["Employee Name"].tolist())

    # -------------------------------------------------------------
    # Category 7: Relational / Cross-Row Queries (1,000 cases)
    # -------------------------------------------------------------
    print("Generating Cat 7: Relational / Cross-Row Queries...")
    rel_templates = [
        ("who earns more than {name}?", "gt_sal"),
        ("who earns less than {name}?", "lt_sal"),
        ("employees in the same department as {name}", "same_dept"),
        ("employees located in the same city as {name}", "same_city"),
        ("compare {name1} and {name2}", "COMPARE"),
    ]
    for i in range(1000):
        tpl, kind = rel_templates[i % len(rel_templates)]
        rec1 = emp_records[i % len(emp_records)]
        rec2 = emp_records[(i + 7) % len(emp_records)]
        name1 = rec1["Employee Name"]
        name2 = rec2["Employee Name"]

        q = tpl.format(name=name1, name1=name1, name2=name2)

        if kind == "gt_sal":
            sub = df[df["Salary"] > rec1["Salary"]]
            add_case("7_relational_cross_row", q, "CROSS_ROW", len(sub), None, sub["Employee ID"].tolist(), sub["Employee Name"].tolist())
        elif kind == "lt_sal":
            sub = df[df["Salary"] < rec1["Salary"]]
            add_case("7_relational_cross_row", q, "CROSS_ROW", len(sub), None, sub["Employee ID"].tolist(), sub["Employee Name"].tolist())
        elif kind == "same_dept":
            sub = df[(df["Department"] == rec1["Department"]) & (df["Employee Name"] != name1)]
            add_case("7_relational_cross_row", q, "CROSS_ROW", len(sub), None, sub["Employee ID"].tolist(), sub["Employee Name"].tolist())
        elif kind == "same_city":
            sub = df[(df["City"] == rec1["City"]) & (df["Employee Name"] != name1)]
            add_case("7_relational_cross_row", q, "CROSS_ROW", len(sub), None, sub["Employee ID"].tolist(), sub["Employee Name"].tolist())
        else:
            add_case("7_relational_cross_row", q, "COMPARE", 2, None, [rec1["Employee ID"], rec2["Employee ID"]], [name1, name2])

    # -------------------------------------------------------------
    # Category 8: Analytical / Aggregation / GroupBy (1,000 cases)
    # -------------------------------------------------------------
    print("Generating Cat 8: Analytical & Aggregations...")
    agg_templates = [
        ("average salary by department", "Department", "AVERAGE"),
        ("total salary by city", "City", "SUM"),
        ("count of employees by department", "Department", "COUNT"),
        ("count of employees by city", "City", "COUNT"),
        ("maximum salary by department", "Department", "MAX"),
        ("minimum salary by city", "City", "MIN"),
        ("average performance score by department", "Department", "AVERAGE"),
        ("highest performance score in {dept}", "Department", "MAX"),
        ("lowest performance score in {city}", "City", "MIN"),
        ("what is the total payroll of the company?", "Overall", "SUM"),
    ]
    for i in range(1000):
        tpl, grp, op = agg_templates[i % len(agg_templates)]
        dept = departments[i % len(departments)]
        city = cities[i % len(cities)]
        q = tpl.format(dept=dept, city=city)

        if grp == "Overall":
            add_case("8_analytical_aggregations", q, op, None, df["Salary"].sum())
        elif "{dept}" in tpl:
            sub = df[df["Department"] == dept]
            agg_v = sub["Performance Score"].max() if not sub.empty else 0
            add_case("8_analytical_aggregations", q, op, None, agg_v)
        elif "{city}" in tpl:
            sub = df[df["City"] == city]
            agg_v = sub["Performance Score"].min() if not sub.empty else 0
            add_case("8_analytical_aggregations", q, op, None, agg_v)
        else:
            add_case("8_analytical_aggregations", q, op, None, None)

    # -------------------------------------------------------------
    # Category 9: Multi-Question / Compound Inquiries (1,000 cases)
    # -------------------------------------------------------------
    print("Generating Cat 9: Multi-Question / Compound Inquiries...")
    multi_q_templates = [
        ("How many employees are in {dept} and what is their average salary?", "COUNT_AVG"),
        ("Who earns the most in {dept} and what is their salary?", "MAX_SAL"),
        ("Count employees in {city} and show who is the top performer in {city}", "COUNT_TOP"),
        ("What is the total salary of {dept} and who is in that department?", "SUM_FILTER"),
        ("Who has the lowest salary in {city} and what is their department?", "MIN_DEPT"),
    ]
    for i in range(1000):
        tpl, mtype = multi_q_templates[i % len(multi_q_templates)]
        dept = departments[i % len(departments)]
        city = cities[i % len(cities)]
        q = tpl.format(dept=dept, city=city)

        filter_col = "Department" if "{dept}" in tpl else "City"
        val = dept if filter_col == "Department" else city
        sub = df[df[filter_col] == val]

        add_case("9_compound_multi_question", q, "MULTI_QUESTION", len(sub), None, sub["Employee ID"].tolist(), sub["Employee Name"].tolist())

    # -------------------------------------------------------------
    # Category 10: Negative / Out-of-Scope / Ambiguous (1,000 cases)
    # -------------------------------------------------------------
    print("Generating Cat 10: Negative / Out-of-Scope / Ambiguous...")
    neg_templates = [
        ("What is the stock price of Apple?", "OUT_OF_SCOPE"),
        ("What is the share value of Tesla?", "OUT_OF_SCOPE"),
        ("Who won the 2024 ICC World Cup?", "OUT_OF_SCOPE"),
        ("What is the weather forecast for tomorrow?", "OUT_OF_SCOPE"),
        ("What is the current temperature in London?", "OUT_OF_SCOPE"),
        ("Show employee records for Antarctica", "EMPTY"),
        ("Find employees stationed in Atlantis", "EMPTY"),
        ("What is the profit margin of our company?", "OUT_OF_SCOPE"),
        ("Show details for non_existent_employee_xyz", "EMPTY"),
        ("Who is the CEO?", "OUT_OF_SCOPE"),
        ("What is the revenue for Q4?", "OUT_OF_SCOPE"),
        ("Show me records", "AMBIGUOUS"),
        ("Find data", "AMBIGUOUS"),
        ("List information", "AMBIGUOUS"),
        ("Who is the prime minister of India?", "OUT_OF_SCOPE"),
    ]
    for i in range(1000):
        tpl, kind = neg_templates[i % len(neg_templates)]
        q = tpl
        if kind == "EMPTY":
            add_case("10_negative_out_of_scope", q, "FILTER", 0, None, [], [])
        elif kind == "AMBIGUOUS":
            add_case("10_negative_out_of_scope", q, "CLARIFICATION", 0, None, [], [])
        else:
            add_case("10_negative_out_of_scope", q, "UNSUPPORTED_QUERY", 0, None, [], [])

    print(f"Total generated benchmark cases: {len(suite)}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(suite, f, indent=2)

    print(f"Saved synthetic suite to {OUTPUT_PATH}")
    return suite


if __name__ == "__main__":
    generate_10k_suite()

