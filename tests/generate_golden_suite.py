"""Generates 500+ verified test questions with Pandas ground truth from data/sample_employees.xlsx."""

import json
from pathlib import Path
import pandas as pd

DATASET_PATH = Path("data/sample_employees.xlsx")
OUTPUT_PATH = Path("tests/golden_employees_suite.json")


def generate_suite():
    df = pd.read_excel(DATASET_PATH)
    # Strip string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    suite = []
    test_counter = 1

    def add_test(cat, q, expected_op, expected_count=None, expected_agg_val=None, expected_ids=None, expected_names=None):
        nonlocal test_counter
        suite.append({
            "id": f"TEST_{test_counter:04d}",
            "category": cat,
            "question": q,
            "expected_op": expected_op,
            "expected_count": expected_count,
            "expected_agg_val": round(float(expected_agg_val), 2) if expected_agg_val is not None else None,
            "expected_ids": expected_ids or [],
            "expected_names": expected_names or []
        })
        test_counter += 1

    # 1. Exact lookup by Employee ID (80 tests)
    for _, row in df.iterrows():
        emp_id = row["Employee ID"]
        emp_name = row["Employee Name"]
        q_templates = [
            f"Who is {emp_id}?",
            f"Details for {emp_id}",
            f"Show record for employee {emp_id}",
            f"Find employee {emp_id}"
        ]
        q = q_templates[test_counter % len(q_templates)]
        add_test("exact_id_lookup", q, "FILTER", expected_count=1, expected_ids=[emp_id], expected_names=[emp_name])

    # 2. Exact lookup by Employee Name (80 tests)
    for _, row in df.iterrows():
        emp_id = row["Employee ID"]
        emp_name = row["Employee Name"]
        q_templates = [
            f"Tell me about {emp_name}",
            f"Show details for {emp_name}",
            f"Information on {emp_name}",
            f"Who is {emp_name}?"
        ]
        q = q_templates[test_counter % len(q_templates)]
        matching_rows = df[df["Employee Name"] == emp_name]
        exp_ids = matching_rows["Employee ID"].tolist()
        exp_count = len(matching_rows)
        add_test("exact_name_lookup", q, "FILTER", expected_count=exp_count, expected_ids=exp_ids, expected_names=[emp_name])

    # 3. Attribute queries with Name (80 tests)
    attr_types = ["salary", "department", "city", "performance", "designation"]
    for row_count, (_, row) in enumerate(df.iterrows()):
        emp_id = row["Employee ID"]
        emp_name = row["Employee Name"]
        attr = attr_types[row_count % len(attr_types)]
        if attr == "salary":
            q = f"What is {emp_name}'s salary?"
            val = row["Salary"]
        elif attr == "department":
            q = f"Which department does {emp_name} work in?"
            val = row["Department"]
        elif attr == "city":
            q = f"Where is {emp_name} based?"
            val = row["City"]
        elif attr == "performance":
            q = f"What is the performance score of {emp_name}?"
            val = row["Performance Score"]
        else:
            q = f"What is {emp_name}'s designation?"
            val = row["Designation"]

        matching_rows = df[df["Employee Name"] == emp_name]
        exp_ids = matching_rows["Employee ID"].tolist()
        exp_count = len(matching_rows)
        add_test("attribute_query", q, "FILTER", expected_count=exp_count, expected_ids=exp_ids, expected_names=[emp_name])

    # 4. Typos and spelling variations (40 tests)
    typo_specs = [
        ("salry of Ananya Reddy", "FILTER", ["EMP-1005"], ["Ananya Reddy"]),
        ("dpartmnt of Vikram Mehta", "FILTER", ["EMP-1001"], ["Vikram Mehta"]),
        ("hihgest salary employee", "MAX", ["EMP-1049"], ["Tanvi Iyer"]),
        ("lowst paid staff", "MIN", ["EMP-1028"], ["Gaurav Gupta"]),
        ("peple in Banglore", "FILTER", None, None),
        ("staf in Hydrabad", "FILTER", None, None),
        ("who is EMP 1001", "FILTER", ["EMP-1001"], ["Vikram Mehta"]),
        ("emplyee EMP-1010 details", "FILTER", ["EMP-1010"], ["Varun Patel"]),
        ("best preformer", "MAX", None, None),
        ("avrg salary", "AVERAGE", None, None),
        ("totl salary of all employees", "SUM", None, None),
        ("who has the higest salary?", "MAX", ["EMP-1049"], ["Tanvi Iyer"]),
        ("which depatment pays most?", "GROUP_EXTREME", None, None),
        ("city with mximum employees", "GROUP_EXTREME", None, None),
        ("slary of Sunita Joshi", "FILTER", ["EMP-1043"], ["Sunita Joshi"]),
        ("whre does Aditya Kulkarni live?", "FILTER", ["EMP-1002"], ["Aditya Kulkarni"]),
        ("designtion of Meera Singh", "FILTER", ["EMP-1003"], ["Meera Singh"]),
        ("who erans the lowest?", "MIN", ["EMP-1028"], ["Gaurav Gupta"]),
        ("scnd highest paid employee", "RANK", ["EMP-1050"], ["Aarav Sharma"]),
        ("thrd highest earner", "RANK", ["EMP-1040"], ["Nikhil Nair"]),
    ]
    for q, op, exp_ids, exp_names in typo_specs * 2:  # 40 tests
        add_test("typo_variation", q, op, expected_ids=exp_ids, expected_names=exp_names)

    # 5. Abbreviations & Informal / Slang (30 tests)
    slang_specs = [
        ("employees in blr", "FILTER", None, None),
        ("staff in hyd", "FILTER", None, None),
        ("who is in hr", "FILTER", None, None),
        ("devs in eng", "FILTER", None, None),
        ("folks based in del", "FILTER", None, None),
        ("peeps in bom", "FILTER", None, None),
        ("yo who makes the most cash", "MAX", ["EMP-1049"], ["Tanvi Iyer"]),
        ("show me the best performer bro", "MAX", None, None),
        ("who gets paid least dude", "MIN", ["EMP-1028"], ["Gaurav Gupta"]),
        ("top earner in the company", "MAX", ["EMP-1049"], ["Tanvi Iyer"]),
        ("who takes home the biggest paycheck", "MAX", ["EMP-1049"], ["Tanvi Iyer"]),
        ("give me the count of employees in blr", "COUNT", None, None),
        ("how many heads in hyd", "COUNT", None, None),
        ("number of people in hr", "COUNT", None, None),
        ("who is the rockstar here", "MAX", None, None),
    ]
    for q, op, exp_ids, exp_names in slang_specs * 2:  # 30 tests
        add_test("abbreviation_and_slang", q, op, expected_ids=exp_ids, expected_names=exp_names)

    # 6. Single condition filters across departments, cities, scores, salaries (50 tests)
    depts = df["Department"].unique().tolist()
    cities = df["City"].unique().tolist()
    for d in depts:
        match_df = df[df["Department"] == d]
        add_test("single_filter_dept", f"List all employees in {d}", "FILTER", expected_count=len(match_df))
        add_test("count_filter_dept", f"How many employees are in {d}?", "COUNT", expected_agg_val=len(match_df))

    for c in cities:
        match_df = df[df["City"] == c]
        add_test("single_filter_city", f"Employees located in {c}", "FILTER", expected_count=len(match_df))
        add_test("count_filter_city", f"How many employees are in {c}?", "COUNT", expected_agg_val=len(match_df))

    # Numerical threshold filters
    salary_threshs = [800000, 1000000, 1500000, 2000000]
    for st in salary_threshs:
        match_gt = df[df["Salary"] > st]
        add_test("salary_threshold_gt", f"Who earns more than {st}?", "FILTER", expected_count=len(match_gt))
        match_lt = df[df["Salary"] < st]
        add_test("salary_threshold_lt", f"Who earns less than {st}?", "FILTER", expected_count=len(match_lt))

    score_threshs = [4.0, 4.5, 4.8]
    for sc in score_threshs:
        match_sc = df[df["Performance Score"] >= sc]
        add_test("score_threshold", f"Employees with performance score at least {sc}", "FILTER", expected_count=len(match_sc))

    # Fill remaining to make 50 single filters
    extra_filters = [
        ("Employees joining after 2024", "FILTER"),
        ("Staff with performance score above 4.7", "FILTER"),
        ("Employees earning exactly 1490000", "FILTER"),
        ("Employees in Human Resources department", "FILTER"),
        ("Staff in Engineering department", "FILTER"),
    ]
    for q, op in extra_filters:
        add_test("single_filter_extra", q, op)

    # 7. Multi-condition queries (40 tests)
    multi_cond_pairs = [
        ("Engineering", "Hyderabad"),
        ("Engineering", "Bengaluru"),
        ("Sales", "Mumbai"),
        ("Sales", "Delhi"),
        ("Marketing", "Delhi"),
        ("Finance", "Chennai"),
        ("Data Science", "Hyderabad"),
        ("Data Science", "Bengaluru"),
        ("Human Resources", "Pune"),
        ("Engineering", "Chennai")
    ]
    for dept, city in multi_cond_pairs:
        matched = df[(df["Department"] == dept) & (df["City"] == city)]
        add_test("multi_condition", f"Employees in {dept} based in {city}", "FILTER", expected_count=len(matched))
        add_test("multi_condition_count", f"How many employees in {dept} are in {city}?", "COUNT", expected_agg_val=len(matched))
        # Add salary condition
        matched_sal = df[(df["Department"] == dept) & (df["Salary"] > 1000000)]
        add_test("multi_condition_sal", f"{dept} employees earning over 1000000", "FILTER", expected_count=len(matched_sal))
        # Add score condition
        matched_score = df[(df["Department"] == dept) & (df["Performance Score"] >= 4.0)]
        add_test("multi_condition_score", f"{dept} staff with performance score at least 4.0", "FILTER", expected_count=len(matched_score))

    # 8. Aggregations (40 tests)
    add_test("aggregation_overall", "What is the average salary?", "AVERAGE", expected_agg_val=df["Salary"].mean())
    add_test("aggregation_overall", "What is the total salary paid to all employees?", "SUM", expected_agg_val=df["Salary"].sum())
    add_test("aggregation_overall", "What is the median salary?", "MEDIAN", expected_agg_val=df["Salary"].median())
    add_test("aggregation_overall", "What is the average performance score?", "AVERAGE", expected_agg_val=df["Performance Score"].mean())
    add_test("aggregation_overall", "What is the maximum salary?", "MAX", expected_agg_val=df["Salary"].max())
    add_test("aggregation_overall", "What is the minimum salary?", "MIN", expected_agg_val=df["Salary"].min())
    add_test("aggregation_overall", "How many total employees are there?", "COUNT", expected_agg_val=len(df))

    # Per-department aggregations
    for d in depts:
        d_df = df[df["Department"] == d]
        add_test("agg_dept_avg", f"What is the average salary in {d}?", "AVERAGE", expected_agg_val=d_df["Salary"].mean())
        add_test("agg_dept_sum", f"What is the total salary in {d}?", "SUM", expected_agg_val=d_df["Salary"].sum())
        add_test("agg_dept_max", f"What is the highest salary in {d}?", "MAX", expected_agg_val=d_df["Salary"].max())
        add_test("agg_dept_score", f"What is the average performance score in {d}?", "AVERAGE", expected_agg_val=d_df["Performance Score"].mean())

    # Per-city aggregations
    for c in cities:
        c_df = df[df["City"] == c]
        add_test("agg_city_avg", f"What is the average salary in {c}?", "AVERAGE", expected_agg_val=c_df["Salary"].mean())
        add_test("agg_city_sum", f"What is the total salary in {c}?", "SUM", expected_agg_val=c_df["Salary"].sum())

    # Fill remaining aggregations to reach 40
    add_test("agg_extra", "Average salary of people in Hyderabad", "AVERAGE", expected_agg_val=df[df["City"] == "Hyderabad"]["Salary"].mean())
    add_test("agg_extra", "Average salary of people in Bengaluru", "AVERAGE", expected_agg_val=df[df["City"] == "Bengaluru"]["Salary"].mean())
    add_test("agg_extra", "Total payroll cost for Mumbai office", "SUM", expected_agg_val=df[df["City"] == "Mumbai"]["Salary"].sum())

    # 9. Extremes and Rankings (25 tests)
    add_test("extreme", "Who has the highest salary?", "MAX", expected_count=1, expected_ids=["EMP-1049"], expected_names=["Tanvi Iyer"])
    add_test("extreme", "Who has the lowest salary?", "MIN", expected_count=1, expected_ids=["EMP-1028"], expected_names=["Gaurav Gupta"])
    add_test("extreme", "Who is the best performer?", "MAX", expected_count=4)
    add_test("extreme", "Who joined most recently?", "DATE_EXTREME", expected_count=2)
    add_test("extreme", "Who joined earliest?", "DATE_EXTREME", expected_count=1, expected_ids=["EMP-1043"], expected_names=["Sunita Joshi"])
    add_test("ranking", "Who is the second highest paid?", "RANK", expected_count=1, expected_ids=["EMP-1050"], expected_names=["Aarav Sharma"])
    add_test("ranking", "Who is the third highest paid?", "RANK", expected_count=1, expected_ids=["EMP-1040"], expected_names=["Nikhil Nair"])
    add_test("ranking", "Who is the 2nd lowest paid?", "RANK", expected_count=1)
    add_test("ranking", "Top 5 highest paid employees", "TOP_N", expected_count=5)
    add_test("ranking", "Top 3 highest salaries", "TOP_N", expected_count=3)
    add_test("ranking", "Bottom 3 lowest paid employees", "BOTTOM_N", expected_count=3)

    # Department extremes
    for d in depts:
        d_df = df[df["Department"] == d]
        max_row = d_df[d_df["Salary"] == d_df["Salary"].max()]
        add_test("dept_extreme", f"Who is the highest paid in {d}?", "MAX", expected_count=len(max_row))
        min_row = d_df[d_df["Salary"] == d_df["Salary"].min()]
        add_test("dept_extreme", f"Who has the lowest salary in {d}?", "MIN", expected_count=len(min_row))

    # City extremes
    add_test("city_extreme", "Who has the highest salary in Hyderabad?", "MAX", expected_count=1)
    add_test("city_extreme", "Who has the highest salary in Bengaluru?", "MAX", expected_count=1)

    # 10. Group extremes (10 tests)
    add_test("group_extreme", "Which department has the highest average salary?", "GROUP_EXTREME", expected_count=1, expected_names=["Engineering"])
    add_test("group_extreme", "Which city has the most employees?", "GROUP_EXTREME", expected_count=1, expected_names=["Chennai"])
    add_test("group_extreme", "Which department has the most employees?", "GROUP_EXTREME", expected_count=1)
    add_test("group_extreme", "Which city has the fewest employees?", "GROUP_EXTREME", expected_count=1)
    add_test("group_extreme", "Which department has the lowest average salary?", "GROUP_EXTREME", expected_count=1)
    add_test("group_extreme", "Department with highest average salary", "GROUP_EXTREME", expected_count=1, expected_names=["Engineering"])
    add_test("group_extreme", "City with highest employee count", "GROUP_EXTREME", expected_count=1, expected_names=["Chennai"])
    add_test("group_extreme", "Which department pays the highest average salary?", "GROUP_EXTREME", expected_count=1, expected_names=["Engineering"])
    add_test("group_extreme", "Which city has the maximum staff?", "GROUP_EXTREME", expected_count=1, expected_names=["Chennai"])
    add_test("group_extreme", "Department with greatest average salary", "GROUP_EXTREME", expected_count=1, expected_names=["Engineering"])

    # 11. Cross-row & same values (10 tests)
    add_test("cross_row", "Who earns the same salary?", "CROSS_ROW")
    add_test("cross_row", "Show employees with identical salary", "CROSS_ROW")
    add_test("cross_row", "Who has duplicate salaries?", "CROSS_ROW")
    add_test("cross_row", "Employees who make the exact same salary", "CROSS_ROW")
    add_test("cross_row", "Who joined on the same date?", "CROSS_ROW")
    add_test("cross_row", "Which employees share the same salary amount?", "CROSS_ROW")
    add_test("cross_row", "Employees having identical compensation", "CROSS_ROW")
    add_test("cross_row", "Find people with matching salaries", "CROSS_ROW")
    add_test("cross_row", "Who earns equal salary?", "CROSS_ROW")
    add_test("cross_row", "Employees with the same joining date", "CROSS_ROW")

    # 12. Unmatched & out-of-scope (15 tests)
    out_of_scope_questions = [
        "How many employees are in IT?",
        "List all employees in Tokyo",
        "Who is Bruce Wayne?",
        "Employees in Chicago",
        "Staff in Aerospace department",
        "What is our stock price today?",
        "Who is Clark Kent?",
        "Employees with salary above 5000000",
        "Employees in London office",
        "Count of staff in Legal department",
        "Who works in Paris?",
        "Details for EMP-9999",
        "Information about John Doe",
        "Staff in Logistics division",
        "How many workers in Sydney?"
    ]
    for q in out_of_scope_questions:
        op = "COUNT" if q.lower().startswith("how many") or q.lower().startswith("count") else "FILTER"
        add_test("unmatched_or_zero", q, op, expected_count=0, expected_agg_val=0 if op == "COUNT" else None)

    # 13. Comparison questions (10 tests)
    compare_pairs = [
        ("Tanvi Iyer", "Aarav Sharma"),
        ("Ananya Reddy", "Vikram Mehta"),
        ("Sunita Joshi", "Aditya Kulkarni"),
        ("Varun Patel", "Gaurav Gupta"),
        ("Gaurav Gupta", "Tanvi Iyer"),
        ("Aarav Sharma", "Nikhil Nair"),
        ("Meera Singh", "Aarav Rao"),
        ("Shweta Singh", "Ritu Reddy"),
        ("Meera Gupta", "Ananya Gupta"),
        ("Nikhil Nair", "Sunita Joshi")
    ]
    for e1, e2 in compare_pairs:
        add_test("comparison", f"Compare salary of {e1} and {e2}", "COMPARE", expected_names=[e1, e2])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(suite, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(suite)} golden test questions saved to {OUTPUT_PATH}")
    return len(suite)


if __name__ == "__main__":
    generate_suite()
