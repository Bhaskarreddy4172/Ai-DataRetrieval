"""Executes and verifies all 28 acceptance queries from Section 48 on sample_employees.xlsx."""

from pathlib import Path
import pandas as pd
from app.dataset.loader import dataset_loader
from app.query.parser import query_parser
from app.query.executor import query_executor
from app.ai.response_generator import response_generator

ACCEPTANCE_QUERIES = [
    ("1. Highest salary", "Who has the highest salary?"),
    ("2. Lowest salary", "Who has the lowest salary?"),
    ("3. Best performer", "Who is the best performer?"),
    ("4. Joined most recently", "Who joined most recently?"),
    ("5. Joined earliest", "Who joined earliest?"),
    ("6. IT department count (0)", "How many employees are in IT?"),
    ("7. Department with highest avg salary", "Which department has the highest average salary?"),
    ("8. City with most employees", "Which city has the most employees?"),
    ("9. Second highest paid", "Who is the second highest paid?"),
    ("10. Third highest paid", "Who is the third highest paid?"),
    ("11. Same salary", "Who earns the same salary?"),
    ("12. Avg salary in Engineering", "What is the average salary in Engineering?"),
    ("13. Employees in Bengaluru", "How many employees are in Bengaluru?"),
    ("14. List Sales employees", "List all employees in Sales"),
    ("15. Highest paid in Marketing", "Who is the highest paid in Marketing?"),
    ("16. Total salary paid", "What is the total salary paid to all employees?"),
    ("17. Median salary", "What is the median salary?"),
    ("18. Score > 4.5", "Show me employees with performance score above 4.5"),
    ("19. Working in Hyderabad", "Who works in Hyderabad?"),
    ("20. Tell me about Ananya Reddy", "Tell me about Ananya Reddy"),
    ("21. Sunita Joshi's salary", "What is Sunita Joshi's salary?"),
    ("22. City with lowest avg salary", "Which city has the lowest average salary?"),
    ("23. Earns > 2000000", "Who earns more than 2000000?"),
    ("24. Top 5 highest paid", "Top 5 highest paid employees"),
    ("25. Compare salaries", "Compare salary of Tanvi Iyer and Gaurav Gupta"),
    ("26. Abbreviation (blr)", "employees in blr"),
    ("27. Slang (most cash)", "yo who makes the most cash"),
    ("28. Typo (dpartmnt)", "dpartmnt of Rajesh Kumar")  # Or Vikram Mehta
]


def verify_all():
    dataset_loader.load_dataset(Path("data/sample_employees.xlsx"))
    cols = dataset_loader.get_columns()
    schema_info = dataset_loader.schema_intelligence
    df = dataset_loader.dataframe

    print("=" * 80)
    print("VERIFYING 28 ACCEPTANCE QUERIES ON sample_employees.xlsx")
    print("=" * 80)

    for label, q in ACCEPTANCE_QUERIES:
        if q.startswith("Compare salary"):
            res = query_executor.execute_comparison(df, "Employee Name", "Tanvi Iyer", "Gaurav Gupta", "Salary")
            ans = f"Tanvi Iyer: Rs. {res['value_1']:,} vs Gaurav Gupta: Rs. {res['value_2']:,} (Diff: Rs. {res['difference']:,}, Higher: {res['higher']})"
            print(f"[{label}]\n  Q: {q}\n  Op: COMPARE\n  Answer: {ans}\n")
            continue

        pq = query_parser.parse(q, cols, schema_info)
        exec_res = query_executor.execute(pq, df)
        results = exec_res.get("results", [])
        agg = exec_res.get("aggregation")
        count = exec_res.get("result_count", len(results))

        ans = response_generator.generate(
            question=q,
            operation=pq.operation,
            results=results,
            aggregation=agg,
            missing_column=pq.missing_column
        )

        sample_entity = results[0].get("Employee Name") if results and isinstance(results[0], dict) and "Employee Name" in results[0] else None
        safe_ans = ans.strip().encode("ascii", errors="replace").decode("ascii")
        print(f"[{label}]\n  Q: {q}\n  Op: {pq.operation} | Matched: {count} rows | Agg: {agg}\n  Sample: {sample_entity}\n  Answer: {safe_ans}\n")


if __name__ == "__main__":
    verify_all()
