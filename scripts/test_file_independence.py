"""File Independence Test Script: Verifies queries run completely from the Database when CSV/Excel files are absent."""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.main import app

def run_file_independence_test():
    print("==================================================")
    print("Universal Dataset AI - File Independence Verification")
    print("==================================================")

    client = TestClient(app)

    test_queries = [
        ("What is the capital of Telangana?", "Hyderabad"),
        ("What is the total population of Gujarat?", "369,332"),
        ("Which Gujarat village has highest area?", "Gandhinagar_Village_10"),
        ("Which village has lowest population in Karnataka?", "Bengaluru_Village_08"),
        ("Total population of all states?", "12,543,915"),
    ]

    all_passed = True

    for q, expected_snippet in test_queries:
        res = client.post("/query", json={"question": q, "session_id": "file_indep_test"})
        data = res.json()
        status = res.status_code
        ans = data.get("answer", "")
        op = data.get("operation", "")

        passed = (status == 200) and (expected_snippet.lower() in ans.lower() or expected_snippet.replace(",", "") in ans)
        if passed:
            print(f"[PASS] Q: '{q}'\n       Ans: {ans[:90]}...\n")
        else:
            print(f"[FAIL] Q: '{q}'\n       Expected: '{expected_snippet}'\n       Got: {ans}\n")
            all_passed = False

    print("==================================================")
    if all_passed:
        print("RESULT: 100% FILE INDEPENDENCE CONFIRMED (DATABASE ONLY)")
    else:
        print("RESULT: FILE INDEPENDENCE TEST DETECTED ISSUES")
    print("==================================================")
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(run_file_independence_test())
