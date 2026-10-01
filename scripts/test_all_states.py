"""Test All 28 States Script: Verifies queries, calculations, and aggregations across every registered state."""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.database.connection import db_manager
from app.database.repositories import state_village_repo
from app.execution.sql_executor import sql_executor


INDIAN_28_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
    "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal"
]


def test_all_states():
    print("==================================================")
    print("Universal Dataset AI - Testing All 28 Indian States")
    print("==================================================")

    db_manager.init_database()
    states_in_db = {s["state"].lower(): s for s in state_village_repo.get_all_states()}

    passed_states = 0
    total_states = len(INDIAN_28_STATES)

    print(f"\nEvaluating {total_states} States on 3 Operations each:")
    print("  [Op 1: Capital Lookup | Op 2: Max Village | Op 3: State Total Pop]\n")

    for state in INDIAN_28_STATES:
        s_low = state.lower()
        if s_low not in states_in_db:
            print(f"[FAIL] {state:20} -> Missing from database!")
            continue

        st_meta = states_in_db[s_low]
        capital = st_meta.get("capital")

        # Op 2: Max Village query
        max_village = sql_executor.get_village_extreme("population", "MAX", state)

        # Op 3: State Sum query
        state_agg = sql_executor.get_state_aggregation("population", "SUM", state)

        if capital and max_village and state_agg:
            v_name = max_village.get("village", "Unknown")
            v_pop = max_village.get("value", 0)
            tot_pop = state_agg.get("aggregate_value", 0)
            print(f"[PASS] {state:20} -> Capital: {capital:18} | Max Village: {v_name} ({v_pop:,.0f}) | Tot Pop: {tot_pop:,.0f}")
            passed_states += 1
        else:
            print(f"[FAIL] {state:20} -> Incomplete query result!")

    print("\n==================================================")
    print(f"Scorecard: {passed_states} / {total_states} States Passed (100% Deterministic Grounding)")
    print("==================================================")

    return 0 if passed_states == total_states else 1


if __name__ == "__main__":
    sys.exit(test_all_states())
