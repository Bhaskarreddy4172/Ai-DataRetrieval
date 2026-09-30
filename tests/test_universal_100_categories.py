"""Universal 100-Type Dataset Question Solver Test Suite.
Tests all 100 question categories and combination queries across registered datasets.
Zero state-specific hardcoding. Deterministic calculations via Pandas/DuckDB.
"""

import time
import pytest
from app.api.routes import process_query, QueryRequest
from app.query.global_aggregation_engine import universal_global_aggregation_engine
from app.query.comparison_engine import universal_comparison_engine
from app.dataset.hierarchical_engine import hierarchical_query_engine
from app.dataset.registry import parent_child_registry


class TestUniversal100Categories:
    """Comprehensive test suite covering the 100 universal question categories."""

    # 1. Direct lookup
    def test_01_direct_lookup(self):
        res = process_query(QueryRequest(question="What is the capital of Telangana?"))
        assert "Hyderabad" in res.answer

    # 2. Reverse lookup
    def test_02_reverse_lookup(self):
        res = process_query(QueryRequest(question="Which state has the capital Hyderabad?"))
        assert "Telangana" in res.answer

    # 3. Entity lookup
    def test_03_entity_lookup(self):
        res = process_query(QueryRequest(question="Details of Telangana"))
        assert "Telangana" in res.answer

    # 4. Column lookup
    def test_04_column_lookup(self):
        res = process_query(QueryRequest(question="What columns are in the main dataset?"))
        assert res.grounded is True

    # 5. Row lookup
    def test_05_row_lookup(self):
        res = process_query(QueryRequest(question="Details of TG-001"))
        assert "117,305" in res.answer or "Hyderabad_Village_01" in res.answer

    # 6. State lookup
    def test_06_state_lookup(self):
        res = process_query(QueryRequest(question="Tell me about Karnataka"))
        assert "Bengaluru" in res.answer or "Karnataka" in res.answer

    # 7. Village lookup
    def test_07_village_lookup(self):
        res = process_query(QueryRequest(question="List villages of Gujarat"))
        assert "Gujarat" in res.answer or "Gandhinagar" in res.answer

    # 8. Capital lookup
    def test_08_capital_lookup(self):
        res = process_query(QueryRequest(question="What is the capital of Bihar?"))
        assert "Patna" in res.answer

    # 9. COUNT
    def test_09_count_states(self):
        res = process_query(QueryRequest(question="How many states are in the dataset?"))
        assert "28" in res.answer

    # 10. Village COUNT
    def test_10_village_count(self):
        res = process_query(QueryRequest(question="How many villages are in Telangana?"))
        assert "10" in res.answer

    # 11. SUM
    def test_11_single_state_sum(self):
        res = process_query(QueryRequest(question="Total population of Telangana"))
        assert "919,813" in res.answer

    # 12. Global SUM
    def test_12_global_sum(self):
        res = process_query(QueryRequest(question="What is the total population of all states?"))
        assert "12,543,915" in res.answer

    # 13. Selected SUM
    def test_13_selected_sum(self):
        res = process_query(QueryRequest(question="Total population of TG and AP"))
        assert "1,295,813" in res.answer

    # 14. AVERAGE
    def test_14_average(self):
        res = process_query(QueryRequest(question="Average population of all states"))
        assert "447,997" in res.answer or "447,996.96" in res.answer

    # 15. MEDIAN
    def test_15_median(self):
        res = process_query(QueryRequest(question="Median population of all states"))
        assert res.grounded is True
        assert res.result_count >= 1

    # 16. MIN
    def test_16_min(self):
        res = process_query(QueryRequest(question="Which state has the lowest population?"))
        assert "Goa" in res.answer

    # 17. MAX
    def test_17_max(self):
        res = process_query(QueryRequest(question="Which state has the highest population?"))
        assert "West Bengal" in res.answer

    # 18. RANGE
    def test_18_range(self):
        res = process_query(QueryRequest(question="What is the range of population across states?"))
        assert res.grounded is True

    # 19. DISTINCT
    def test_19_distinct(self):
        res = process_query(QueryRequest(question="List distinct capitals"))
        assert res.grounded is True
        assert res.result_count >= 1

    # 20. TOP-N
    def test_20_top_n(self):
        res = process_query(QueryRequest(question="Top 5 states by population"))
        assert "West Bengal" in res.answer
        assert "Maharashtra" in res.answer

    # 21. BOTTOM-N
    def test_21_bottom_n(self):
        res = process_query(QueryRequest(question="Bottom 5 states by population"))
        assert "Goa" in res.answer

    # 22. RANKING
    def test_22_ranking(self):
        res = process_query(QueryRequest(question="Rank all states by population"))
        assert "West Bengal" in res.answer

    # 23. SORT
    def test_23_sort(self):
        res = process_query(QueryRequest(question="Sort states by population"))
        assert res.grounded is True

    # 24. FILTER
    def test_24_filter(self):
        res = process_query(QueryRequest(question="Villages in Telangana with population above 80000"))
        assert res.grounded is True

    # 25. MULTI-FILTER
    def test_25_multi_filter(self):
        res = process_query(QueryRequest(question="Villages in Telangana with population above 50000 and literacy above 80"))
        assert res.grounded is True

    # 26. GROUP BY
    def test_26_group_by(self):
        res = process_query(QueryRequest(question="Group villages by state"))
        assert res.grounded is True

    # 27. GROUP AGGREGATION
    def test_27_group_aggregation(self):
        res = process_query(QueryRequest(question="Total population breakdown by state"))
        assert "West Bengal" in res.answer
        assert "Maharashtra" in res.answer

    # 28. COMPARISON
    def test_28_comparison(self):
        res = process_query(QueryRequest(question="Compare Telangana and Andhra Pradesh population"))
        assert "919,813" in res.answer
        assert "376,000" in res.answer

    # 29. DIFFERENCE
    def test_29_difference(self):
        res = process_query(QueryRequest(question="Difference in population between TG and AP"))
        assert "543,813" in res.answer

    # 30. MORE THAN
    def test_30_more_than(self):
        res = process_query(QueryRequest(question="How many more people does Telangana have than Andhra Pradesh?"))
        assert "543,813 more people" in res.answer

    # 31. LESS THAN
    def test_31_less_than(self):
        res = process_query(QueryRequest(question="How many fewer people does Andhra Pradesh have than Telangana?"))
        assert "543,813 fewer people" in res.answer

    # 32. PERCENTAGE
    def test_32_percentage(self):
        res = process_query(QueryRequest(question="What percentage of Village 1 in TG is male?"))
        assert res.grounded is True

    # 33. PERCENTAGE CHANGE
    def test_33_percentage_change(self):
        res = process_query(QueryRequest(question="Percentage difference in population between TG and AP"))
        assert res.grounded is True
        assert "%" in res.answer

    # 34. RATIO
    def test_34_ratio(self):
        res = process_query(QueryRequest(question="Ratio of population between TG and AP"))
        assert ":" in res.answer

    # 35. BOOLEAN
    def test_35_boolean(self):
        res = process_query(QueryRequest(question="Is Telangana more populated than Andhra Pradesh?"))
        assert "Yes" in res.answer

    # 36. YES/NO
    def test_36_yes_no(self):
        res = process_query(QueryRequest(question="Does Andhra Pradesh have more population than Telangana?"))
        assert "No" in res.answer

    # 37. SAME/DIFFERENT
    def test_37_same_different(self):
        res = process_query(QueryRequest(question="Are Telangana and AP different in population?"))
        assert "different" in res.answer.lower() or "yes" in res.answer.lower()

    # 38. GREATER/LESS
    def test_38_greater_less(self):
        res = process_query(QueryRequest(question="Is AP population lower than TG?"))
        assert "Yes" in res.answer

    # 39. CROSS-STATE
    def test_39_cross_state(self):
        res = process_query(QueryRequest(question="Compare village 1 in Maharashtra and village 2 in Karnataka"))
        assert "25,289" in res.answer
        assert "Maharashtra" in res.answer
        assert "Karnataka" in res.answer

    # 40. CROSS-COLUMN
    def test_40_cross_column(self):
        res = process_query(QueryRequest(question="Villages in Telangana where Males > Females"))
        assert res.grounded is True

    # 41. CROSS-ROW
    def test_41_cross_row(self):
        res = process_query(QueryRequest(question="Compare TG-001 and TG-002"))
        assert "38,024" in res.answer

    # 42. PARENT LOOKUP
    def test_42_parent_lookup(self):
        res = process_query(QueryRequest(question="Which state contains Hyderabad_Village_01?"))
        assert "Telangana" in res.answer

    # 43. CHILD LOOKUP
    def test_43_child_lookup(self):
        res = process_query(QueryRequest(question="What are the villages of Maharashtra?"))
        assert "Maharashtra" in res.answer or "Mumbai" in res.answer

    # 44. PARENT -> CHILD
    def test_44_parent_to_child(self):
        res = process_query(QueryRequest(question="Total village population of Telangana"))
        assert "919,813" in res.answer

    # 45. CHILD -> PARENT
    def test_45_child_to_parent(self):
        res = process_query(QueryRequest(question="What state does Panaji belong to?"))
        assert "Goa" in res.answer

    # 46. MULTI-HOP
    def test_46_multi_hop(self):
        res = process_query(QueryRequest(question="Total population of villages in the state whose capital is Hyderabad"))
        assert "919,813" in res.answer

    # 47. GLOBAL QUERY
    def test_47_global_query(self):
        res = process_query(QueryRequest(question="Total population of all villages"))
        assert "12,543,915" in res.answer

    # 48. DATE QUERY
    def test_48_date_query(self):
        res = process_query(QueryRequest(question="When was the dataset loaded or created?"))
        assert res.grounded is True

    # 49. STATISTICS
    def test_49_statistics(self):
        res = process_query(QueryRequest(question="Standard deviation of population across states"))
        assert res.grounded is True

    # 50. DISTRIBUTION
    def test_50_distribution(self):
        res = process_query(QueryRequest(question="Population distribution across states"))
        assert res.grounded is True

    # 51. MISSING DATA
    def test_51_missing_data(self):
        res = process_query(QueryRequest(question="Are there missing values in the dataset?"))
        assert res.grounded is True

    # 52. DUPLICATES
    def test_52_duplicates(self):
        res = process_query(QueryRequest(question="Are there any duplicate rows in the dataset?"))
        assert res.grounded is True

    # 53. DATA QUALITY
    def test_53_data_quality(self):
        res = process_query(QueryRequest(question="Data quality report of the dataset"))
        assert res.grounded is True

    # 54. SCHEMA
    def test_54_schema(self):
        res = process_query(QueryRequest(question="What columns are available?"))
        assert res.grounded is True

    # 55. DATASET SUMMARY
    def test_55_dataset_summary(self):
        res = process_query(QueryRequest(question="Summarize the dataset"))
        assert res.grounded is True

    # 56. FOLLOW-UP
    def test_56_follow_up(self):
        sid = f"test-fup-{int(time.time())}"
        process_query(QueryRequest(question="What is the population of Telangana?", session_id=sid))
        res = process_query(QueryRequest(question="What about AP?", session_id=sid))
        assert "376,000" in res.answer

    # 57. CONTEXTUAL QUERY
    def test_57_contextual_query(self):
        sid = f"test-ctx-{int(time.time())}"
        process_query(QueryRequest(question="Compare Telangana and Andhra Pradesh", session_id=sid))
        res = process_query(QueryRequest(question="Which one has more?", session_id=sid))
        assert "Telangana" in res.answer

    # 58. TYPO QUERY
    def test_58_typo_query(self):
        res = process_query(QueryRequest(question="Total pop of Telengana?"))
        assert "919,813" in res.answer

    # 59. ABBREVIATION QUERY
    def test_59_abbreviation_query(self):
        res = process_query(QueryRequest(question="Total pop of TG?"))
        assert "919,813" in res.answer

    # 60. BROKEN/INFORMAL QUERY
    def test_60_broken_informal_query(self):
        res = process_query(QueryRequest(question="tg total ppl?"))
        assert "919,813" in res.answer

    # 61. MULTI-ENTITY LOOKUP
    def test_61_multi_entity_lookup(self):
        res = process_query(QueryRequest(question="Capitals of Telangana, Andhra Pradesh and Karnataka"))
        assert "Hyderabad" in res.answer
        assert "Amaravati" in res.answer
        assert "Bengaluru" in res.answer

    # 62. MULTI-ENTITY COMPARISON
    def test_62_multi_entity_comparison(self):
        res = process_query(QueryRequest(question="Compare population of Telangana, Andhra Pradesh and Karnataka"))
        assert "919,813" in res.answer
        assert "376,000" in res.answer
        assert "680,515" in res.answer

    # 63. MULTI-CONDITION COMPARISON
    def test_63_multi_condition_comparison(self):
        res = process_query(QueryRequest(question="Compare village 1 in TG and village 2 in AP for population"))
        assert "74,461" in res.answer

    # 64. ABSOLUTE DIFFERENCE
    def test_64_absolute_difference(self):
        res = process_query(QueryRequest(question="What is the absolute population difference between TG and AP?"))
        assert "543,813" in res.answer

    # 65. PERCENTAGE DIFFERENCE
    def test_65_percentage_difference(self):
        res = process_query(QueryRequest(question="What percentage more population does Telangana have than Andhra Pradesh?"))
        assert "%" in res.answer
        assert "144.63%" in res.answer or "more" in res.answer.lower()

    # 66. POPULATION SHARE
    def test_66_population_share(self):
        res = process_query(QueryRequest(question="What is the population share of Telangana across all states?"))
        assert res.grounded is True

    # 67. WEIGHTED AVERAGE
    def test_67_weighted_average(self):
        res = process_query(QueryRequest(question="Weighted average literacy across all states"))
        assert res.grounded is True

    # 68. CONDITIONAL SUM
    def test_68_conditional_sum(self):
        res = process_query(QueryRequest(question="Total population of villages in Telangana with population above 50000"))
        assert res.grounded is True

    # 69. CONDITIONAL COUNT
    def test_69_conditional_count(self):
        res = process_query(QueryRequest(question="How many villages in Telangana have population above 50000?"))
        assert res.grounded is True

    # 70. CONDITIONAL AVERAGE
    def test_70_conditional_average(self):
        res = process_query(QueryRequest(question="Average population of villages in Telangana with population above 50000"))
        assert res.grounded is True

    # 71. CONDITIONAL MIN
    def test_71_conditional_min(self):
        res = process_query(QueryRequest(question="Lowest population among villages in Telangana with population above 50000"))
        assert res.grounded is True

    # 72. CONDITIONAL MAX
    def test_72_conditional_max(self):
        res = process_query(QueryRequest(question="Highest population among villages in Telangana with population above 50000"))
        assert res.grounded is True

    # 73. GROUPED COUNT
    def test_73_grouped_count(self):
        res = process_query(QueryRequest(question="Number of villages per state"))
        assert res.grounded is True

    # 74. GROUPED AVERAGE
    def test_74_grouped_average(self):
        res = process_query(QueryRequest(question="Average village population by state"))
        assert res.grounded is True

    # 75. GROUPED MIN
    def test_75_grouped_min(self):
        res = process_query(QueryRequest(question="Lowest village population in each state"))
        assert res.grounded is True

    # 76. GROUPED MAX
    def test_76_grouped_max(self):
        res = process_query(QueryRequest(question="Highest village population in each state"))
        assert res.grounded is True

    # 77. GROUPED STATISTICS
    def test_77_grouped_statistics(self):
        res = process_query(QueryRequest(question="Population statistics per state"))
        assert res.grounded is True

    # 78. ENTITY EXISTENCE
    def test_78_entity_existence(self):
        res = process_query(QueryRequest(question="Does Telangana exist in the dataset?"))
        assert "Yes" in res.answer

    # 79. VALUE EXISTENCE
    def test_79_value_existence(self):
        res = process_query(QueryRequest(question="Is there any village with population greater than 100000?"))
        assert "Yes" in res.answer

    # 80. ATTRIBUTE EXISTENCE
    def test_80_attribute_existence(self):
        res = process_query(QueryRequest(question="Does the dataset have Literacy Rate?"))
        assert "Yes" in res.answer or "Literacy" in res.answer

    # 81. MISSING-VALUE SEARCH
    def test_81_missing_value_search(self):
        res = process_query(QueryRequest(question="Find rows where population is null"))
        assert res.grounded is True

    # 82. DUPLICATE DETECTION
    def test_82_duplicate_detection(self):
        res = process_query(QueryRequest(question="Check for duplicate rows"))
        assert res.grounded is True

    # 83. UNIQUE-VALUE LISTING
    def test_83_unique_value_listing(self):
        res = process_query(QueryRequest(question="List all unique capitals"))
        assert res.grounded is True

    # 84. CONDITIONAL EXISTENCE
    def test_84_conditional_existence(self):
        res = process_query(QueryRequest(question="Are there any villages in Telangana with population over 100000?"))
        assert "Yes" in res.answer

    # 85. THRESHOLD RANKING
    def test_85_threshold_ranking(self):
        res = process_query(QueryRequest(question="States with population above 800000"))
        assert res.grounded is True

    # 86. RANGE FILTERING
    def test_86_range_filtering(self):
        res = process_query(QueryRequest(question="Villages in Telangana with population between 50000 and 80000"))
        assert res.grounded is True

    # 87. MULTIPLE RANGE FILTERING
    def test_87_multiple_range_filtering(self):
        res = process_query(QueryRequest(question="Villages in Telangana with population between 50000 and 80000 and literacy between 70% and 90%"))
        assert res.grounded is True

    # 88. ATTRIBUTE-TO-ATTRIBUTE COMPARISON
    def test_88_attr_to_attr_comparison(self):
        res = process_query(QueryRequest(question="Which villages have more males than females?"))
        assert res.grounded is True

    # 89. RATIO COMPARISON
    def test_89_ratio_comparison(self):
        res = process_query(QueryRequest(question="Compare male to female ratio in TG vs AP"))
        assert res.grounded is True

    # 90. PERCENTAGE COMPOSITION
    def test_90_percentage_composition(self):
        res = process_query(QueryRequest(question="What percentage of Telangana's population is in Hyderabad_Village_01?"))
        assert "%" in res.answer or res.grounded is True

    # 91. PARENT AGGREGATION
    def test_91_parent_aggregation(self):
        res = process_query(QueryRequest(question="Total population of all 28 states"))
        assert "12,543,915" in res.answer

    # 92. CHILD AGGREGATION
    def test_92_child_aggregation(self):
        res = process_query(QueryRequest(question="Sum of all village populations in Telangana"))
        assert "919,813" in res.answer

    # 93. PARENT RANKING
    def test_93_parent_ranking(self):
        res = process_query(QueryRequest(question="Rank all states by village count"))
        assert res.grounded is True

    # 94. CHILD RANKING
    def test_94_child_ranking(self):
        res = process_query(QueryRequest(question="Top 5 villages by population across all states"))
        assert res.grounded is True

    # 95. CROSS-LEVEL COMPARISON
    def test_95_cross_level_comparison(self):
        res = process_query(QueryRequest(question="Compare Telangana population with Hyderabad population"))
        assert "919,813" in res.answer
        assert "same" in res.answer.lower()

    # 96. MULTI-HOP AGGREGATION
    def test_96_multi_hop_aggregation(self):
        res = process_query(QueryRequest(question="Total population of villages for capital Amaravati"))
        assert "376,000" in res.answer

    # 97. NATURAL-LANGUAGE PARAPHRASE
    def test_97_natural_language_paraphrase(self):
        res = process_query(QueryRequest(question="How many citizens live in the state of Telangana?"))
        assert "919,813" in res.answer

    # 98. MULTILINGUAL/CODE-SWITCHED QUERY
    def test_98_multilingual_code_switched(self):
        res = process_query(QueryRequest(question="tg lo total population entha?"))
        assert "919,813" in res.answer

    # 99. AMBIGUOUS QUERY
    def test_99_ambiguous_query(self):
        res = process_query(QueryRequest(question="Population of XYZ Unknown Entity?"))
        assert "not found" in res.answer.lower() or "not available" in res.answer.lower() or "could not find" in res.answer.lower() or "does not contain" in res.answer.lower() or "cannot determine" in res.answer.lower()

    # 100. UNKNOWN/NO-MATCH QUERY
    def test_100_unknown_no_match(self):
        res = process_query(QueryRequest(question="What is the population of Atlantis?"))
        assert "not found" in res.answer.lower() or "not available" in res.answer.lower() or "could not find" in res.answer.lower() or "does not contain" in res.answer.lower() or "cannot determine" in res.answer.lower()


class TestUniversalCombinationQueries:
    """Test required combination query patterns from prompt section 41."""

    def test_comb_01_filter_sum(self):
        """FILTER + SUM: Total population of villages in Telangana with literacy above 80%."""
        res = process_query(QueryRequest(question="Total population of villages in Telangana with literacy above 80%"))
        assert res.grounded is True

    def test_comb_02_filter_count(self):
        """FILTER + COUNT: Count of villages with population greater than 50000."""
        res = process_query(QueryRequest(question="Count of villages with population greater than 50000"))
        assert res.grounded is True

    def test_comb_03_filter_average(self):
        """FILTER + AVERAGE: Average population of villages with literacy above 80%."""
        res = process_query(QueryRequest(question="Average population of villages with literacy above 80%"))
        assert res.grounded is True

    def test_comb_04_group_sum(self):
        """GROUP + SUM: Total population per state."""
        res = process_query(QueryRequest(question="Total population per state"))
        assert res.grounded is True

    def test_comb_05_group_count(self):
        """GROUP + COUNT: Number of villages per state."""
        res = process_query(QueryRequest(question="Number of villages per state"))
        assert res.grounded is True

    def test_comb_06_group_max(self):
        """GROUP + MAX: Highest village population in each state."""
        res = process_query(QueryRequest(question="Highest village population in each state"))
        assert res.grounded is True

    def test_comb_07_group_min(self):
        """GROUP + MIN: Lowest village population in each state."""
        res = process_query(QueryRequest(question="Lowest village population in each state"))
        assert res.grounded is True

    def test_comb_08_group_rank(self):
        """GROUP + RANK: Top 5 states by total village population."""
        res = process_query(QueryRequest(question="Top 5 states by total village population"))
        assert "West Bengal" in res.answer

    def test_comb_09_compare_percentage(self):
        """COMPARE + PERCENTAGE: What percentage more population does Telangana have than Andhra Pradesh?"""
        res = process_query(QueryRequest(question="What percentage more population does Telangana have than Andhra Pradesh?"))
        assert "%" in res.answer

    def test_comb_10_compare_difference(self):
        """COMPARE + DIFFERENCE: How many more people does Telangana have than Andhra Pradesh?"""
        res = process_query(QueryRequest(question="How many more people does Telangana have than Andhra Pradesh?"))
        assert "543,813 more people" in res.answer

    def test_comb_11_parent_child_sum(self):
        """PARENT + CHILD + SUM: Total population of villages in Telangana."""
        res = process_query(QueryRequest(question="Total population of villages in Telangana"))
        assert "919,813" in res.answer

    def test_comb_12_cross_state_comparison(self):
        """CROSS-STATE + COMPARISON: Compare village 1 in Maharashtra and village 2 in Karnataka."""
        res = process_query(QueryRequest(question="Compare village 1 in Maharashtra and village 2 in Karnataka"))
        assert "25,289" in res.answer
        assert "Maharashtra" in res.answer
        assert "Karnataka" in res.answer

    def test_comb_13_multi_hop_aggregation(self):
        """MULTI-HOP + AGGREGATION: Total population of villages in the state whose capital is Hyderabad."""
        res = process_query(QueryRequest(question="What is the total population of villages in the state whose capital is Hyderabad?"))
        assert "919,813" in res.answer

    def test_comb_14_multi_entity_ranking(self):
        """MULTI-ENTITY + RANKING: Rank Telangana, Andhra Pradesh, Maharashtra, Karnataka, and Tamil Nadu by population."""
        res = process_query(QueryRequest(question="Rank Telangana, Andhra Pradesh, Maharashtra, Karnataka, and Tamil Nadu by population"))
        assert "Maharashtra" in res.answer
        assert "Telangana" in res.answer

    def test_comb_15_multi_condition_aggregation(self):
        """MULTI-CONDITION + AGGREGATION: Total population of villages with population > 50000 and literacy > 80%."""
        res = process_query(QueryRequest(question="Total population of villages with population > 50000 and literacy > 80%"))
        assert res.grounded is True
