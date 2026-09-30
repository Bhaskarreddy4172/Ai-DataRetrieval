"""Query planner: transforms normalized natural language intent into an optimized StructuredQuery."""

import re
from typing import Any, Dict, List, Optional, Tuple
from app.ai.intent_parser import intent_parser
from app.dataset.indexer import dataset_indexer
from app.dataset.schema import schema_intelligence
from app.dataset.semantic_mapper import semantic_column_mapper
from app.dataset.value_resolver import dataset_value_resolver
from app.query.schema import FilterCondition, StructuredQuery
from app.query.semantic_parser import semantic_relationship_parser
from app.utils.fuzzy_match import expand_abbreviations, extract_best_match_from_text
from app.utils.logger import logger
from app.utils.normalization import normalize_question, extract_numeric_filter, extract_date_filter


class QueryPlanner:
    """Plans deterministic StructuredQuery using schema intelligence, semantic mapping, and ambiguity guards."""

    NON_DATASET_TOPICS = [
        "weather", "temperature", "president", "prime minister",
        "crime", "gdp", "capital of india", "who is the pm", "governor", "currency",
        "movie", "capital of usa", "who won the world cup", "stock price", "stock market", "share price"
    ]

    def plan_query(
        self,
        question: str,
        available_columns: List[str],
        schema_dict: Optional[Dict[str, Any]] = None
    ) -> StructuredQuery:
        """Analyze intent, fuzzy-match columns and values, and build executable StructuredQuery with debug trace."""
        normalized_q = normalize_question(question)
        expanded_q = expand_abbreviations(normalized_q)

        # Universal Dataset-Aware Spell Checker & Typo Resolution
        try:
            from app.dataset.spell_checker import dataset_spell_checker
            correction_meta = dataset_spell_checker.correct_full_question(expanded_q)
            if correction_meta and correction_meta.get("corrected_question"):
                expanded_q = correction_meta["corrected_question"]
        except Exception:
            pass

        q_lower = expanded_q.lower()
        cols_lower = [c.lower() for c in available_columns]

        # Trace record
        trace = {
            "original_question": question,
            "normalized_question": normalized_q,
            "expanded_question": expanded_q,
            "column_mappings": {},
            "value_mappings": {},
            "intent": None,
            "confidence": 1.0,
            "is_ambiguous": False
        }

        # 0. Anti-Hallucination Out-of-Scope Guard
        from app.dataset.registry import parent_child_registry
        all_child_cols = [c.lower() for c in parent_child_registry.get_child_columns()]
        for topic in self.NON_DATASET_TOPICS:
            if (topic in q_lower or topic in normalized_q.lower()) and not any(topic in c for c in cols_lower):
                if any(topic in cc for cc in all_child_cols):
                    continue
                logger.info(f"Out-of-scope question detected: '{topic}' not in columns {available_columns}")
                trace["intent"] = "UNKNOWN"
                return StructuredQuery(
                    operation="UNKNOWN",
                    missing_column=topic,
                    explanation=f"The uploaded dataset does not contain information about '{topic}'.",
                    debug_trace=trace
                )

        # 0b. Strict No-Match & Unsupported Diagnosis Engine (Entity Not Found, Column Not Found, Ambiguous, Null Value)
        from app.query.no_match_engine import no_match_engine
        from app.dataset.loader import dataset_loader
        diag = no_match_engine.diagnose_query(
            question=expanded_q,
            available_columns=available_columns,
            df=dataset_loader.dataframe,
            schema_dict=schema_dict
        )
        if diag.category == "AMBIGUOUS":
            trace["is_ambiguous"] = True
            return StructuredQuery(
                operation="CLARIFICATION",
                explanation=diag.user_message,
                clarification_prompt=diag.user_message,
                no_match_type="AMBIGUOUS",
                debug_trace=trace
            )
        elif diag.category in {"COLUMN_NOT_FOUND", "UNSUPPORTED"}:
            return StructuredQuery(
                operation="UNSUPPORTED_QUERY",
                missing_column=diag.attribute_text,
                explanation=diag.user_message,
                no_match_type="COLUMN_NOT_FOUND",
                debug_trace=trace
            )
        elif diag.category == "ENTITY_NOT_FOUND":
            return StructuredQuery(
                operation="NO_MATCH",
                missing_entity=diag.entity_text,
                target_column=diag.target_column,
                explanation=diag.user_message,
                no_match_type="ENTITY_NOT_FOUND",
                debug_trace=trace
            )
        elif diag.category == "NO_MATCHING_ROWS":
            return StructuredQuery(
                operation="NO_MATCH",
                missing_entity=diag.entity_text,
                target_column=diag.target_column,
                explanation=diag.user_message,
                no_match_type="NO_MATCHING_ROWS",
                debug_trace=trace
            )
        elif diag.category == "NULL_VALUE":
            return StructuredQuery(
                operation="NULL_VALUE",
                missing_entity=diag.entity_text,
                target_column=diag.target_column,
                explanation=diag.user_message,
                no_match_type="NULL_VALUE",
                debug_trace=trace
            )

        # 1. Classify Intent
        detected_intent, intent_conf = intent_parser.classify(expanded_q, available_columns)
        trace["intent"] = detected_intent
        trace["confidence"] = intent_conf

        # 2. Extract numeric & date conditions
        numeric_cond = extract_numeric_filter(expanded_q)
        date_cond = extract_date_filter(expanded_q)

        # 3. Handle Special Operations
        # DUPLICATES
        if detected_intent == "DUPLICATES":
            target_col = self._resolve_target_column(expanded_q, available_columns, schema_dict, trace)
            return StructuredQuery(
                operation="DUPLICATES",
                target_column=target_col,
                explanation="Detect duplicate records or values",
                debug_trace=trace
            )

        # DISTINCT
        if detected_intent == "DISTINCT":
            target_col = self._resolve_target_column(expanded_q, available_columns, schema_dict, trace)
            return StructuredQuery(
                operation="DISTINCT",
                target_column=target_col or (available_columns[0] if available_columns else None),
                explanation="List distinct values",
                debug_trace=trace
            )

        # MISSING DATA
        if detected_intent == "MISSING_DATA":
            target_col = self._resolve_target_column(expanded_q, available_columns, schema_dict, trace)
            return StructuredQuery(
                operation="MISSING_DATA",
                target_column=target_col,
                explanation="Find records with missing data",
                debug_trace=trace
            )

        # Relative cross-row comparison: who earns more than / less than {Name}
        rel_cmp_m = re.search(r"\b(?:who\s+)?(?:earns?|makes?|paid|salary|compensation)?\s*(more than|greater than|higher than|less than|lower than)\s+([A-Za-z\s]+?)(?:\?|$)", q_lower)
        if rel_cmp_m:
            op_word = rel_cmp_m.group(1).strip()
            entity_cand = rel_cmp_m.group(2).strip()
            from app.utils.normalization import parse_numeric_value
            if parse_numeric_value(entity_cand) is None:
                from app.dataset.loader import dataset_loader
                cur_df = dataset_loader.dataframe
                if not cur_df.empty:
                    name_col = None
                    for c in available_columns:
                        if any(k in c.lower() for k in ["name", "employee", "customer"]):
                            name_col = c
                            break
                    if name_col:
                        res = dataset_value_resolver.resolve_value_in_column(entity_cand, name_col)
                        matched_name = res.get("resolved_value")
                        if matched_name:
                            matched_row = cur_df[cur_df[name_col] == matched_name]
                            num_col = self._find_numeric_column("salary", available_columns, schema_dict, trace)
                            if not matched_row.empty and num_col:
                                target_val = float(matched_row[num_col].iloc[0])
                                cond_op = ">" if any(w in op_word for w in ["more", "greater", "higher"]) else "<"
                                cond = FilterCondition(column=num_col, operator=cond_op, value=target_val)
                                return StructuredQuery(
                                    operation="CROSS_ROW",
                                    conditions=[cond],
                                    explanation=f"Records where {num_col} {cond_op} {matched_name}'s {num_col} ({target_val})",
                                    debug_trace=trace
                                )

        # Relative cross-row: in the same {Department/City} as {Name}
        same_as_m = re.search(r"\b(?:in the\s+)?same\s+(department|dept|city|role|designation|location)\s+as\s+([A-Za-z\s]+?)(?:\?|$)", q_lower)
        if same_as_m:
            attr_word = same_as_m.group(1).strip()
            entity_cand = same_as_m.group(2).strip()
            from app.dataset.loader import dataset_loader
            cur_df = dataset_loader.dataframe
            if not cur_df.empty:
                name_col = None
                for c in available_columns:
                    if any(k in c.lower() for k in ["name", "employee", "customer"]):
                        name_col = c
                        break
                attr_col = self._resolve_target_column(attr_word, available_columns, schema_dict, trace)
                if name_col and attr_col:
                    res = dataset_value_resolver.resolve_value_in_column(entity_cand, name_col)
                    matched_name = res.get("resolved_value")
                    if matched_name:
                        matched_row = cur_df[cur_df[name_col] == matched_name]
                        if not matched_row.empty:
                            attr_val = matched_row[attr_col].iloc[0]
                            conds = [
                                FilterCondition(column=attr_col, operator="=", value=attr_val),
                                FilterCondition(column=name_col, operator="!=", value=matched_name)
                            ]
                            return StructuredQuery(
                                operation="CROSS_ROW",
                                conditions=conds,
                                explanation=f"Records in the same {attr_col} ({attr_val}) as {matched_name}",
                                debug_trace=trace
                            )

        # CROSS_ROW (e.g. "who earns the same salary", "who works in the same city", "identical salary", "duplicate salaries")
        same_m = re.search(r"\b(?:same|identical|duplicate|matching|equal)\s+([a-zA-Z\s]+)", q_lower)
        if same_m:
            cand_w = same_m.group(1).rstrip("?").strip()
            target_col = self._resolve_target_column(cand_w, available_columns, schema_dict, trace)
            if not target_col:
                target_col = self._find_numeric_column(cand_w, available_columns, schema_dict, trace)
            if target_col:
                return StructuredQuery(
                    operation="CROSS_ROW",
                    target_column=target_col,
                    explanation=f"Find records sharing the same {target_col}",
                    debug_trace=trace
                )

        # GROUP_EXTREME (e.g. "which department has the highest average salary", "city with highest employee count", "which city has fewest employees")
        grp_ext_m = re.search(r"\b(?:which\s+)?(department|dept|division|city|state|location|office|category|unit|team)\s+(?:has|with|pays|having)?\s*(?:the\s+)?(highest|lowest|most|least|fewest|maximum|minimum|top|bottom|greatest|largest|smallest)\b", q_lower)
        if grp_ext_m:
            dim_word = grp_ext_m.group(1).strip()
            direction_word = grp_ext_m.group(2).strip()
            dim_col = self._resolve_target_column(dim_word, available_columns, schema_dict, trace)
            if dim_col:
                sort_dir = "ASC" if direction_word in {"lowest", "least", "fewest", "minimum", "bottom", "smallest"} else "DESC"
                num_col = None
                if any(w in q_lower for w in ["salary", "pay", "cost", "amount", "revenue", "price", "score", "rating"]):
                    num_col = self._find_numeric_column(q_lower, available_columns, schema_dict, trace)
                    if not num_col:
                        metric_w = self._extract_metric_word(q_lower) or "numeric"
                        return StructuredQuery(
                            operation="UNSUPPORTED_QUERY",
                            missing_column=metric_w,
                            explanation=f"The uploaded dataset does not contain {metric_w} information, so I cannot determine it from the provided data.",
                            no_match_type="COLUMN_NOT_FOUND",
                            debug_trace=trace
                        )
                return StructuredQuery(
                    operation="GROUP_EXTREME",
                    group_by_column=dim_col,
                    target_column=num_col,
                    sort_order=sort_dir,
                    explanation=f"Find {dim_col} with {direction_word} {num_col or 'count'}",
                    debug_trace=trace
                )
            else:
                return StructuredQuery(
                    operation="UNSUPPORTED_QUERY",
                    missing_column=dim_word,
                    explanation=f"The uploaded dataset does not contain {dim_word} information, so I cannot determine it from the provided data.",
                    no_match_type="COLUMN_NOT_FOUND",
                    debug_trace=trace
                )

        # DATE_EXTREME (e.g. "who joined most recently", "who joined earliest", "latest joiner")
        date_ext_words = ["recently", "recent", "latest", "earliest", "first joined", "last joined"]
        if any(w in q_lower for w in date_ext_words):
            date_col = None
            for col in available_columns:
                if any(t in col.lower() for t in ["date", "time", "joining", "joined", "service"]):
                    date_col = col
                    break
            if date_col:
                sort_dir = "DESC" if any(w in q_lower for w in ["recently", "recent", "latest", "last joined"]) else "ASC"
                return StructuredQuery(
                    operation="DATE_EXTREME",
                    target_column=date_col,
                    sort_column=date_col,
                    sort_order=sort_dir,
                    limit=1,
                    explanation=f"Find record with {sort_dir} {date_col}",
                    debug_trace=trace
                )
            else:
                return StructuredQuery(
                    operation="UNSUPPORTED_QUERY",
                    missing_column="date/joining",
                    explanation="The uploaded dataset does not contain date or joining information, so I cannot determine it from the provided data.",
                    no_match_type="COLUMN_NOT_FOUND",
                    debug_trace=trace
                )

        # Ordinal Ranking with rank_offset (e.g. "second highest", "third highest", "2nd highest", "3rd highest")
        ordinal_match = re.search(r"\b(second|2nd|third|3rd|fourth|4th|fifth|5th)\s+(?:highest|lowest|top|bottom|paid|earner|salary)\b", q_lower)
        if ordinal_match:
            ord_word = ordinal_match.group(1).lower()
            ord_map = {"second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4, "fifth": 5, "5th": 5}
            rank_val = ord_map.get(ord_word, 2)
            is_desc = not any(w in q_lower for w in ["lowest", "bottom"])
            num_col = self._find_numeric_column(expanded_q, available_columns, schema_dict, trace)
            if not num_col:
                metric_w = self._extract_metric_word(expanded_q) or "numeric"
                return StructuredQuery(
                    operation="UNSUPPORTED_QUERY",
                    missing_column=metric_w,
                    explanation=f"The uploaded dataset does not contain {metric_w} information, so I cannot determine it from the provided data.",
                    no_match_type="COLUMN_NOT_FOUND",
                    debug_trace=trace
                )
            return StructuredQuery(
                operation="RANK",
                target_column=num_col,
                sort_column=num_col,
                sort_order="DESC" if is_desc else "ASC",
                rank_offset=rank_val,
                limit=1,
                explanation=f"{ord_word.title()} {num_col}",
                debug_trace=trace
            )

        # MEDIAN
        if detected_intent == "MEDIAN":
            num_col = self._find_numeric_column(expanded_q, available_columns, schema_dict, trace)
            if not num_col:
                metric_w = self._extract_metric_word(expanded_q) or "numeric"
                return StructuredQuery(
                    operation="UNSUPPORTED_QUERY",
                    missing_column=metric_w,
                    explanation=f"The uploaded dataset does not contain {metric_w} information, so I cannot determine it from the provided data.",
                    no_match_type="COLUMN_NOT_FOUND",
                    debug_trace=trace
                )
            conditions = self._extract_conditions(expanded_q, available_columns, schema_dict, trace, numeric_cond, date_cond)
            return StructuredQuery(
                operation="MEDIAN",
                target_column=num_col,
                conditions=conditions,
                explanation="Compute median value",
                debug_trace=trace
            )

        # COUNT
        if detected_intent == "COUNT":
            conditions = self._extract_conditions(expanded_q, available_columns, schema_dict, trace, numeric_cond, date_cond)
            return StructuredQuery(
                operation="COUNT",
                conditions=conditions,
                explanation="Count matching records",
                debug_trace=trace
            )

        # SUM
        if detected_intent == "SUM":
            num_col = self._find_numeric_column(expanded_q, available_columns, schema_dict, trace)
            if not num_col:
                metric_w = self._extract_metric_word(expanded_q) or "numeric"
                return StructuredQuery(
                    operation="UNSUPPORTED_QUERY",
                    missing_column=metric_w,
                    explanation=f"The uploaded dataset does not contain {metric_w} information, so I cannot determine it from the provided data.",
                    no_match_type="COLUMN_NOT_FOUND",
                    debug_trace=trace
                )
            conditions = self._extract_conditions(expanded_q, available_columns, schema_dict, trace, numeric_cond, date_cond)
            return StructuredQuery(
                operation="SUM",
                target_column=num_col,
                conditions=conditions,
                explanation="Calculate sum aggregate",
                debug_trace=trace
            )

        # AVERAGE
        if detected_intent == "AVERAGE":
            num_col = self._find_numeric_column(expanded_q, available_columns, schema_dict, trace)
            if not num_col:
                metric_w = self._extract_metric_word(expanded_q) or "numeric"
                return StructuredQuery(
                    operation="UNSUPPORTED_QUERY",
                    missing_column=metric_w,
                    explanation=f"The uploaded dataset does not contain {metric_w} information, so I cannot determine it from the provided data.",
                    no_match_type="COLUMN_NOT_FOUND",
                    debug_trace=trace
                )
            conditions = self._extract_conditions(expanded_q, available_columns, schema_dict, trace, numeric_cond, date_cond)
            return StructuredQuery(
                operation="AVERAGE",
                target_column=num_col,
                conditions=conditions,
                explanation="Calculate average aggregate",
                debug_trace=trace
            )

        # OUTLIER
        if detected_intent == "OUTLIER":
            num_col = self._find_numeric_column(expanded_q, available_columns, schema_dict, trace)
            if not num_col:
                metric_w = self._extract_metric_word(expanded_q) or "numeric"
                return StructuredQuery(
                    operation="UNSUPPORTED_QUERY",
                    missing_column=metric_w,
                    explanation=f"The uploaded dataset does not contain {metric_w} information, so I cannot determine it from the provided data.",
                    no_match_type="COLUMN_NOT_FOUND",
                    debug_trace=trace
                )
            return StructuredQuery(
                operation="OUTLIER",
                target_column=num_col,
                explanation=f"Detect statistical outliers for {num_col or 'numeric column'} via IQR",
                debug_trace=trace
            )

        # RANK
        if detected_intent == "RANK":
            num_col = self._find_numeric_column(expanded_q, available_columns, schema_dict, trace)
            if not num_col:
                metric_w = self._extract_metric_word(expanded_q) or "numeric"
                return StructuredQuery(
                    operation="UNSUPPORTED_QUERY",
                    missing_column=metric_w,
                    explanation=f"The uploaded dataset does not contain {metric_w} information, so I cannot determine it from the provided data.",
                    no_match_type="COLUMN_NOT_FOUND",
                    debug_trace=trace
                )
            return StructuredQuery(
                operation="FILTER",
                sort_column=num_col,
                sort_order="DESC",
                limit=5,
                explanation="Ranked retrieval",
                debug_trace=trace
            )

        # TOP_N / MAX
        if detected_intent in {"TOP_N", "MAX"}:
            num_col = self._find_numeric_column(expanded_q, available_columns, schema_dict, trace)
            if not num_col:
                metric_w = self._extract_metric_word(expanded_q) or "numeric"
                return StructuredQuery(
                    operation="UNSUPPORTED_QUERY",
                    missing_column=metric_w,
                    explanation=f"The uploaded dataset does not contain {metric_w} information, so I cannot determine it from the provided data.",
                    no_match_type="COLUMN_NOT_FOUND",
                    debug_trace=trace
                )
            conditions = self._extract_conditions(expanded_q, available_columns, schema_dict, trace, numeric_cond, date_cond)
            limit = 1
            top_m = re.search(r"\btop\s+(\d+)\b", q_lower)
            if top_m:
                limit = int(top_m.group(1))
            return StructuredQuery(
                operation="MAX" if limit == 1 else "TOP_N",
                target_column=num_col,
                sort_column=num_col,
                sort_order="DESC",
                conditions=conditions,
                limit=limit,
                explanation=f"Top {limit} records",
                debug_trace=trace
            )

        # BOTTOM_N / MIN
        if detected_intent in {"BOTTOM_N", "MIN"}:
            num_col = self._find_numeric_column(expanded_q, available_columns, schema_dict, trace)
            if not num_col:
                metric_w = self._extract_metric_word(expanded_q) or "numeric"
                return StructuredQuery(
                    operation="UNSUPPORTED_QUERY",
                    missing_column=metric_w,
                    explanation=f"The uploaded dataset does not contain {metric_w} information, so I cannot determine it from the provided data.",
                    no_match_type="COLUMN_NOT_FOUND",
                    debug_trace=trace
                )
            conditions = self._extract_conditions(expanded_q, available_columns, schema_dict, trace, numeric_cond, date_cond)
            limit = 1
            bot_m = re.search(r"\b(?:bottom|lowest)\s+(\d+)\b", q_lower)
            if bot_m:
                limit = int(bot_m.group(1))
            return StructuredQuery(
                operation="MIN" if limit == 1 else "BOTTOM_N",
                target_column=num_col,
                sort_column=num_col,
                sort_order="ASC",
                conditions=conditions,
                limit=limit,
                explanation=f"Bottom {limit} records",
                debug_trace=trace
            )

        # GROUP
        if detected_intent == "GROUP":
            group_col = self._resolve_target_column(expanded_q, available_columns, schema_dict, trace)
            num_col = self._find_numeric_column(expanded_q, available_columns, schema_dict, trace)
            return StructuredQuery(
                operation="GROUP",
                group_by_column=group_col,
                target_column=num_col,
                explanation=f"Group by {group_col}",
                debug_trace=trace
            )

        # 4. Check for Semantic Entity-Attribute Relationships (possessive, prepositional, reverse lookup)
        # e.g. "West Bengal's capital", "Rahul Patel's salary", "Which state has Kolkata as its capital?"
        if not numeric_cond and not date_cond:
            rel = semantic_relationship_parser.parse(expanded_q, available_columns, schema_dict)
            if not rel and question != expanded_q:
                rel = semantic_relationship_parser.parse(question, available_columns, schema_dict)

            if rel:
                trace["intent"] = "LOOKUP"
                trace["confidence"] = rel.confidence
                trace["column_mappings"][rel.raw_attribute_text] = {"column": rel.target_col, "confidence": rel.confidence}
                trace["value_mappings"][rel.raw_entity_text] = {"column": rel.subject_col, "resolved_value": rel.subject_val, "confidence": rel.confidence}
                cond = FilterCondition(column=rel.subject_col, operator="=", value=rel.subject_val)
                return StructuredQuery(
                    operation="LOOKUP",
                    conditions=[cond],
                    target_column=rel.target_col,
                    limit=10,
                    explanation=rel.explanation,
                    debug_trace=trace
                )

        # FILTER / MULTI_FILTER / LOOKUP / SEARCH
        conditions = self._extract_conditions(expanded_q, available_columns, schema_dict, trace, numeric_cond, date_cond)

        # Ambiguity check
        for cond in conditions:
            if hasattr(cond, "_ambiguity") and cond._ambiguity:
                trace["is_ambiguous"] = True
                return StructuredQuery(
                    operation="CLARIFICATION",
                    explanation=cond._ambiguity,
                    debug_trace=trace
                )

        return StructuredQuery(
            operation="FILTER",
            conditions=conditions,
            limit=50,
            explanation="Filtered records",
            debug_trace=trace
        )

    def _resolve_target_column(
        self,
        q: str,
        available_columns: List[str],
        schema_dict: Optional[Dict[str, Any]],
        trace: Dict[str, Any]
    ) -> Optional[str]:
        """Find the column mentioned in the query using semantic mapper."""
        words = q.lower().split()
        for i in range(len(words)):
            candidate_chunk = " ".join(words[i:i+3])
            mapping = semantic_column_mapper.map_column(candidate_chunk, available_columns, schema_dict)
            if mapping["column"]:
                trace["column_mappings"][candidate_chunk] = mapping
                return mapping["column"]

        for col in available_columns:
            if col.lower() in q.lower():
                return col
        return None

    def _extract_metric_word(self, q: str) -> Optional[str]:
        """Extract requested metric or numeric concept keyword from question."""
        q_l = q.lower()
        for kw in [
            "salary", "pay", "income", "earning", "compensation", "wage", "money", "cash", "bucks", "dough",
            "performance", "performer", "score", "rating", "gpa", "marks", "grade", "points",
            "price", "cost", "revenue", "amount", "spending", "fee", "mrp", "bill", "value", "expensive", "costliest", "cheapest", "cheap",
            "stock", "quantity", "qty", "inventory", "count", "age", "population", "gdp",
            "mileage", "speed", "distance", "weight", "height", "units", "sales", "profit"
        ]:
            if kw in q_l:
                return kw
        return None

    def _is_column_numeric(self, col: str, schema_dict: Optional[Dict[str, Any]] = None) -> bool:
        """Check if a column is numeric according to schema or active dataframe."""
        from app.dataset.loader import dataset_loader
        cur_schema = schema_dict or dataset_loader.schema_intelligence
        if cur_schema and "columns" in cur_schema and col in cur_schema["columns"]:
            c_info = cur_schema["columns"][col]
            if c_info.get("is_numeric") or c_info.get("type") in ("numeric", "integer", "float"):
                return True
        df = dataset_loader.dataframe
        if df is not None and not df.empty and col in df.columns:
            import pandas as pd
            return bool(pd.api.types.is_numeric_dtype(df[col]))
        return False

    def _find_numeric_column(
        self,
        q: str,
        available_columns: List[str],
        schema_dict: Optional[Dict[str, Any]],
        trace: Dict[str, Any]
    ) -> Optional[str]:
        """Find appropriate numeric column dynamically prioritizing explicit mentions and schema semantics."""
        q_l = q.lower()

        # 1. Direct match: check if any numeric column name is explicitly in question (e.g. "gpa", "salary", "price")
        for col in available_columns:
            if col.lower() in q_l and self._is_column_numeric(col, schema_dict):
                return col

        # 2. Semantic n-gram mapping to available columns
        words = q_l.split()
        for i in range(len(words)):
            candidate_chunk = " ".join(words[i:i+2])
            mapping = semantic_column_mapper.map_column(candidate_chunk, available_columns, schema_dict)
            if mapping.get("column") and self._is_column_numeric(mapping["column"], schema_dict):
                trace["column_mappings"][candidate_chunk] = mapping
                return mapping["column"]

        # 3. Performance / Rating / GPA priority
        if any(w in q_l for w in ["performance", "performer", "score", "rating", "gpa", "marks", "grade", "perf", "well they performed"]):
            for col in available_columns:
                if any(w in col.lower() for w in ["performance", "score", "rating", "gpa", "marks", "grade", "perf"]):
                    return col

        # 4. Salary / Earnings priority
        if any(w in q_l for w in ["salary", "pay", "income", "earning", "earns", "paid", "gets", "getting", "makes", "making", "compensation", "wage", "makes the most", "makes the least", "bucks", "dough", "money", "cash", "bank", "paycheck", "package", "ctc", "lakh", "lac", "crore", "cr"]) or re.search(r"\b\d+\s*(?:l|lac|lacs|lakh|lakhs|cr|crore|crores|k)\b", q_l) or any(c in q_l for c in ["₹", "rs", "$", "€", "£"]):
            for col in available_columns:
                if "salary" in col.lower() or "compensation" in col.lower() or "pay" in col.lower() or "income" in col.lower() or "wage" in col.lower():
                    return col

        # 5. Price / Cost / Amount
        if any(w in q_l for w in ["price", "cost", "amount", "spending", "fee", "mrp", "revenue", "bill", "expensive", "cheapest", "costliest", "cheap"]):
            for col in available_columns:
                if any(w in col.lower() for w in ["price", "amount", "cost", "revenue", "spending"]):
                    return col

        # 6. Stock / Quantity
        if any(w in q_l for w in ["stock", "quantity", "qty", "inventory", "units"]):
            for col in available_columns:
                if any(w in col.lower() for w in ["stock", "quantity", "qty", "units"]):
                    return col

        # 7. Common semantic mappings
        for kw in ["salary", "pay", "income", "amount", "cost", "revenue", "price", "stock", "score", "rating", "performance", "gpa"]:
            if kw in q_l:
                mapping = semantic_column_mapper.map_column(kw, available_columns, schema_dict)
                if mapping.get("column"):
                    trace["column_mappings"][kw] = mapping
                    return mapping["column"]

        # 8. If dataset has exactly one non-ID numeric column, bind to it
        numeric_cols = [
            c for c in available_columns
            if self._is_column_numeric(c, schema_dict) and not any(id_w in c.lower() for id_w in ["id", "code", "zip", "pin", "phone"])
        ]
        if len(numeric_cols) == 1:
            return numeric_cols[0]

        return None

    def _extract_conditions(
        self,
        q: str,
        available_columns: List[str],
        schema_dict: Optional[Dict[str, Any]],
        trace: Dict[str, Any],
        numeric_cond: Optional[Dict[str, Any]],
        date_cond: Optional[Dict[str, Any]]
    ) -> List[FilterCondition]:
        """Extract validated conditions with numeric boundaries and value resolution."""
        conditions: List[FilterCondition] = []

        # High-value pattern
        if "high-value" in q.lower() or "high value" in q.lower():
            num_col = self._find_numeric_column(q, available_columns, schema_dict, trace)
            if num_col:
                conditions.append(FilterCondition(column=num_col, operator=">", value=10000))

        # 1. Strict numeric filter from normalization
        if numeric_cond:
            num_col = self._find_numeric_column(q, available_columns, schema_dict, trace)
            if num_col:
                if numeric_cond.get("type") == "range":
                    conditions.append(FilterCondition(column=num_col, operator=numeric_cond["min_operator"], value=numeric_cond["min_value"]))
                    conditions.append(FilterCondition(column=num_col, operator=numeric_cond["max_operator"], value=numeric_cond["max_value"]))
                else:
                    conditions.append(FilterCondition(column=num_col, operator=numeric_cond["operator"], value=numeric_cond["value"]))

        # 2. Date filter from normalization
        if date_cond:
            date_col = None
            for col in available_columns:
                if any(t in col.lower() for t in ["date", "time", "joining", "expiry"]):
                    date_col = col
                    break
            if date_col:
                if date_cond.get("type") == "range":
                    conditions.append(FilterCondition(column=date_col, operator=date_cond["min_operator"], value=date_cond["min_value"]))
                    conditions.append(FilterCondition(column=date_col, operator=date_cond["max_operator"], value=date_cond["max_value"]))
                else:
                    conditions.append(FilterCondition(column=date_col, operator=date_cond["operator"], value=date_cond["value"]))

        # 3. Semantic entity-attribute relationship check
        rel = semantic_relationship_parser.parse(q, available_columns, schema_dict)
        if rel:
            trace["value_mappings"][rel.raw_entity_text] = {
                "column": rel.subject_col,
                "resolved_value": rel.subject_val,
                "confidence": rel.confidence
            }
            conditions.append(FilterCondition(column=rel.subject_col, operator="=", value=rel.subject_val))
            return conditions

        # 3.5. Starts with / begins with filter: e.g. "Which states have a capital beginning with Z?"
        sw_m = re.search(r"\b(?:starting\s+with|beginning\s+with|starts\s+with|begins\s+with)\s+['\"]?([a-zA-Z0-9])['\"]?\b", q.lower())
        if sw_m:
            target_char = sw_m.group(1).upper()
            target_col = self._resolve_target_column(q, available_columns, schema_dict, trace)
            if not target_col:
                for c in available_columns:
                    if c.lower() in q.lower():
                        target_col = c
                        break
            if not target_col and available_columns:
                target_col = available_columns[0]
            if target_col:
                cond = FilterCondition(column=target_col, operator="starts_with", value=target_char)
                conditions.append(cond)
                return conditions

        # 4. Check ID / Code columns first (e.g. Employee ID: "EMP-1001", "EMP 1001", "emp1001")
        for col in available_columns:
            if any(term in col.lower() for term in ["id", "code"]):
                vals = dataset_indexer.get_column_values(col)
                for v in vals:
                    if not v:
                        continue
                    v_str = str(v).strip()
                    v_pats = [
                        r"\b" + re.escape(v_str) + r"\b",
                        r"\b" + re.escape(v_str.replace("-", " ")) + r"\b",
                        r"\b" + re.escape(v_str.replace("-", "")) + r"\b"
                    ]
                    if any(re.search(p, q, re.IGNORECASE) for p in v_pats):
                        conditions.append(FilterCondition(column=col, operator="=", value=v_str))
                        trace["value_mappings"][v_str] = {"column": col, "resolved_value": v_str, "confidence": 1.0}
                        return conditions

        # 5. Entity resolution across categorical / geographic / entity columns
        matched_cols = set()
        matched_tokens = set()
        for col in available_columns:
            if any(term in col.lower() for term in ["id", "amount", "price", "salary", "cost", "date"]):
                continue

            vals = dataset_indexer.get_column_values(col)
            found_token = extract_best_match_from_text(q, [str(v) for v in vals if v])
            if found_token and found_token.lower() not in matched_tokens:
                # Anti-collusion guard: do not match tokens that overlap with already matched multi-word tokens
                if any(found_token.lower() in mt for mt in matched_tokens):
                    continue

                res = dataset_value_resolver.resolve_value_in_column(found_token, col)
                trace["value_mappings"][found_token] = res
                matched_cols.add(col.lower())
                matched_tokens.add(found_token.lower())
                for w in found_token.lower().split():
                    matched_tokens.add(w)

                if res.get("is_ambiguous"):
                    cond = FilterCondition(column=col, operator="=", value=res.get("resolved_value") or found_token)
                    setattr(cond, "_ambiguity", res.get("clarification_prompt"))
                    conditions.append(cond)
                elif res.get("resolved_value"):
                    conditions.append(FilterCondition(column=col, operator="=", value=res["resolved_value"]))

        # 7. Unmatched entity guard for categorical dimensions (e.g. "How many employees are in IT?")
        # Only apply if no condition was already found
        if not conditions:
            dept_col = None
            for c in available_columns:
                if any(term in c.lower() for term in ["department", "dept", "division"]):
                    dept_col = c
                    break
            if dept_col:
                m_dept = re.search(r"\b(?:in|from|for|department|dept)\s+([a-zA-Z]+)\b", q.lower())
                if m_dept:
                    token = m_dept.group(1).strip()
                    STOP_TOKENS = {
                        "the", "a", "an", "all", "each", "every", "this", "that", "most", "least", "salary", "score",
                        "does", "is", "are", "has", "have", "do", "did", "was", "were", "work", "works", "live", "lives", "work in"
                    }
                    if token not in STOP_TOKENS:
                        val = token.upper() if len(token) <= 3 else token.title()
                        conditions.append(FilterCondition(column=dept_col, operator="=", value=val))

        # 8. Unmatched lookup guard (e.g. "Who is Bruce Wayne?", "Tell me about Clark Kent", "Information about John Doe")
        if not conditions:
            lookup_patterns = ["who is ", "tell me about ", "details for ", "show record for ", "information about ", "info for ", "find employee "]
            if any(q.lower().startswith(p) for p in lookup_patterns):
                m_lookup = re.search(r"(?:who is|tell me about|details for|show record for|information about|info for|find employee)\s+([a-zA-Z\s]+)", q, re.IGNORECASE)
                if m_lookup:
                    cand_name = m_lookup.group(1).strip().rstrip("?").strip()
                    # Filter out non-names like 'the best performer', 'the highest paid'
                    if not any(cand_name.lower().startswith(p) for p in ["the ", "a ", "an ", "most ", "least ", "best ", "worst "]) and not any(w in cand_name.lower() for w in ["performer", "salary", "earner", "paid", "highest", "lowest", "rockstar"]):
                        name_col = None
                        for c in available_columns:
                            if any(term in c.lower() for term in ["name", "employee", "customer"]):
                                name_col = c
                                break
                        if name_col and cand_name:
                            conditions.append(FilterCondition(column=name_col, operator="=", value=cand_name.title()))

        return conditions


query_planner = QueryPlanner()
