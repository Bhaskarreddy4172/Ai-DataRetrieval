"""Prompt templates for Dynamic Dataset Query Understanding and Grounded Response Generation."""

import json
from typing import Any, Dict, List, Optional


def build_query_understanding_prompt(
    question: str,
    schema: Dict[str, Any],
    recent_history: Optional[List[Dict[str, Any]]] = None
) -> str:
    """Construct LLM prompt with dataset schema and user question."""
    schema_json = json.dumps(schema, indent=2)

    history_str = ""
    if recent_history:
        history_items = [f"- User: {h.get('question')}" for h in recent_history[-2:]]
        history_str = "\nRecent Conversation:\n" + "\n".join(history_items) + "\n"

    return f"""You are a precise data query translator.
Convert the user's question into a structured JSON query based strictly on the available dataset schema.

AVAILABLE DATASET SCHEMA:
{schema_json}
{history_str}
USER QUESTION:
"{question}"

SUPPORTED OPERATIONS:
- "FILTER": Filter records matching conditions.
- "LOOKUP": Direct single-entity retrieval.
- "COUNT": Count number of matching records.
- "SUM": Sum values of a numeric column.
- "AVERAGE": Average values of a numeric column.
- "MIN": Minimum value in a column.
- "MAX": Maximum value in a column.
- "SORT": Sort records by column (order: "ASC" or "DESC").
- "GROUP": Group by a category column and summarize.
- "UNKNOWN": The user is asking for information or columns NOT in the dataset.
- "AMBIGUOUS": The user question is too vague and needs clarification.

CRITICAL INSTRUCTIONS:
1. Output ONLY a valid JSON object matching this exact structure:
{{
  "operation": "FILTER" | "LOOKUP" | "COUNT" | "SUM" | "AVERAGE" | "MIN" | "MAX" | "SORT" | "GROUP" | "UNKNOWN" | "AMBIGUOUS",
  "conditions": [
    {{
      "column": "ExactColumnName",
      "operator": "=" | "!=" | ">" | "<" | ">=" | "<=" | "contains" | "starts_with" | "ends_with",
      "value": "string or number"
    }}
  ],
  "logical_operator": "AND" | "OR",
  "target_column": "ColumnNameToAggregateOrRetrieve" or null,
  "group_by_column": "ColumnToGroupBy" or null,
  "select_columns": ["Col1", "Col2"] or null,
  "sort_column": "SortColumn" or null,
  "sort_order": "ASC" or "DESC",
  "limit": 50,
  "missing_column": "RequestedMissingColumnName" or null,
  "explanation": "Brief description of the query"
}}
2. Match the user's phrasing to the actual column names in the schema (e.g. "spent over 10000" -> Service Amount > 10000).
3. If the user asks for a topic or column completely missing from the schema (e.g. asking for population or age when not in schema), set "operation": "UNKNOWN" and "missing_column" to that attribute.
4. Return ONLY the JSON object. Do not add markdown commentary outside the JSON.
"""


def build_grounded_response_prompt(
    question: str,
    operation: str,
    results: List[Dict[str, Any]],
    aggregation: Optional[Dict[str, Any]] = None,
    missing_column: Optional[str] = None
) -> str:
    """Prompt for generating concise, grounded natural-language answer."""
    if operation == "UNKNOWN":
        col_msg = f" '{missing_column}'" if missing_column else ""
        return f"""User Question: "{question}"
The dataset does not contain{col_msg} information.
Politely inform the user that the dataset does not contain this information."""

    agg_str = f"Calculated Result: {json.dumps(aggregation)}" if aggregation else "No aggregation"
    records_str = json.dumps(results[:15], indent=2) if results else "[]"

    return f"""You are a helpful data analyst.
Answer the user's question concisely using ONLY the retrieved dataset results below.

USER QUESTION:
"{question}"

OPERATION:
{operation}

{agg_str}

RETRIEVED RECORDS (Total {len(results)} found):
{records_str}

RULES:
- Base your answer strictly and exclusively on the retrieved data.
- Do NOT invent records or numbers.
- State the numbers and entities clearly.
- If no matching records were found, clearly state that no matching records were found in the dataset.
- Keep the response concise, professional, and clear.
"""
