"""Grounded response generator preventing hallucination and reporting verified dataset facts."""

from typing import Any, Dict, List, Optional
from app.ai.ollama_client import ollama_client, OllamaClient
from app.ai.prompts import build_grounded_response_prompt
from app.utils.logger import logger


class ResponseGenerator:
    """Produces verified, grounded natural language responses strictly based on retrieved records."""

    def __init__(self, ai_client: OllamaClient = ollama_client):
        self.client = ai_client

    def generate_hybrid_response(
        self,
        dataset_results: List[Dict[str, Any]],
        general_knowledge_answer: str,
        question: str,
        city_or_entity: Optional[str] = None
    ) -> str:
        """Combine verified dataset facts with verified general knowledge facts."""
        gk_part = general_knowledge_answer.strip()

        if not dataset_results:
            target = f" in {city_or_entity}" if city_or_entity else ""
            return f"No matching records were found in the dataset{target}. Note that {gk_part}"

        # Extract entity names from dataset results
        name_keys = ["Employee Name", "Customer Name", "Product Name", "name", "state", "Student Name"]
        names = []
        for r in dataset_results:
            for nk in name_keys:
                if nk in r and r[nk]:
                    names.append(str(r[nk]))
                    break

        if names:
            names_str = ", ".join(names[:5])
            suffix = f" and {len(names) - 5} more" if len(names) > 5 else ""
            return f"{names_str}{suffix} work/reside in {city_or_entity or 'the area'}. {gk_part}"

        return f"Found {len(dataset_results)} matching record(s) in the dataset. {gk_part}"

    def generate_grounded_fallback(
        self,
        question: str,
        operation: str,
        results: List[Dict[str, Any]],
        aggregation: Optional[Dict[str, Any]] = None,
        missing_column: Optional[str] = None,
        missing_entity: Optional[str] = None,
        explanation: Optional[str] = None
    ) -> str:
        """Deterministic Python response generation when Ollama is offline or for guaranteed precision."""
        if operation == "NO_MATCH":
            if explanation:
                return explanation
            ent = missing_entity or "the requested entity"
            return f'No matching state or entity was found in the uploaded dataset for "{ent}".'

        if operation in {"UNSUPPORTED_QUERY", "UNKNOWN"}:
            if explanation:
                return explanation
            col_msg = missing_column or "the requested"
            return f"The uploaded dataset does not contain {col_msg} information, so I cannot determine it from the provided data."

        if operation == "NULL_VALUE":
            if explanation:
                return explanation
            ent = missing_entity or "The record"
            attr = missing_column or "requested"
            return f"{ent} was found in the dataset, but its {attr} value is not available."

        if operation in {"AMBIGUOUS", "CLARIFICATION"}:
            return explanation or "Could you specify which state you mean?"

        if operation == "DUPLICATES":
            dup_count = len(results)
            if dup_count == 0:
                return "No duplicate records were found in the dataset."
            return f"Found {dup_count} duplicate record(s) matching your criteria."

        if operation == "EXISTS":
            if results and len(results) > 0:
                return f"Yes, matching records exist in the dataset ({len(results)} found)."
            return "No, no matching records exist in the dataset for that inquiry."

        if operation == "DISTINCT":
            if results:
                distinct_vals = [str(r.get(list(r.keys())[0])) for r in results[:15] if r]
                return f"Found {len(results)} distinct values: {', '.join(distinct_vals)}."
            return "No distinct values found."

        # If MAX or MIN has retrieved entity records, report the entity AND the value!
        if aggregation and aggregation.get("metric") in {"MAX", "MIN"} and results:
            val = aggregation.get("value", "")
            col = aggregation.get("column", "")
            metric = aggregation.get("metric", "")
            is_max = (metric == "MAX")
            direction_str = "highest" if is_max else "lowest"

            name_keys = ["Employee Name", "Customer Name", "Product Name", "name", "state", "Student Name"]
            r = results[0]
            entity_name = None
            for nk in name_keys:
                if nk in r and r[nk]:
                    entity_name = str(r[nk])
                    break

            fmt_val = f"₹{val:,.2f}" if isinstance(val, (int, float)) and val > 1000 else f"{val}"

            if entity_name:
                extra_details = []
                if "Department" in r:
                    extra_details.append(f"Department: {r['Department']}")
                if "City" in r:
                    extra_details.append(f"City: {r['City']}")
                if "Designation" in r:
                    extra_details.append(f"Designation: {r['Designation']}")

                detail_str = f" ({', '.join(extra_details)})" if extra_details else ""
                if len(results) == 1:
                    return f"{entity_name}{detail_str} has the {direction_str} {col or 'value'} of {fmt_val}."
                else:
                    all_names = [str(row.get(nk, "")) for row in results for nk in name_keys if nk in row]
                    return f"{', '.join(all_names)} share the {direction_str} {col or 'value'} of {fmt_val}."

            return f"The {direction_str} {col or 'value'} is {fmt_val}."

        if aggregation:
            metric = aggregation.get("metric", "")
            val = aggregation.get("value", "")
            col = aggregation.get("column", "")
            if metric == "COUNT":
                return f"There are {val} matching records in the dataset."
            elif metric == "SUM":
                return f"The total {col or 'amount'} is ₹{val:,.2f}." if isinstance(val, (int, float)) else f"The total is {val}."
            elif metric in {"AVERAGE", "AVG", "MEAN"}:
                return f"The average {col or 'amount'} is ₹{val:,.2f}." if isinstance(val, (int, float)) else f"The average is {val}."
            elif metric == "MEDIAN":
                return f"The median {col or 'amount'} is ₹{val:,.2f}." if isinstance(val, (int, float)) else f"The median is {val}."
            elif metric == "MAX":
                return f"The highest {col or 'value'} is {val}."
            elif metric == "MIN":
                return f"The lowest {col or 'value'} is {val}."

        if not results:
            if explanation:
                return explanation
            return "No matching records were found in the dataset for your query."

        # Filtered / Tabular responses
        if len(results) == 1:
            r = results[0]
            # If customer dataset
            if "Customer Name" in r and "Service Amount" in r:
                return f"{r['Customer Name']} (City: {r.get('City', '-')}) spent ₹{r['Service Amount']:,} on {r.get('Vehicle', 'service')}."
            # If employee dataset
            if "Employee Name" in r and "Salary" in r:
                return f"{r['Employee Name']} is a {r.get('Designation', 'Employee')} in {r.get('Department', 'General')} with a salary of ₹{r['Salary']:,}."
            # If products dataset
            if "Product Name" in r and "Price" in r:
                return f"{r['Product Name']} ({r.get('Category', '-')}) is priced at ₹{r['Price']:,} with {r.get('Stock Quantity', 0)} in stock."
            # If state / capital
            if "state" in r and "capital" in r:
                q_l = question.lower()
                if any(w in q_l for w in ["which state has", "which state does", "to which state", "belongs to which state", "capital of which state", "in which state"]):
                    return f"{r['capital']} is the capital of {r['state']}."
                return f"The capital of {r['state']} is {r['capital']}."
            # Generic single row
            items = [f"{k}: {v}" for k, v in r.items() if not k.startswith("_")][:4]
            return f"Found 1 record: {', '.join(items)}."

        # Multiple records
        if len(results) > 1:
            name_keys = ["Customer Name", "Employee Name", "Product Name", "state", "Student Name"]
            for nk in name_keys:
                if nk in results[0]:
                    names = [str(r[nk]) for r in results[:4] if nk in r]
                    return f"Found {len(results)} matching records, including {', '.join(names)}."
            return f"Found {len(results)} matching records in the dataset."

        return "Query executed successfully."

    def generate(
        self,
        question: str,
        operation: str,
        results: List[Dict[str, Any]],
        aggregation: Optional[Dict[str, Any]] = None,
        missing_column: Optional[str] = None,
        missing_entity: Optional[str] = None,
        explanation: Optional[str] = None
    ) -> str:
        """Generate response using Ollama if online; otherwise deterministic fallback."""
        if operation in {"UNKNOWN", "UNSUPPORTED_QUERY", "NO_MATCH", "NULL_VALUE", "AMBIGUOUS", "CLARIFICATION"}:
            return self.generate_grounded_fallback(
                question, operation, results, aggregation, missing_column, missing_entity, explanation
            )

        health = self.client.check_health()
        if health.get("available"):
            prompt = build_grounded_response_prompt(question, operation, results, aggregation, missing_column)
            ai_output = self.client.generate(prompt=prompt)
            if ai_output:
                logger.info("Grounded response generated via Ollama.")
                return ai_output

        return self.generate_grounded_fallback(question, operation, results, aggregation, missing_column, missing_entity, explanation)


response_generator = ResponseGenerator()
