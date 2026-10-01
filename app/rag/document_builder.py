"""RAG Document Builder: Generates structured knowledge documents for schema, columns, entities, and context."""

from typing import Any, Dict, List, Optional
from app.database.repositories import (
    DatasetRepository, EntityRepository, StateVillageRepository
)
from app.utils.logger import logger


class RAGDocumentBuilder:
    """Builds semantic RAG chunks from the database catalog and ingested data."""

    def __init__(self):
        self.dataset_repo = DatasetRepository()
        self.entity_repo = EntityRepository()
        self.state_village_repo = StateVillageRepository()

    def build_all_documents(self) -> List[Dict[str, Any]]:
        """Construct all semantic RAG documents for indexing."""
        documents = []

        # 1. Dataset Schema Documents
        datasets = self.dataset_repo.list_datasets()
        for ds in datasets:
            ds_id = ds["dataset_id"]
            ds_type = ds.get("dataset_type", "DATASET")
            p_entity = ds.get("parent_entity") or ""
            c_entity = ds.get("child_entity") or ""
            rows = ds.get("row_count", 0)

            content = (
                f"Dataset: '{ds['dataset_name']}' (ID: {ds_id}). Type: {ds_type}. "
                f"Contains {rows} records. Primary entities: Parent={p_entity}, Child={c_entity}. "
                f"File format: {ds.get('file_type')}."
            )
            documents.append({
                "document_id": f"schema_{ds_id}",
                "dataset_id": ds_id,
                "document_type": "DATASET_SCHEMA",
                "content": content,
                "metadata": {
                    "dataset_id": ds_id,
                    "dataset_name": ds["dataset_name"],
                    "dataset_type": ds_type,
                    "row_count": rows,
                }
            })

            # 2. Column Description Documents
            cols = self.dataset_repo.get_columns(ds_id)
            for c in cols:
                orig = c["original_name"]
                norm = c["normalized_name"]
                sem = c["semantic_type"]
                dtype = c["data_type"]
                is_num = c.get("is_numeric", False)
                is_agg = c.get("is_aggregatable", False)

                col_content = (
                    f"Column '{orig}' (normalized: '{norm}') in dataset '{ds_id}'. "
                    f"Semantic Type: {sem}. Storage Data Type: {dtype}. "
                    f"Is Numeric: {is_num}. Aggregatable: {is_agg}. "
                    f"Supports mathematical operations: {is_agg}."
                )
                documents.append({
                    "document_id": f"col_{ds_id}_{norm}",
                    "dataset_id": ds_id,
                    "document_type": "COLUMN_DESCRIPTION",
                    "content": col_content,
                    "metadata": {
                        "dataset_id": ds_id,
                        "column_name": orig,
                        "normalized_name": norm,
                        "semantic_type": sem,
                        "is_aggregatable": is_agg,
                    }
                })

        # 3. Entity Documents & Parent-Child Relationships
        entities = self.entity_repo.search_entities(query="", limit=1000)
        for ent in entities:
            e_name = ent["name"]
            e_type = ent["entity_type"]
            p_id = ent.get("parent_id")
            ent_content = f"Entity: '{e_name}' is a recognized {e_type} in the Indian administrative dataset."
            if p_id:
                ent_content += f" Parent entity ID: {p_id}."

            documents.append({
                "document_id": f"ent_{ent['entity_id']}",
                "dataset_id": ent.get("dataset_id"),
                "document_type": "ENTITY",
                "content": ent_content,
                "metadata": {
                    "entity_id": ent["entity_id"],
                    "name": e_name,
                    "entity_type": e_type,
                    "parent_id": p_id,
                }
            })

        # 4. Entity Aliases Documents
        aliases = self.entity_repo.list_aliases(limit=1000)
        for al in aliases:
            a_name = al["alias"]
            a_type = al["alias_type"]
            canonical = al["canonical_name"]
            e_type = al["entity_type"]

            alias_content = (
                f"Alias: '{a_name}' refers to canonical {e_type} '{canonical}' "
                f"(Alias Type: {a_type}). Questions mentioning '{a_name}' refer to '{canonical}'."
            )
            documents.append({
                "document_id": f"alias_{al['alias_id']}",
                "dataset_id": None,
                "document_type": "ENTITY_ALIAS",
                "content": alias_content,
                "metadata": {
                    "alias": a_name,
                    "canonical_name": canonical,
                    "entity_type": e_type,
                    "alias_type": a_type,
                }
            })

        # 5. Row Context & State Summaries
        state_rows = self.state_village_repo.get_all_states()
        for st in state_rows:
            s_name = st["state"]
            s_cap = st["capital"]
            s_code = st.get("state_code", "")

            # Aggregations for this state across villages
            summary = self.state_village_repo.get_state_aggregates(s_name)
            v_count = summary.get("village_count", 0)
            tot_pop = summary.get("total_population", 0)
            avg_lit = summary.get("avg_literacy", 0)
            tot_area = summary.get("total_area", 0)

            context_content = (
                f"State: {s_name}. Capital: {s_cap}. State Code: {s_code}. "
                f"Recorded Villages: {v_count}. Total Population across villages: {tot_pop:,.0f}. "
                f"Average Literacy Rate: {avg_lit:.2f}%. Total Area: {tot_area:,.2f} sq km."
            )
            documents.append({
                "document_id": f"summary_state_{s_name.lower().replace(' ', '_')}",
                "dataset_id": "india_states_capitals_main",
                "document_type": "DATASET_SUMMARY",
                "content": context_content,
                "metadata": {
                    "state": s_name,
                    "capital": s_cap,
                    "state_code": s_code,
                    "village_count": v_count,
                    "total_population": tot_pop,
                    "avg_literacy": avg_lit,
                    "total_area": tot_area,
                }
            })

        logger.info(f"RAG Document Builder: Generated {len(documents)} structured documents.")
        return documents


rag_document_builder = RAGDocumentBuilder()
