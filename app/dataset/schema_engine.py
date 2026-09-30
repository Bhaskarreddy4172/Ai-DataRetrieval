"Dataset schema engine providing dynamic schema generation and intelligence."

from app.dataset.schema import schema_intelligence, SchemaIntelligence
from app.dataset.semantic_mapper import semantic_column_mapper

class SchemaEngine(SchemaIntelligence):
    "Schema engine with dynamic descriptions and semantic type classification."
    pass

schema_engine = schema_intelligence
