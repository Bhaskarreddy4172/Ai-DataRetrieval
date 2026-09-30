"""Central Tool Package & Default Registration."""

from app.tools.base import BaseTool
from app.tools.registry import tool_registry, ToolRegistry
from app.tools.dataset_search import SearchDatasetTool
from app.tools.dataset_filter import FilterDatasetTool
from app.tools.dataset_aggregate import AggregateDatasetTool
from app.tools.dataset_sort import SortDatasetTool, TopNTool, BottomNTool
from app.tools.dataset_group import GroupByTool
from app.tools.dataset_compare import CompareTool
from app.tools.dataset_lookup import LookupTool
from app.tools.dataset_statistics import StatisticsTool
from app.tools.dataset_schema import SchemaInspectionTool
from app.tools.dataset_relationship import RelationshipTool

# Register default deterministic dataset tools
tool_registry.register(SearchDatasetTool())
tool_registry.register(FilterDatasetTool())
tool_registry.register(AggregateDatasetTool())
tool_registry.register(SortDatasetTool())
tool_registry.register(TopNTool())
tool_registry.register(BottomNTool())
tool_registry.register(GroupByTool())
tool_registry.register(CompareTool())
tool_registry.register(LookupTool())
tool_registry.register(StatisticsTool())
tool_registry.register(SchemaInspectionTool())
tool_registry.register(RelationshipTool())

__all__ = [
    "BaseTool",
    "ToolRegistry",
    "tool_registry",
    "SearchDatasetTool",
    "FilterDatasetTool",
    "AggregateDatasetTool",
    "SortDatasetTool",
    "TopNTool",
    "BottomNTool",
    "GroupByTool",
    "CompareTool",
    "LookupTool",
    "StatisticsTool",
    "SchemaInspectionTool",
    "RelationshipTool",
]

