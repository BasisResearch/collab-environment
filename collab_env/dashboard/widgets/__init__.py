"""
Analysis widgets for spatial analysis dashboard.

This package provides a modular plugin architecture for analysis widgets.
Each widget is self-contained and registered via YAML configuration.
"""

from .query_scope import QueryScope, ScopeType
from .analysis_context import AnalysisContext
from .base_analysis_widget import BaseAnalysisWidget
from .widget_registry import WidgetRegistry

__all__ = [
    "QueryScope",
    "ScopeType",
    "AnalysisContext",
    "BaseAnalysisWidget",
    "WidgetRegistry",
]
