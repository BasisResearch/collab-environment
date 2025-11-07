"""
Widget registry for discovering and instantiating analysis widgets.

Loads widget configuration from YAML and manages widget lifecycle.
"""

import importlib
import logging
from pathlib import Path
from typing import List, Type, Dict, Any, Optional

import yaml

from .base_analysis_widget import BaseAnalysisWidget

logger = logging.getLogger(__name__)


class WidgetRegistry:
    """
    Manages available analysis widgets.

    Discovers widgets from YAML configuration and instantiates them
    on demand. Supports enabling/disabling widgets and ordering.

    Examples
    --------
    >>> registry = WidgetRegistry("analysis_widgets.yaml")
    >>> widgets = registry.get_enabled_widgets()
    >>> for widget in widgets:
    ...     print(f"Loaded: {widget.widget_name}")
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize registry.

        Parameters
        ----------
        config_path : str, optional
            Path to YAML config file. If None, uses default location.
        """
        self.widgets: Dict[str, Type[BaseAnalysisWidget]] = {}
        self.config: Dict[str, Any] = {}

        if config_path:
            self.load_from_config(config_path)

    def register(self, widget_class: Type[BaseAnalysisWidget], name: Optional[str] = None):
        """
        Register widget programmatically.

        Parameters
        ----------
        widget_class : Type[BaseAnalysisWidget]
            Widget class to register
        name : str, optional
            Registration name (defaults to class name)
        """
        widget_name = name or widget_class.__name__
        self.widgets[widget_name] = widget_class
        logger.info(f"Registered widget: {widget_name}")

    def load_from_config(self, config_path: str):
        """
        Load widgets from YAML configuration.

        Parameters
        ----------
        config_path : str
            Path to YAML config file

        Examples
        --------
        Config file format:

        ```yaml
        defaults:
          spatial_bin_size: 10.0

        widgets:
          - class: collab_env.dashboard.widgets.heatmap_widget.HeatmapWidget
            enabled: true
            order: 1
            category: spatial
        ```
        """
        config_file = Path(config_path)

        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}")
            return

        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)

        if not self.config:
            logger.warning(f"Empty config file: {config_path}")
            return

        # Load widget classes
        widget_configs = self.config.get('widgets', [])
        for widget_config in widget_configs:
            class_path = widget_config.get('class')
            if not class_path:
                logger.warning(f"Widget config missing 'class' field: {widget_config}")
                continue

            try:
                widget_class = self._import_class(class_path)
                self.register(widget_class, name=class_path)
                logger.info(f"Loaded widget from config: {class_path}")

            except Exception as e:
                logger.error(f"Failed to load widget {class_path}: {e}")

    def _import_class(self, class_path: str) -> Type[BaseAnalysisWidget]:
        """
        Import class from module path.

        Parameters
        ----------
        class_path : str
            Full module path (e.g., "package.module.ClassName")

        Returns
        -------
        Type[BaseAnalysisWidget]
            Imported widget class

        Raises
        ------
        ImportError
            If module or class cannot be imported
        TypeError
            If class is not a BaseAnalysisWidget subclass
        """
        # Split module and class name
        module_path, class_name = class_path.rsplit('.', 1)

        # Import module
        module = importlib.import_module(module_path)

        # Get class
        widget_class = getattr(module, class_name)

        # Validate it's a widget
        if not issubclass(widget_class, BaseAnalysisWidget):
            raise TypeError(f"{class_path} is not a BaseAnalysisWidget subclass")

        return widget_class

    def get_enabled_widgets(self) -> List[BaseAnalysisWidget]:
        """
        Get instantiated widgets that are enabled in config.

        Returns widgets sorted by order field in config.

        Returns
        -------
        list of BaseAnalysisWidget
            Instantiated enabled widgets

        Examples
        --------
        >>> registry = WidgetRegistry("widgets.yaml")
        >>> widgets = registry.get_enabled_widgets()
        >>> for w in widgets:
        ...     print(f"{w.widget_name}: {w.widget_description}")
        """
        enabled_widgets = []
        widget_configs = self.config.get('widgets', [])

        for widget_config in widget_configs:
            # Check if enabled
            if not widget_config.get('enabled', True):
                continue

            class_path = widget_config.get('class')
            if class_path not in self.widgets:
                logger.warning(f"Widget not registered: {class_path}")
                continue

            # Instantiate widget
            try:
                widget_class = self.widgets[class_path]
                widget = widget_class()

                # Store metadata
                widget._config_order = widget_config.get('order', 999)
                widget._config_category = widget_config.get('category', widget.widget_category)

                enabled_widgets.append(widget)

            except Exception as e:
                logger.error(f"Failed to instantiate {class_path}: {e}")

        # Sort by order
        enabled_widgets.sort(key=lambda w: w._config_order)

        logger.info(f"Enabled {len(enabled_widgets)} widgets")
        return enabled_widgets

    def get_defaults(self) -> Dict[str, Any]:
        """
        Get default parameter values from config.

        Returns
        -------
        dict
            Default parameter values

        Examples
        --------
        >>> defaults = registry.get_defaults()
        >>> spatial_bin_size = defaults.get('spatial_bin_size', 10.0)
        """
        return self.config.get('defaults', {})
