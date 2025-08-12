"""
Main dashboard application using Panel/HoloViz.
"""

import panel as pn
import param
from typing import Dict, Optional, List, Any
from pathlib import Path
import traceback
from loguru import logger

from .rclone_client import RcloneClient
from .session_manager import SessionManager, SessionInfo
from .file_viewers import FileContentManager

# Enable Panel extensions
pn.extension("tabulator")

# No custom CSS - using only Panel widgets


class DataDashboard(param.Parameterized):
    """Main dashboard application for browsing GCS data."""

    # Parameters for reactive UI
    selected_session = param.String(default="", doc="Currently selected session")
    selected_file = param.String(default="", doc="Currently selected file")
    current_bucket_type = param.String(
        default="curated", doc="Current bucket type view"
    )

    def __init__(self, **params):
        super().__init__(**params)

        # Initialize clients
        try:
            self.rclone_client = RcloneClient()
            self.session_manager = SessionManager(self.rclone_client)
            self.file_manager = FileContentManager(self.rclone_client)
        except Exception as e:
            logger.error(f"Failed to initialize dashboard: {e}")
            raise

        # UI state
        self.sessions: Dict[str, SessionInfo] = {}
        self.current_session_files: List[Dict] = []
        self.current_file_content: Optional[bytes] = None
        self.current_file_info: Dict[str, Any] = {}

        # Mapping from display names to actual file paths
        self.display_to_path_map: Dict[str, str] = {}

        # UI components
        self.session_select = pn.widgets.Select(name="Session", options=[], width=300)

        self.bucket_type_toggle = pn.widgets.RadioButtonGroup(
            name="Data Type",
            options=["curated", "processed"],
            value="curated",
            button_type="success",
            visible=False,  # Hidden initially until session selected
        )

        # File tree with wider width for better visibility
        self.file_tree = pn.widgets.Select(
            name="Files",
            options=[],
            size=12,  # Show files at once
            width=550,  # Much wider for long file names
            height=300,  # Reduced height
            visible=False,  # Hidden initially until session selected
        )

        # File name header for preview panel
        self.file_name_header = pn.pane.HTML(
            "<h3>No file selected</h3>", margin=(0, 0, 10, 0)
        )

        self.file_viewer = pn.pane.HTML(
            "<p>Select a file to view its contents</p>",
            width=900,
            height=400,
            sizing_mode="fixed",
        )

        self.file_editor = pn.widgets.TextAreaInput(
            width=900,
            height=300,
            sizing_mode="fixed",
            placeholder="File content will appear here when editing...",
        )

        self.edit_mode = False
        self.edit_button = pn.widgets.Button(
            name="Edit File", button_type="primary", width=100, disabled=True
        )

        self.save_button = pn.widgets.Button(
            name="Save Changes", button_type="success", width=100, disabled=True
        )

        self.cancel_edit_button = pn.widgets.Button(
            name="Cancel Edit", button_type="light", width=100, disabled=True
        )

        self.status_pane = pn.pane.HTML("<p>Ready</p>", width=800, height=30)

        # Loading indicator using Panel's built-in loading
        self.loading_indicator = pn.indicators.LoadingSpinner(
            value=False,
            visible=False,
            size=200,
            align="center",
            name="Loading files...",
        )

        # Cache management buttons
        self.clear_cache_button = pn.widgets.Button(name="Clear Cache", width=100)
        self.cache_info_pane = pn.pane.HTML("", width=300)

        # Wire up callbacks
        self.session_select.param.watch(self._on_session_change, "value")
        self.bucket_type_toggle.param.watch(self._on_bucket_type_change, "value")
        self.file_tree.param.watch(self._on_file_select, "value")
        self.edit_button.on_click(self._start_edit)
        self.save_button.on_click(self._save_edit)
        self.cancel_edit_button.on_click(self._cancel_edit)
        self.clear_cache_button.on_click(self._clear_cache)

        # Load initial data
        self._load_sessions()
        self._update_cache_info()

    def _load_sessions(self):
        """Load available sessions."""
        try:
            self.status_pane.object = "<p>Loading sessions...</p>"
            self.sessions = self.session_manager.discover_sessions()

            session_options = [""] + sorted(self.sessions.keys())
            self.session_select.options = session_options

            total_sessions = len(self.sessions)
            complete_sessions = sum(1 for s in self.sessions.values() if s.is_complete)

            self.status_pane.object = f"<p>Loaded {total_sessions} sessions ({complete_sessions} complete)</p>"

        except Exception as e:
            logger.error(f"Error loading sessions: {e}")
            self.status_pane.object = (
                f"<p style='color:red'>Error loading sessions: {e}</p>"
            )

    def _on_session_change(self, event):
        """Handle session selection change."""
        session_name = event.new
        if not session_name:
            # Hide UI elements when no session selected
            self.bucket_type_toggle.visible = False
            self.file_tree.visible = False
            self.file_tree.options = []
            self.file_viewer.object = "<p>No session selected</p>"
            self.file_name_header.object = "<h3>No session selected</h3>"
            return

        try:
            self._show_loading(f"Loading session: {session_name}")
            session = self.session_manager.load_session_files(session_name)

            if session:
                self.selected_session = session_name
                # Show UI elements when session is selected
                self.bucket_type_toggle.visible = True
                self.file_tree.visible = True
                self._update_bucket_type_buttons(session)
                self._update_file_tree(session)
                self._hide_loading()
                self.status_pane.object = f"<p>✅ Loaded session: {session_name}</p>"
            else:
                self._hide_loading()
                self.status_pane.object = (
                    f"<p style='color:red'>❌ Session not found: {session_name}</p>"
                )

        except Exception as e:
            logger.error(f"Error loading session {session_name}: {e}")
            self._hide_loading()
            self.status_pane.object = (
                f"<p style='color:red'>❌ Error loading session: {e}</p>"
            )

    def _update_bucket_type_buttons(self, session: SessionInfo):
        """Update bucket type button availability based on session data."""
        # Update button options based on what's available
        available_options = []
        if session.has_curated:
            available_options.append("curated")
        if session.has_processed:
            available_options.append("processed")

        if available_options:
            self.bucket_type_toggle.options = available_options
            # Set to first available option
            self.bucket_type_toggle.value = available_options[0]
            self.current_bucket_type = available_options[0]
            self.bucket_type_toggle.disabled = False
        else:
            # No data available - default to curated
            self.bucket_type_toggle.options = ["curated", "processed"]
            self.bucket_type_toggle.value = "curated"
            self.current_bucket_type = "curated"
            self.bucket_type_toggle.disabled = True

    def _on_bucket_type_change(self, event):
        """Handle bucket type toggle change."""
        # Ensure we always have a valid string value
        self.current_bucket_type = event.new or "curated"
        if self.selected_session:
            session = self.sessions.get(self.selected_session)
            if session:
                self._update_file_tree(session)

    def _update_file_tree(self, session: SessionInfo):
        """Update file tree based on current session and bucket type."""
        if self.current_bucket_type == "curated" and session.has_curated:
            files = session.curated_files or []
        elif self.current_bucket_type == "processed" and session.has_processed:
            files = session.processed_files or []
        else:
            files = []

        # Create file tree options - only show displayable files
        file_options: List[str] = [""]
        displayable_count = 0
        total_files = 0

        # Clear the mapping for the new file list
        self.display_to_path_map.clear()

        for file_info in files:
            if not file_info.get("IsDir", False):  # Only show files, not directories
                total_files += 1
                relative_path = file_info.get("RelativePath", file_info["Name"])

                # Check if file is displayable
                if self._is_file_displayable(relative_path):
                    displayable_count += 1
                    size = file_info.get("Size", 0)
                    size_str = self._format_file_size(size)

                    # Check if file is cached
                    cache_indicator = self._get_cache_indicator(
                        session.name, self.current_bucket_type, relative_path
                    )

                    # Truncate long file names for better display
                    display_path = self._truncate_filename(relative_path, max_length=60)

                    display_name = f"{cache_indicator} {display_path} ({size_str})"
                    file_options.append(display_name)

                    # Store mapping from display name to actual path
                    self.display_to_path_map[display_name] = relative_path

        self.file_tree.options = file_options
        self.current_session_files = files

        # Update viewer
        bucket_status = (
            "✓"
            if (
                (self.current_bucket_type == "curated" and session.has_curated)
                or (self.current_bucket_type == "processed" and session.has_processed)
            )
            else "✗"
        )

        self.file_viewer.object = f"""
        <div>
            <h3>{session.name} - {self.current_bucket_type.title()} Data {bucket_status}</h3>
            <p>Found {displayable_count} displayable files (of {total_files} total)</p>
            <p>Select a file from the list to view its contents</p>
            <p><small>📁 Cache location: {self.file_manager.get_cache_location()}</small></p>
        </div>
        """

    def _on_file_select(self, event):
        """Handle file selection."""
        selected_display = event.new
        if not selected_display or not self.selected_session or selected_display == "":
            self.file_viewer.object = "<p>No file selected</p>"
            self.file_name_header.object = "<h3>No file selected</h3>"
            self.edit_button.disabled = True
            self.selected_file = ""
            return

        # Get actual file path from mapping
        file_path = self.display_to_path_map.get(selected_display)
        if not file_path:
            logger.warning(
                f"Could not find file path for display name: {selected_display}"
            )
            self.file_viewer.object = "<p>Error: Could not find file</p>"
            self.file_name_header.object = (
                "<h3 style='color: red;'>Error: File not found</h3>"
            )
            return

        # Use shared file selection logic
        self._on_file_select_by_path(file_path)

    def _on_file_select_by_path(self, file_path: str):
        """Handle file selection by path (used by both old and new selection methods)."""
        if not file_path or not self.selected_session:
            self.file_viewer.object = "<p>No file selected</p>"
            self.file_name_header.object = "<h3>No file selected</h3>"
            self.edit_button.disabled = True
            self.selected_file = ""
            return

        # Skip if same file is selected again
        if file_path == self.selected_file:
            return

        try:
            # Show loading overlay
            self._show_loading(f"Loading {file_path}...")

            # Update file name header
            self.file_name_header.object = f"<h3>📁 {file_path}</h3>"

            # Get bucket and full path
            bucket, full_path = self.session_manager.get_file_path(
                self.selected_session, self.current_bucket_type, file_path
            )

            # Load file content with progress tracking
            content, render_info = self.file_manager.get_file_content_with_progress(
                bucket, full_path
            )

            self.current_file_content = content
            self.current_file_info = render_info
            self.selected_file = file_path

            # Update viewer
            self._update_file_viewer(render_info)

            # Update edit button state
            self.edit_button.disabled = not render_info.get("can_edit", False)

            # Update cache info and refresh file tree if needed
            if not render_info.get("from_cache", False):
                self._update_cache_info()
                # Refresh file tree to update cache indicators (without re-selection to avoid recursion)
                session = self.sessions.get(self.selected_session)
                if session:
                    self._update_file_tree(session)

            # Hide loading and show completion status
            self._hide_loading()
            cache_status = "💾" if render_info.get("from_cache", False) else "📡"
            self.status_pane.object = f"<p>{cache_status} Loaded: {file_path} ({self._format_file_size(len(content))})</p>"

        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            self.file_name_header.object = (
                f"<h3 style='color:red;'>❌ Error loading {file_path}</h3>"
            )
            self.file_viewer.object = f"<p style='color:red'>Error: {e}</p>"
            self._hide_loading()
            self.status_pane.object = f"<p style='color:red'>❌ Error: {e}</p>"

    def _update_file_viewer(self, render_info: Dict[str, Any]):
        """Update the file viewer based on render info."""
        if render_info["type"] == "text":
            # Text content with syntax highlighting
            content = render_info["content"]
            # language = render_info.get("language", "text")  # Available for future syntax highlighting

            # Display text content with basic syntax highlighting
            self.file_viewer.object = f"""
            <div>
                <h4>Text File ({render_info["lines"]} lines)</h4>
                <pre style="background-color: #f8f8f8; padding: 10px; border-radius: 4px; overflow-x: auto;"><code>{content}</code></pre>
            </div>
            """

        elif render_info["type"] == "table":
            # Tabular data
            stats = render_info["stats"]
            html_table = render_info["html"]

            header = f"""
            <div>
                <h4>Data Table</h4>
                <p><strong>Rows:</strong> {stats["rows"]}, <strong>Columns:</strong> {stats["columns"]}</p>
                <p><strong>Columns:</strong> {", ".join(stats["column_names"])}</p>
            </div>
            """

            if render_info.get("truncated", False):
                header += (
                    f"<p><em>Showing first {render_info['display_limit']} rows</em></p>"
                )

            self.file_viewer.object = header + html_table

        elif render_info["type"] == "video":
            # Video content
            size_mb = render_info["size_mb"]
            if size_mb > 50:  # Large video warning
                self.file_viewer.object = f"""
                <div>
                    <h4>Video File ({size_mb:.1f} MB)</h4>
                    <p style='color:orange'>Video is large and may take time to load.</p>
                    <video controls width="100%" height="400">
                        <source src="{render_info["data_url"]}" type="{render_info["mime_type"]}">
                        Your browser does not support video playback.
                    </video>
                </div>
                """
            else:
                self.file_viewer.object = f"""
                <div>
                    <h4>Video File ({size_mb:.1f} MB)</h4>
                    <video controls width="100%" height="400">
                        <source src="{render_info["data_url"]}" type="{render_info["mime_type"]}">
                        Your browser does not support video playback.
                    </video>
                </div>
                """

        elif render_info["type"] == "error":
            self.file_viewer.object = f"""
            <div>
                <h4 style='color:red'>Error</h4>
                <p>{render_info["message"]}</p>
                <p>File size: {self._format_file_size(render_info["size"])}</p>
            </div>
            """

        elif render_info["type"] == "unknown":
            self.file_viewer.object = f"""
            <div>
                <h4>Unknown File Type</h4>
                <p>{render_info["message"]}</p>
                <p>File size: {self._format_file_size(render_info["size"])}</p>
            </div>
            """

    def _start_edit(self, _):
        """Start editing the current file."""
        if not self.selected_file or not self.selected_session:
            return

        try:
            bucket, full_path = self.session_manager.get_file_path(
                self.selected_session, self.current_bucket_type, self.selected_file
            )

            edit_content = self.file_manager.prepare_file_for_edit(bucket, full_path)

            self.file_editor.value = edit_content

            # Switch to edit mode
            self.edit_mode = True
            self.edit_button.disabled = True
            self.save_button.disabled = False
            self.cancel_edit_button.disabled = False

            self.status_pane.object = f"<p>Editing: {self.selected_file}</p>"

        except Exception as e:
            logger.error(f"Error starting edit: {e}")
            self.status_pane.object = (
                f"<p style='color:red'>Error starting edit: {e}</p>"
            )

    def _save_edit(self, _):
        """Save the edited file."""
        if not self.edit_mode or not self.selected_file:
            return

        try:
            bucket, full_path = self.session_manager.get_file_path(
                self.selected_session, self.current_bucket_type, self.selected_file
            )

            edited_content = self.file_editor.value
            self.file_manager.save_edited_file(bucket, full_path, edited_content)

            # Refresh file view
            self._on_file_select(type("Event", (), {"new": self.selected_file})())

            # Exit edit mode
            self._exit_edit_mode()

            self.status_pane.object = (
                f"<p style='color:green'>Saved: {self.selected_file}</p>"
            )

        except Exception as e:
            logger.error(f"Error saving file: {e}")
            self.status_pane.object = f"<p style='color:red'>Error saving file: {e}</p>"

    def _cancel_edit(self, _):
        """Cancel editing."""
        self._exit_edit_mode()
        self.status_pane.object = f"<p>Cancelled editing: {self.selected_file}</p>"

    def _exit_edit_mode(self):
        """Exit edit mode and reset UI."""
        self.edit_mode = False
        self.edit_button.disabled = not self.current_file_info.get("can_edit", False)
        self.save_button.disabled = True
        self.cancel_edit_button.disabled = True
        self.file_editor.value = ""

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes / (1024**2):.1f} MB"
        else:
            return f"{size_bytes / (1024**3):.1f} GB"

    def _is_file_displayable(self, file_path: str) -> bool:
        """Check if a file can be displayed by the dashboard."""
        return self.file_manager.is_file_supported(file_path)

    def _get_cache_indicator(
        self, session_name: str, bucket_type: str, file_path: str
    ) -> str:
        """Get cache indicator icon for a file."""
        if self.file_manager.is_file_cached(session_name, bucket_type, file_path):
            return "💾"  # Cached locally
        else:
            return "📡"  # Remote only

    def _truncate_filename(self, file_path: str, max_length: int = 50) -> str:
        """Truncate long file names for better display."""
        if len(file_path) <= max_length:
            return file_path

        # Try to keep the file extension visible
        path_obj = Path(file_path)
        name = path_obj.stem
        ext = path_obj.suffix

        # Calculate how much of the name we can show
        available_length = max_length - len(ext) - 3  # 3 for "..."

        if available_length > 10:  # Only truncate if we have reasonable space
            truncated_name = name[:available_length] + "..."
            return truncated_name + ext
        else:
            # If the extension is too long, just truncate the whole thing
            return file_path[: max_length - 3] + "..."

    def _reselect_current_file(self):
        """Re-select the current file after file tree update."""
        if not self.selected_file:
            return

        # Find the display name for the current file
        for display_name, file_path in self.display_to_path_map.items():
            if file_path == self.selected_file:
                # Temporarily disconnect the callback to avoid recursion
                watchers = self.file_tree.param.watchers.get("value", [])
                for watcher in watchers[:]:
                    if watcher.fn == self._on_file_select:
                        self.file_tree.param.unwatch(watcher)

                self.file_tree.value = display_name

                # Reconnect the callback
                self.file_tree.param.watch(self._on_file_select, "value")
                break

    def _clear_cache(self, _):
        """Clear the cache directory."""
        try:
            cache_dir = Path(self.file_manager.get_cache_location())
            if cache_dir.exists():
                import shutil

                shutil.rmtree(cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)
                self.status_pane.object = (
                    "<p style='color:green'>Cache cleared successfully</p>"
                )
                self._update_cache_info()
                # Refresh current file tree to update cache indicators
                if self.selected_session:
                    session = self.sessions.get(self.selected_session)
                    if session:
                        self._update_file_tree(session)
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            self.status_pane.object = (
                f"<p style='color:red'>Error clearing cache: {e}</p>"
            )

    def _update_cache_info(self):
        """Update cache information display."""
        try:
            cache_dir = Path(self.file_manager.get_cache_location())
            if cache_dir.exists():
                cache_files = list(cache_dir.glob("*"))
                cache_count = len(cache_files)

                # Calculate cache size
                cache_size = sum(f.stat().st_size for f in cache_files if f.is_file())
                cache_size_str = self._format_file_size(cache_size)

                self.cache_info_pane.object = f"""
                <small>
                Cache: {cache_count} files ({cache_size_str})<br>
                Location: {cache_dir}
                </small>
                """
            else:
                self.cache_info_pane.object = "<small>Cache: Empty</small>"
        except Exception as e:
            self.cache_info_pane.object = f"<small>Cache: Error - {e}</small>"

    def _show_loading(self, message: str = "Loading..."):
        """Show the loading indicator."""
        self.loading_indicator.value = True
        self.loading_indicator.visible = True
        self.loading_indicator.name = message
        # Also show in status
        self.status_pane.object = f"<p>🔄 {message}</p>"

    def _hide_loading(self):
        """Hide the loading indicator."""
        self.loading_indicator.value = False
        self.loading_indicator.visible = False

    def create_layout(self):
        """Create the dashboard layout."""
        # Navigation panel - enlarged width with cache management
        cache_controls = pn.Column(self.cache_info_pane, self.clear_cache_button)

        nav_panel = pn.Column(
            "## Data Browser",
            self.session_select,
            self.bucket_type_toggle,
            self.file_tree,
            # "---",
            "### Cache Management",
            cache_controls,
            # width=850,  # Wider navigation panel to accommodate file tree
            # height=500  # Reduced height
        )

        # Main content area with file name header
        viewer_tabs = pn.Tabs(
            ("View", self.file_viewer),
            (
                "Edit",
                pn.Column(
                    pn.Row(self.edit_button, self.save_button, self.cancel_edit_button),
                    self.file_editor,
                ),
            ),
            dynamic=True,
        )

        content_panel = pn.Column(
            self.file_name_header,
            viewer_tabs,
            # sizing_mode="stretch_both"
        )

        # Status panel at the bottom with loading indicator
        status_panel = pn.Row(
            self.status_pane,
            # pn.Spacer(),
            # sizing_mode="stretch_width"
        )

        # Main layout (unused - kept for potential future use)
        # main_layout = pn.Row(
        #     nav_panel,
        #     content_panel,
        #     sizing_mode="stretch_width",
        #     height=700
        # )

        return pn.template.MaterialTemplate(
            title="CIS Data Dashboard",
            sidebar=[nav_panel],
            main=[status_panel, self.loading_indicator, content_panel],
            header_background="#2596be",
            sidebar_width=600,  # Much wider sidebar for file tree
        )


def create_app():
    """Create and return the dashboard application."""
    try:
        dashboard = DataDashboard()
        return dashboard.create_layout()
    except Exception as e:
        logger.error(f"Failed to create dashboard: {e}")
        error_pane = pn.pane.HTML(f"""
        <div style='color:red; padding:20px'>
            <h2>Dashboard Initialization Error</h2>
            <p><strong>Error:</strong> {e}</p>
            <p>Please check:</p>
            <ul>
                <li>rclone is installed and configured</li>
                <li>Remote 'collab-data' is set up</li>
                <li>Required dependencies are installed</li>
            </ul>
            <pre>{traceback.format_exc()}</pre>
        </div>
        """)
        return error_pane


def serve_dashboard(port: int = 5007, show: bool = True, autoreload: bool = True):
    """Serve the dashboard application."""
    app = create_app()

    return pn.serve(
        app,
        port=port,
        show=show,
        title="GCS Data Dashboard",
        autoreload=autoreload,
    )
