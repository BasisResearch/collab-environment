"""
Main dashboard application using Panel/HoloViz.
"""

import panel as pn
import param
from typing import Dict, Optional, List, Any
from pathlib import Path
import traceback
import subprocess
import requests
from loguru import logger

from .rclone_client import RcloneClient
from .session_manager import SessionManager, SessionInfo
from .file_viewers import FileContentManager

# Enable Panel extensions
pn.extension("tabulator", "vtk")

# No custom CSS - using only Panel widgets


class DataDashboard(param.Parameterized):
    """Main dashboard application for browsing GCS data."""

    # Parameters for reactive UI
    selected_session = param.String(default="", doc="Currently selected session")
    selected_file = param.String(default="", doc="Currently selected file")
    current_bucket_type = param.String(
        default="curated", doc="Current bucket type view"
    )

    def __init__(
        self,
        remote_name: Optional[str] = None,
        curated_bucket: Optional[str] = None,
        processed_bucket: Optional[str] = None,
        **params,
    ):
        super().__init__(**params)

        # Initialize clients
        try:
            # RcloneClient expects a string, so provide default if None
            if remote_name:
                self.rclone_client = RcloneClient(remote_name=remote_name)
            else:
                self.rclone_client = RcloneClient()  # Uses default "collab-data"
            self.session_manager = SessionManager(
                self.rclone_client,
                curated_bucket=curated_bucket or "fieldwork_curated",
                processed_bucket=processed_bucket or "fieldwork_processed",
            )
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
            width=550,  # Width for container
            height=300,  # Height
            visible=False,  # Hidden initially until session selected
            stylesheets=[
                """
                select {
                    overflow-x: auto !important;
                    white-space: nowrap !important;
                }
                option {
                    white-space: nowrap !important;
                }
            """
            ],
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

        # Video conversion button
        self.convert_video_button = pn.widgets.Button(
            name="Convert to H.264", button_type="primary", width=120, visible=False
        )

        # Video bbox viewer button
        self.bbox_viewer_button = pn.widgets.Button(
            name="View with Overlays", button_type="success", width=140, visible=False
        )
        
        # 3D mesh track viewer button
        self.mesh_3d_viewer_button = pn.widgets.Button(
            name="View 3D Tracks", button_type="success", width=140, visible=False
        )

        # Stop persistent video server button
        self.stop_all_viewers_button = pn.widgets.Button(
            name="Stop Server", button_type="warning", width=100, visible=False
        )

        # General file management buttons
        self.replace_file_button = pn.widgets.Button(
            name="Replace in Cloud", button_type="primary", width=140
        )
        self.download_original_button = pn.widgets.Button(
            name="Download Original", button_type="warning", width=140, visible=False
        )
        self.delete_file_button = pn.widgets.Button(
            name="Delete local and remote", button_type="danger", width=160
        )

        # File management controls container with proper spacing
        self.file_management_controls = pn.Row(
            self.replace_file_button,
            pn.Spacer(width=10),  # Add space between buttons
            self.download_original_button,
            pn.Spacer(width=10),  # Add space between buttons
            self.delete_file_button,
            margin=(5, 0),
            visible=False,  # Initially hidden until file is loaded
        )

        self.status_pane = pn.pane.HTML("<p>Ready</p>", width=800, height=30)

        # Loading indicator as a proper overlay modal
        self.loading_modal = pn.pane.HTML("", visible=False, sizing_mode="stretch_both")

        # Single reusable VTK pane - defer creation until needed
        # Panel VTK pane requires a valid render window, can't be created with None
        self.vtk_pane: Optional[pn.pane.VTK] = None
        logger.info("🔨 VTK pane creation deferred until first PLY file is loaded")

        # Track available bbox CSV files for current video
        self._current_bbox_csvs: list[dict] = []
        
        # Track available 3D CSV files and camera params for current PLY
        self._current_csv_3d_files: list[str] = []
        self._current_camera_params_file: Optional[str] = None

        # Single persistent Flask server
        self._persistent_flask_process: Optional[subprocess.Popen] = None
        self._persistent_flask_port = 5050

        # Container that holds the VTK pane (starts empty, VTK pane added on demand)
        self.persistent_vtk_container = pn.Column(
            width=800,
            height=0,  # Start collapsed
            sizing_mode="fixed",
            styles={"display": "none"},
        )

        # Modal dialog buttons (will be created in create_layout)
        self.confirm_delete_button = pn.widgets.Button(
            name="Yes, Delete", button_type="danger", width=100
        )
        self.cancel_delete_button = pn.widgets.Button(
            name="Cancel", button_type="light", width=100
        )

        # Cache management buttons
        self.clear_cache_button = pn.widgets.Button(name="Clear Cache", width=100)
        self.cache_info_pane = pn.pane.HTML("", width=300)

        # PLY viewer state
        self.current_ply_viewer = None

        # Wire up callbacks
        self.session_select.param.watch(self._on_session_change, "value")
        self.bucket_type_toggle.param.watch(self._on_bucket_type_change, "value")
        self.file_tree.param.watch(self._on_file_select, "value")
        self.edit_button.on_click(self._start_edit)
        self.save_button.on_click(self._save_edit)
        self.cancel_edit_button.on_click(self._cancel_edit)
        self.clear_cache_button.on_click(self._clear_cache)
        self.convert_video_button.on_click(self._convert_video)
        self.bbox_viewer_button.on_click(self._open_bbox_viewer)
        self.mesh_3d_viewer_button.on_click(self._open_mesh_3d_viewer)
        self.stop_all_viewers_button.on_click(self._stop_all_bbox_viewers)
        self.replace_file_button.on_click(self._replace_file_in_cloud)
        self.download_original_button.on_click(self._download_original_file)
        self.delete_file_button.on_click(self._delete_file)
        self.confirm_delete_button.on_click(self._confirm_delete)
        self.cancel_delete_button.on_click(self._cancel_delete)

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
            self.selected_file = ""
            # Clean up any existing PLY viewer when no session selected
            self._cleanup_ply_viewer()
            self._update_file_management_buttons()
            return

        try:
            # Clean up any existing PLY viewer when switching to a different session
            self._cleanup_ply_viewer()

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
        # Clean up any existing PLY viewer when switching bucket types
        self._cleanup_ply_viewer()

        # Ensure we always have a valid string value
        self.current_bucket_type = event.new or "curated"
        if self.selected_session:
            session = self.sessions.get(self.selected_session)
            if session:
                self._update_file_tree(session)

    def _update_file_tree(self, session: SessionInfo):
        """Update file tree based on current session and bucket type."""
        # Preserve current selection before clearing mapping
        current_file_path = None
        if self.file_tree.value and self.file_tree.value in self.display_to_path_map:
            current_file_path = self.display_to_path_map[self.file_tree.value]

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

                    # Use full path for clarity
                    display_path = relative_path

                    display_name = f"{cache_indicator} {display_path} ({size_str})"
                    file_options.append(display_name)

                    # Store mapping from display name to actual path
                    self.display_to_path_map[display_name] = relative_path

        self.file_tree.options = file_options
        self.current_session_files = files

        # Restore selection based on file path (find the new display name for the same file)
        if current_file_path:
            for display_name, file_path in self.display_to_path_map.items():
                if file_path == current_file_path:
                    self.file_tree.value = display_name
                    break

        # Only update viewer to show session info if no file is currently selected
        if not self.selected_file:
            bucket_status = (
                "✓"
                if (
                    (self.current_bucket_type == "curated" and session.has_curated)
                    or (
                        self.current_bucket_type == "processed"
                        and session.has_processed
                    )
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
            self._update_file_management_buttons()
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
            # Hide all file management buttons when no file selected
            self.convert_video_button.visible = False
            self.bbox_viewer_button.visible = False
            self.mesh_3d_viewer_button.visible = False
            self._current_bbox_csvs = []
            self._current_csv_3d_files = []
            self._current_camera_params_file = None
            self._update_file_management_buttons()
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

            # Add path information to render_info for display
            render_info["remote_path"] = f"{bucket}/{full_path}"
            render_info["cache_path"] = self.file_manager.get_cache_path(
                bucket, full_path
            )

            self.current_file_content = content
            self.current_file_info = render_info
            self.selected_file = file_path

            # Update viewer first
            self._update_file_viewer(render_info)

            # Update edit button state
            self.edit_button.disabled = not render_info.get("can_edit", False)

            # Show file management buttons for all files
            self._update_file_management_buttons()

            # Update cache info and refresh file tree AFTER updating viewer
            if not render_info.get("from_cache", False):
                self._update_cache_info()
                # Refresh file tree to update cache indicators
                session = self.sessions.get(self.selected_session)
                if session:
                    self._update_file_tree(session)
                    # Ensure buttons remain visible after tree update
                    self._update_file_management_buttons()

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

    def _get_file_path_info(self, render_info: Dict[str, Any]) -> str:
        """Generate file path information HTML with modification status."""
        cache_status = (
            "💾 Cached" if render_info.get("from_cache", False) else "📡 Downloaded"
        )
        remote_path = render_info.get("remote_path", "Unknown")
        cache_path = render_info.get("cache_path", "Unknown")

        # Check modification status
        modification_status = ""
        if self.selected_file and self.selected_session:
            try:
                bucket, full_path = self.session_manager.get_file_path(
                    self.selected_session, self.current_bucket_type, self.selected_file
                )
                is_modified = self.file_manager.is_file_modified(bucket, full_path)
                if is_modified:
                    modification_status = '<div><strong>Modified:</strong> <span style="color: #d63384;">📝 Local differs from remote</span></div>'
                else:
                    modification_status = '<div><strong>Modified:</strong> <span style="color: #198754;">💾 Matches remote</span></div>'
            except Exception:
                modification_status = '<div><strong>Modified:</strong> <span style="color: #6c757d;">❓ Unknown</span></div>'

        return f"""
        <div style="background-color: #f0f0f0; padding: 10px; margin-bottom: 10px; border-radius: 4px; font-size: 12px;">
            <div><strong>Status:</strong> {cache_status}</div>
            {modification_status}
            <div><strong>Remote:</strong> <code>{remote_path}</code></div>
            <div><strong>Cache:</strong> <code>{cache_path}</code></div>
        </div>
        """

    def _cleanup_ply_viewer(self):
        """Hide the reusable VTK container but keep VTK pane for reuse."""
        if hasattr(self, "persistent_vtk_container"):
            logger.info("🙈 HIDING reusable VTK container (keeping VTK pane for reuse)")
            # Just collapse and hide the container - DON'T clear it or reset vtk_pane
            self.persistent_vtk_container.height = 0
            self.persistent_vtk_container.styles = {"display": "none"}

            # Keep the VTK pane alive for reuse - don't set to None
            if hasattr(self, "vtk_pane") and self.vtk_pane is not None:
                logger.info("✅ VTK pane preserved for reuse")
        else:
            logger.debug("🙈 VTK container not found")

    def _update_view_container(self):
        """Update visibility of persistent VTK container."""
        # No need to change container objects since persistent_vtk_container is always there
        # Just control its visibility
        logger.info(
            "📋 LAYOUT: Using persistent VTK container approach - no container changes needed"
        )

    def _update_file_viewer(self, render_info: Dict[str, Any]):
        """Update the file viewer based on render info."""
        logger.info(
            f"🔍 UPDATING file viewer for type: {render_info.get('type', 'unknown')}"
        )
        # ALWAYS clean up any existing PLY viewer first, regardless of file type
        self._cleanup_ply_viewer()

        if render_info["type"] == "text":
            # Text content with syntax highlighting
            content = render_info["content"]
            # language = render_info.get("language", "text")  # Available for future syntax highlighting
            path_info = self._get_file_path_info(render_info)

            # Hide video management buttons for text files
            self.convert_video_button.visible = False
            self.bbox_viewer_button.visible = False
            self.mesh_3d_viewer_button.visible = False
            self._current_bbox_csvs = []
            self._current_csv_3d_files = []
            self._current_camera_params_file = None

            self._update_view_container()

            # Display text content with basic syntax highlighting
            self.file_viewer.object = f"""
            <div>
                {path_info}
                <h4>Text File ({render_info["lines"]} lines)</h4>
                <pre style="background-color: #f8f8f8; padding: 10px; border-radius: 4px; overflow-x: auto;"><code>{content}</code></pre>
            </div>
            """

        elif render_info["type"] == "table":
            # Tabular data
            stats = render_info["stats"]
            html_table = render_info["html"]
            path_info = self._get_file_path_info(render_info)

            # Hide video management buttons for table files
            self.convert_video_button.visible = False
            self.bbox_viewer_button.visible = False
            self.mesh_3d_viewer_button.visible = False
            self._current_bbox_csvs = []
            self._current_csv_3d_files = []
            self._current_camera_params_file = None

            self._update_view_container()

            header = f"""
            <div>
                {path_info}
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
            path_info = self._get_file_path_info(render_info)
            codec_info = render_info.get("codec_info", {})
            bbox_csvs = render_info.get("bbox_csvs", [])

            self._update_view_container()

            # Build codec information display
            codec_name = codec_info.get("codec_name", "unknown")
            browser_compatible = codec_info.get("browser_compatible")
            width = codec_info.get("width")
            height = codec_info.get("height")

            # Format codec details
            codec_details = f"<p><strong>Codec:</strong> {codec_name}"
            if width and height:
                codec_details += f" | <strong>Resolution:</strong> {width}×{height}"
            codec_details += "</p>"

            # Show/hide conversion buttons based on codec compatibility
            # Only show convert button for incompatible or limited codecs that aren't already H.264/AVC
            safe_codecs = ["h264", "avc"]
            needs_conversion = (
                browser_compatible in [False, "limited"]
                and codec_name.lower() not in safe_codecs
            )
            self.convert_video_button.visible = needs_conversion
            self.convert_video_button.disabled = False
            
            # Hide 3D viewer button for video files
            self.mesh_3d_viewer_button.visible = False
            self._current_csv_3d_files = []
            self._current_camera_params_file = None

            # Update bbox viewer button visibility based on available CSV files
            if bbox_csvs:
                self.bbox_viewer_button.visible = True
                self.bbox_viewer_button.disabled = False
                # Store bbox CSV info for the button click handler
                self._current_bbox_csvs = bbox_csvs
            else:
                self.bbox_viewer_button.visible = False
                self._current_bbox_csvs = []

            # Show compatibility warning for incompatible codecs
            compatibility_warning = ""
            if browser_compatible is False:
                compatibility_warning = f"""
                <div style="background-color: #f8d7da; border: 1px solid #f5c6cb; padding: 10px; margin: 10px 0; border-radius: 4px;">
                    <strong>❌ Unsupported Codec</strong><br>
                    This video uses the <code>{codec_name}</code> codec, which is not supported by modern browsers.<br>
                    <small>Tip: Use the convert button below to create an H.264 version.</small>
                </div>
                """
            elif browser_compatible == "limited":
                compatibility_warning = f"""
                <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; margin: 10px 0; border-radius: 4px;">
                    <strong>⚠️ Limited Browser Support</strong><br>
                    This video uses the <code>{codec_name}</code> codec, which has limited browser support.<br>
                    <small>May not work in all browsers. Use the convert button below for better compatibility.</small>
                </div>
                """
            elif browser_compatible is True:
                compatibility_warning = """
                <div style="background-color: #d4edda; border: 1px solid #c3e6cb; padding: 5px 10px; margin: 10px 0; border-radius: 4px; font-size: 12px;">
                    ✅ Widely supported codec
                </div>
                """

            # Build bbox viewer info section
            bbox_info = ""
            if bbox_csvs:
                bbox_count = len(bbox_csvs)
                bbox_names = ", ".join(
                    [csv.get("display_name", csv["name"]) for csv in bbox_csvs[:3]]
                )  # Show first 3 names with cross-folder indication
                if bbox_count > 3:
                    bbox_names += f" (+{bbox_count - 3} more)"

                bbox_info = f"""
                <div style="background-color: #e3f2fd; border: 1px solid #90caf9; padding: 10px; margin: 10px 0; border-radius: 4px;">
                    <strong>📊 Tracking Data Available</strong><br>
                    Found {bbox_count} CSV file{"s" if bbox_count != 1 else ""} with bounding box data: <code>{bbox_names}</code><br>
                    <small>Use the "View with Overlays" button below to see video with tracking visualizations.</small>
                </div>
                """

            if size_mb > 50:  # Large video warning
                self.file_viewer.object = f"""
                <div>
                    {path_info}
                    <h4>Video File ({size_mb:.1f} MB)</h4>
                    {codec_details}
                    {compatibility_warning}
                    {bbox_info}
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
                    {path_info}
                    <h4>Video File ({size_mb:.1f} MB)</h4>
                    {codec_details}
                    {compatibility_warning}
                    {bbox_info}
                    <video controls width="100%" height="400">
                        <source src="{render_info["data_url"]}" type="{render_info["mime_type"]}">
                        Your browser does not support video playback.
                    </video>
                </div>
                """

        elif render_info["type"] == "ply_3d":
            # PLY 3D mesh visualization
            path_info = self._get_file_path_info(render_info)
            stats = render_info.get("stats", {})

            # Hide video management buttons for PLY files
            self.convert_video_button.visible = False
            self.bbox_viewer_button.visible = False
            self._current_bbox_csvs = []
            
            # Check for 3D CSV files and show 3D viewer button if available
            csv_3d_files = render_info.get("csv_3d_files", [])
            camera_params_file = render_info.get("camera_params_file")
            
            if csv_3d_files:
                self.mesh_3d_viewer_button.visible = True
                self.mesh_3d_viewer_button.disabled = False
                self._current_csv_3d_files = csv_3d_files
                self._current_camera_params_file = camera_params_file
                logger.info(f"Found {len(csv_3d_files)} 3D CSV files for PLY mesh")
            else:
                self.mesh_3d_viewer_button.visible = False
                self._current_csv_3d_files = []
                self._current_camera_params_file = None

            # Build statistics display based on data type
            is_point_cloud = stats.get("is_point_cloud", False)
            data_type = "Point Cloud" if is_point_cloud else "3D Mesh"
            stats_html = f"<h4>{data_type} Statistics</h4><ul>"

            if stats.get("points"):
                stats_html += f"<li><strong>Points:</strong> {stats['points']:,}</li>"
            if stats.get("cells") and not is_point_cloud:
                stats_html += f"<li><strong>Cells:</strong> {stats['cells']:,}</li>"
            if stats.get("area"):
                stats_html += (
                    f"<li><strong>Surface Area:</strong> {stats['area']:.2f}</li>"
                )
            if stats.get("volume"):
                stats_html += f"<li><strong>Volume:</strong> {stats['volume']:.2f}</li>"
            if stats.get("bounds"):
                bounds = stats["bounds"]
                stats_html += f"<li><strong>Bounds:</strong> X: [{bounds[0]:.2f}, {bounds[1]:.2f}], Y: [{bounds[2]:.2f}, {bounds[3]:.2f}], Z: [{bounds[4]:.2f}, {bounds[5]:.2f}]</li>"
            stats_html += "</ul>"

            # Get the render window to update the persistent VTK pane
            render_window = render_info.get("render_window")

            if render_window:
                info_section = f"""
                <div>
                    {path_info}
                    <h4>PLY {data_type} File</h4>
                    {stats_html}
                    <p><em><strong>Interactive 3D viewer</strong></em></p>
                    <p>🎮 Use mouse to rotate, zoom, and pan. Keyboard shortcuts:
                    <ul>
                        <li>s: set representation of all actors to surface</li>
                        <li>w: set representation of all actors to wireframe</li>
                        <li>v: set representation of all actors to vertex</li>
                        <li>r: center the actors and move the camera so that all actors are visible</li>
                    </ul>
                    </p>
                </div>
                """
                self.file_viewer.object = info_section

                # Create or update the reusable VTK pane with new render window
                logger.info("🔄 Creating/updating VTK pane with new render window")

                try:
                    # Create VTK pane on demand if it doesn't exist
                    if self.vtk_pane is None:
                        logger.info("🔨 Creating VTK pane ONCE with render window")
                        self.vtk_pane = pn.pane.VTK(
                            render_window,
                            width=800,
                            height=600,
                            sizing_mode="fixed",
                            enable_keybindings=True,
                            orientation_widget=True,
                        )

                        # Add to the container
                        self.persistent_vtk_container.clear()
                        self.persistent_vtk_container.append(self.vtk_pane)
                        logger.info("✅ VTK pane created ONCE and added to container")

                        # The camera should already be properly set by PyVista, but ensure it's visible
                        logger.info("📷 Initial camera view set by PyVista")
                    else:
                        # Update existing VTK pane with new render window
                        logger.info(
                            "🔄 REUSING existing VTK pane, updating render window"
                        )
                        self.vtk_pane.object = render_window  # type: ignore[attr-defined]

                        # Reset camera view to properly frame the new data
                        logger.info("📷 Resetting camera view for new PLY data")
                        try:
                            # Get mesh bounds and center for better camera positioning
                            mesh_bounds = render_info.get("mesh_bounds")
                            mesh_center = render_info.get("mesh_center")

                            # Access the render window from the VTK pane and reset camera
                            ren_win = self.vtk_pane.object  # type: ignore[attr-defined]
                            if ren_win and hasattr(ren_win, "GetRenderers"):
                                renderers = ren_win.GetRenderers()
                                if renderers.GetNumberOfItems() > 0:
                                    renderer = renderers.GetFirstRenderer()
                                    if renderer:
                                        # Reset camera to fit the bounds
                                        renderer.ResetCamera()

                                        # If we have mesh center, position camera for better iso view
                                        if (
                                            mesh_center is not None
                                            and mesh_bounds is not None
                                        ):
                                            camera = renderer.GetActiveCamera()
                                            # Calculate a good distance from the bounds
                                            bounds_range = max(
                                                mesh_bounds[1]
                                                - mesh_bounds[0],  # x range
                                                mesh_bounds[3]
                                                - mesh_bounds[2],  # y range
                                                mesh_bounds[5]
                                                - mesh_bounds[4],  # z range
                                            )
                                            distance = (
                                                bounds_range * 2.0
                                            )  # Good viewing distance

                                            # Set isometric view position
                                            camera.SetPosition(
                                                mesh_center[0] + distance,
                                                mesh_center[1] + distance,
                                                mesh_center[2] + distance,
                                            )
                                            camera.SetFocalPoint(
                                                mesh_center[0],
                                                mesh_center[1],
                                                mesh_center[2],
                                            )
                                            camera.SetViewUp(0, 0, 1)  # Z-up
                                            renderer.ResetCameraClippingRange()

                                        ren_win.Render()
                                        logger.info(
                                            "✅ Camera view reset with mesh bounds successfully"
                                        )
                        except Exception as camera_error:
                            logger.warning(
                                f"⚠️ Could not reset camera view: {camera_error}"
                            )

                    # Show and expand the VTK container
                    logger.info("📦 EXPANDING VTK container")
                    self.persistent_vtk_container.height = 600
                    self.persistent_vtk_container.margin = 5
                    self.persistent_vtk_container.styles = {"display": "block"}

                except Exception as vtk_error:
                    logger.error(f"🚨 VTK pane creation/update failed: {vtk_error}")
                    # Fall back to error section if VTK pane creation fails
                    error_section = f"""
                    <div>
                        {path_info}
                        <h4>PLY {data_type} File</h4>
                        {stats_html}
                        <p style='color: red;'><strong>❌ VTK viewer creation failed</strong></p>
                        <p>Error: {vtk_error}</p>
                    </div>
                    """
                    self.file_viewer.object = error_section

            else:
                # Fallback if render window failed
                error_section = f"""
                <div>
                    {path_info}
                    <h4>PLY 3D Mesh File</h4>
                    {stats_html}
                    <p style='color: red;'><strong>❌ Failed to create 3D viewer</strong></p>
                    <p>The PLY file was loaded but the 3D visualization could not be created.</p>
                </div>
                """
                self.file_viewer.object = error_section
                # Make sure VTK container is collapsed but keep VTK pane for reuse
                if hasattr(self, "persistent_vtk_container"):
                    self.persistent_vtk_container.height = 0
                    self.persistent_vtk_container.styles = {"display": "none"}
                # Keep VTK pane alive for reuse

        elif render_info["type"] == "error":
            path_info = self._get_file_path_info(render_info)

            # Hide video management buttons for error files
            self.convert_video_button.visible = False
            self.bbox_viewer_button.visible = False
            self.mesh_3d_viewer_button.visible = False
            self._current_bbox_csvs = []
            self._current_csv_3d_files = []
            self._current_camera_params_file = None

            self._update_view_container()

            self.file_viewer.object = f"""
            <div>
                {path_info}
                <h4 style='color:red'>Error</h4>
                <p>{render_info["message"]}</p>
                <p>File size: {self._format_file_size(render_info["size"])}</p>
            </div>
            """

        elif render_info["type"] == "unknown":
            path_info = self._get_file_path_info(render_info)

            # Hide video management buttons for unknown files
            self.convert_video_button.visible = False
            self.bbox_viewer_button.visible = False
            self.mesh_3d_viewer_button.visible = False
            self._current_bbox_csvs = []
            self._current_csv_3d_files = []
            self._current_camera_params_file = None

            self._update_view_container()

            self.file_viewer.object = f"""
            <div>
                {path_info}
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
        """Save the edited file to local cache only."""
        if not self.edit_mode or not self.selected_file:
            return

        try:
            bucket, full_path = self.session_manager.get_file_path(
                self.selected_session, self.current_bucket_type, self.selected_file
            )

            edited_content = self.file_editor.value

            # Save to local cache only (not to cloud)
            self._save_to_local_cache(bucket, full_path, edited_content)

            # Force reload from cache to show updated content
            # Clear any cached content in the file manager first
            cache_path = self.file_manager._get_cache_path(bucket, full_path)
            if cache_path.exists():
                # Reload the file content directly
                content, render_info = self.file_manager.get_file_content_with_progress(
                    bucket, full_path
                )
                self.current_file_content = content
                self.current_file_info = render_info
                self._update_file_viewer(render_info)

            # Exit edit mode
            self._exit_edit_mode()

            self.status_pane.object = f"<p style='color:green'>💾 Saved to local cache: {self.selected_file} - Use 'Replace in Cloud' to upload</p>"

        except Exception as e:
            logger.error(f"Error saving file to cache: {e}")
            self.status_pane.object = (
                f"<p style='color:red'>Error saving file to cache: {e}</p>"
            )

    def _save_to_local_cache(self, bucket: str, full_path: str, edited_content: str):
        """Save edited content to local cache file only."""
        # Get the file viewer for proper content processing
        viewer = self.file_manager.viewer_registry.get_viewer(full_path)
        if not viewer or not viewer.can_edit():
            raise ValueError(f"File {full_path} is not editable")

        # Process the content (e.g., encode to bytes)
        processed_content = viewer.process_edit(edited_content, full_path)

        # Get cache path and write directly to cache
        cache_path = self.file_manager._get_cache_path(bucket, full_path)
        cache_path.write_bytes(processed_content)

        logger.info(f"Saved edited content to cache: {cache_path}")

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
        """Get cache and modification indicator icon for a file."""
        # Get the actual bucket name from session_manager
        if bucket_type == "curated":
            bucket = self.session_manager.curated_bucket
        else:
            bucket = self.session_manager.processed_bucket

        # Construct the full path
        full_path = f"{session_name}/{file_path.strip('/')}"

        # Check if file is cached
        cache_path = self.file_manager._get_cache_path(bucket, full_path)
        is_cached = cache_path.exists()

        if not is_cached:
            return "📡"  # Remote only

        # File is cached, check if modified
        try:
            is_modified = self.file_manager.is_file_modified(bucket, full_path)
            if is_modified:
                return "📝"  # Cached and modified
            else:
                return "💾"  # Cached, not modified

        except Exception as e:
            logger.warning(f"Error checking modification status for {file_path}: {e}")
            return "💾"  # Default to cached if we can't check

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
        """Show the blocking loading modal."""
        # Create a full-screen overlay modal
        modal_html = f"""
        <div style='
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            z-index: 10000;
            display: flex;
            justify-content: center;
            align-items: center;
        '>
            <div style='
                background: rgba(255, 255, 255, 0.95);
                padding: 40px 30px;
                border-radius: 16px;
                box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
                backdrop-filter: blur(10px);
                text-align: center;
                min-width: 300px;
            '>
                <div style='
                    width: 60px;
                    height: 60px;
                    border: 4px solid #e3e3e3;
                    border-top: 4px solid #3498db;
                    border-radius: 50%;
                    animation: spin 2s linear infinite;
                    margin: 0 auto 20px auto;
                '></div>
                <p style='
                    margin: 0;
                    color: #555;
                    font-size: 16px;
                    font-weight: 500;
                    letter-spacing: 0.5px;
                '>{message}</p>
            </div>
        </div>
        <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        </style>
        """
        self.loading_modal.object = modal_html
        self.loading_modal.visible = True
        # Also show in status
        self.status_pane.object = f"<p>🔄 {message}</p>"

    def _hide_loading(self):
        """Hide the blocking loading modal."""
        self.loading_modal.object = ""
        self.loading_modal.visible = False

    def _convert_video(self, _):
        """Convert current video to H.264 format."""
        if not self.selected_file or not self.selected_session:
            return

        try:
            # Show loading
            self._show_loading("Converting video to H.264...")
            self.convert_video_button.disabled = True

            # Get bucket and full path
            bucket, full_path = self.session_manager.get_file_path(
                self.selected_session, self.current_bucket_type, self.selected_file
            )

            # Convert video
            success, message, converted_path = self.file_manager.convert_video_to_h264(
                bucket, full_path
            )

            if success:
                self.status_pane.object = f"<p style='color:green'>✅ {message} - Use 'Replace in Cloud' button to upload</p>"
                # Replace the local cached file with the converted version
                self._replace_local_video_and_reload(bucket, full_path, converted_path)
            else:
                self.status_pane.object = f"<p style='color:red'>❌ {message}</p>"
                self.convert_video_button.disabled = False

        except Exception as e:
            logger.error(f"Error in video conversion: {e}")
            self.status_pane.object = (
                f"<p style='color:red'>❌ Conversion error: {e}</p>"
            )
            self.convert_video_button.disabled = False
        finally:
            self._hide_loading()

    def _replace_local_video_and_reload(
        self, bucket: str, full_path: str, converted_path: str
    ):
        """Replace local cached video with converted version and reload viewer."""
        try:
            # Get original cache path
            original_cache_path = self.file_manager.get_cache_path(bucket, full_path)

            # Replace the cached file with the converted version
            import shutil
            import os

            shutil.copy2(converted_path, original_cache_path)
            logger.info(
                f"Replaced cached video with H.264 version: {original_cache_path}"
            )

            # Clean up the temporary _h264 file to avoid duplicates
            try:
                os.remove(converted_path)
                logger.info(f"Cleaned up temporary converted file: {converted_path}")
            except Exception as cleanup_error:
                logger.warning(
                    f"Could not clean up temporary file {converted_path}: {cleanup_error}"
                )

            # Reload the current file to show the converted version
            self._on_file_select_by_path(self.selected_file)

            # Refresh the file list to show updated cache status
            self._refresh_file_list()

        except Exception as e:
            logger.error(f"Error replacing local video: {e}")
            self.status_pane.object = f"<p style='color:orange'>⚠️ Conversion successful but couldn't replace local file: {e}</p>"

    def _open_bbox_viewer(self, _):
        """Open the video bbox viewer using persistent Flask server."""
        if not self.selected_file or not self.selected_session:
            return

        if not hasattr(self, "_current_bbox_csvs") or not self._current_bbox_csvs:
            self.status_pane.object = (
                "<p style='color:orange'>⚠️ No bbox CSV files available</p>"
            )
            return

        try:
            import webbrowser
            import requests

            # Ensure persistent server is running
            if not self._ensure_persistent_server_running():
                return

            # Get video file info
            bucket, video_path = self.session_manager.get_file_path(
                self.selected_session, self.current_bucket_type, self.selected_file
            )

            # Get cached video path
            video_cache_path = self.file_manager.get_cache_path(bucket, video_path)

            if not Path(video_cache_path).exists():
                self.status_pane.object = (
                    "<p style='color:red'>❌ Video file not cached locally</p>"
                )
                return

            # Handle CSV selection
            bbox_csvs = self._current_bbox_csvs
            selected_csv = bbox_csvs[0]  # Use first CSV

            if len(bbox_csvs) > 1:
                display_name = selected_csv.get("display_name", selected_csv["name"])
                self.status_pane.object = f"<p style='color:blue'>📊 Using first CSV: {display_name} ({len(bbox_csvs)} available)</p>"

            # Get cached CSV path
            csv_cache_path = self.file_manager.get_cache_path(
                bucket, selected_csv["path"]
            )

            if not Path(csv_cache_path).exists():
                # Download the CSV file if not cached
                self._show_loading("Downloading CSV file...")
                try:
                    csv_content = self.rclone_client.read_file(
                        bucket, selected_csv["path"]
                    )
                    Path(csv_cache_path).write_bytes(csv_content)
                except Exception as e:
                    self.status_pane.object = (
                        f"<p style='color:red'>❌ Failed to download CSV: {e}</p>"
                    )
                    self._hide_loading()
                    return
                finally:
                    self._hide_loading()

            # Create unique video ID
            video_id = f"{Path(video_cache_path).stem}_{Path(csv_cache_path).stem}"

            self._show_loading("Adding video to persistent server...")

            try:
                # Add video to persistent server via HTTP API
                server_url = f"http://localhost:{self._persistent_flask_port}"

                # Call the server's API to add video
                add_video_data = {
                    "video_id": video_id,
                    "video_path": str(video_cache_path),
                    "csv_path": str(csv_cache_path),
                    "remote_path": f"{bucket}/{video_path}",  # Full remote path for display
                    "fps": 30.0,
                }

                logger.info(f"Sending video to persistent server: {add_video_data}")

                response = requests.post(
                    f"{server_url}/api/add_video", json=add_video_data, timeout=10
                )

                logger.info(
                    f"Server response: {response.status_code} - {response.text}"
                )

                if response.status_code != 200:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("error", "Unknown error")
                    except Exception:
                        error_msg = response.text
                    self.status_pane.object = (
                        f"<p style='color:red'>❌ Failed to add video: {error_msg}</p>"
                    )
                    return

                # Open browser to the persistent server
                url = f"http://localhost:{self._persistent_flask_port}"
                webbrowser.open(url)

                # Update status
                self.status_pane.object = f"""<p style='color:green'>✅ Video added to persistent viewer: <a href="{url}" target="_blank">{url}</a></p>
                <p><small>💡 Select "{Path(video_cache_path).name}" from the dropdown in the viewer. Server runs until dashboard restart.</small></p>"""

                # Show stop button
                if hasattr(self, "stop_all_viewers_button"):
                    self.stop_all_viewers_button.visible = True

            except Exception as e:
                self.status_pane.object = (
                    f"<p style='color:red'>❌ Failed to add video to server: {e}</p>"
                )
            finally:
                self._hide_loading()

        except Exception as e:
            logger.error(f"Error in bbox viewer: {e}")
            self.status_pane.object = (
                f"<p style='color:red'>❌ Error opening bbox viewer: {e}</p>"
            )
            self._hide_loading()

    def _open_mesh_3d_viewer(self, _):
        """Open the 3D mesh track viewer using persistent Flask server."""
        if not self.selected_file or not self.selected_session:
            return
        
        if not hasattr(self, "_current_csv_3d_files") or not self._current_csv_3d_files:
            self.status_pane.object = (
                "<p style='color:orange'>⚠️ No 3D CSV files available</p>"
            )
            return
        
        try:
            import webbrowser
            import requests
            
            # Ensure persistent server is running
            if not self._ensure_persistent_server_running():
                return
            
            # Get PLY file info
            bucket, ply_path = self.session_manager.get_file_path(
                self.selected_session, self.current_bucket_type, self.selected_file
            )
            
            # Get cached PLY path
            mesh_cache_path = self.file_manager.get_cache_path(bucket, ply_path)
            if not Path(mesh_cache_path).exists():
                self.status_pane.object = (
                    "<p style='color:red'>❌ PLY file not cached locally</p>"
                )
                return
            
            # Select best 3D CSV file (prefer rgb_1, then others)
            csv_3d_info = self._select_best_3d_csv(self._current_csv_3d_files)
            if not csv_3d_info:
                self.status_pane.object = (
                    "<p style='color:red'>❌ No valid 3D CSV files found</p>"
                )
                return
            
            csv_3d_path = csv_3d_info["path"]
            camera_info = csv_3d_info["camera"]
            
            # Show which CSV was selected if multiple options
            if len(self._current_csv_3d_files) > 1:
                self.status_pane.object = f"<p style='color:blue'>📊 Using {csv_3d_info['display_name']} ({len(self._current_csv_3d_files)} available)</p>"
            
            # Get cached 3D CSV path
            csv_3d_cache_path = self.file_manager.get_cache_path(bucket, csv_3d_path)
            if not Path(csv_3d_cache_path).exists():
                # Download the 3D CSV file if not cached
                self._show_loading("Downloading 3D CSV file...")
                try:
                    csv_content = self.rclone_client.read_file(bucket, csv_3d_path)
                    Path(csv_3d_cache_path).write_bytes(csv_content)
                except Exception as e:
                    self.status_pane.object = (
                        f"<p style='color:red'>❌ Failed to download 3D CSV: {e}</p>"
                    )
                    self._hide_loading()
                    return
                finally:
                    self._hide_loading()
            
            # Get camera params if available
            camera_params_cache_path = None
            if self._current_camera_params_file:
                # self._current_camera_params_file already contains the full path from _find_camera_params_with_bucket
                camera_params_path = self._current_camera_params_file
                camera_params_cache_path = self.file_manager.get_cache_path(bucket, camera_params_path)
                logger.info(f"🔍 Camera params path construction:")
                logger.info(f"  📁 PLY path: {ply_path}")
                logger.info(f"  📄 Camera file: {self._current_camera_params_file}")
                logger.info(f"  🔗 Full camera path: {camera_params_path}")
                logger.info(f"  💾 Cache path: {camera_params_cache_path}")
                
                if not Path(camera_params_cache_path).exists():
                    # Download camera params file if not cached
                    self._show_loading("Downloading camera parameters...")
                    try:
                        logger.info(f"📷 Downloading camera params: {bucket}/{camera_params_path}")
                        params_content = self.rclone_client.read_file(bucket, camera_params_path)
                        logger.info(f"📏 Downloaded {len(params_content)} bytes of camera params")
                        Path(camera_params_cache_path).write_bytes(params_content)
                        logger.info(f"✅ Camera params cached to: {camera_params_cache_path}")
                    except Exception as e:
                        logger.error(f"❌ Failed to download camera params {bucket}/{camera_params_path}: {e}")
                        camera_params_cache_path = None
                    finally:
                        self._hide_loading()
                else:
                    logger.info(f"📋 Camera params already cached: {camera_params_cache_path}")
            
            # Create unique mesh ID including camera info
            mesh_id = f"{Path(mesh_cache_path).stem}_{camera_info}_{Path(csv_3d_cache_path).stem}"
            
            self._show_loading("Adding mesh to 3D viewer...")
            try:
                # Add mesh to persistent server via HTTP API
                server_url = f"http://localhost:{self._persistent_flask_port}"
                
                # Call the server's API to add mesh
                add_mesh_data = {
                    "mesh_id": mesh_id,
                    "mesh_path": str(mesh_cache_path),
                    "csv_3d_path": str(csv_3d_cache_path),
                    "camera_params_path": str(camera_params_cache_path) if camera_params_cache_path else None,
                    "remote_path": f"{bucket}/{ply_path}",  # Full remote path for display
                }
                
                logger.info(f"Sending mesh to persistent server: {add_mesh_data}")
                response = requests.post(
                    f"{server_url}/api/add_mesh", json=add_mesh_data, timeout=10
                )
                logger.info(f"Server response: {response.status_code} - {response.text}")
                
                if response.status_code != 200:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("error", "Unknown error")
                    except Exception:
                        error_msg = response.text
                    
                    self.status_pane.object = (
                        f"<p style='color:red'>❌ Failed to add mesh: {error_msg}</p>"
                    )
                    return
                
                # Open browser to the 3D viewer
                url = f"http://localhost:{self._persistent_flask_port}/3d"
                webbrowser.open(url)
                
                # Update status
                self.status_pane.object = f"""<p style='color:green'>✅ Mesh added to 3D viewer: <a href="{url}" target="_blank">{url}</a></p>
                <p><small>💡 Select "{Path(mesh_cache_path).name}" from the dropdown in the 3D viewer. Interactive controls available!</small></p>"""
                
                # Show stop button
                if hasattr(self, "stop_all_viewers_button"):
                    self.stop_all_viewers_button.visible = True
                    
            except Exception as e:
                self.status_pane.object = (
                    f"<p style='color:red'>❌ Failed to add mesh to server: {e}</p>"
                )
            finally:
                self._hide_loading()
                
        except Exception as e:
            logger.error(f"Error in 3D mesh viewer: {e}")
            self.status_pane.object = (
                f"<p style='color:red'>❌ Error opening 3D viewer: {e}</p>"
            )
            self._hide_loading()

    def _select_best_3d_csv(self, csv_3d_files: list) -> Optional[dict]:
        """Select the best 3D CSV file when multiple are available."""
        if not csv_3d_files:
            return None
        
        # If only one file, use it
        if len(csv_3d_files) == 1:
            return csv_3d_files[0]
        
        # Priority order: rgb_1, rgb_2, thermal_1, thermal_2, unknown
        priority_order = ["rgb_1", "rgb_2", "thermal_1", "thermal_2", "unknown"]
        
        for preferred_camera in priority_order:
            for csv_file in csv_3d_files:
                if csv_file["camera"] == preferred_camera:
                    logger.info(f"Selected 3D CSV: {csv_file['display_name']} (priority: {preferred_camera})")
                    return csv_file
        
        # Fallback to first file if no matches
        logger.info(f"No priority match, using first file: {csv_3d_files[0]['display_name']}")
        return csv_3d_files[0]

    def _ensure_persistent_server_running(self) -> bool:
        """Ensure the persistent Flask server is running."""
        try:
            import requests

            # Check if server is already running and has the API endpoints
            try:
                response = requests.get(
                    f"http://localhost:{self._persistent_flask_port}/api/health",
                    timeout=2,
                )
                if response.status_code == 200:
                    logger.info("Persistent server already running with API endpoints")
                    return True
                else:
                    logger.warning(
                        f"Server responding but health check failed: {response.status_code}"
                    )
            except requests.exceptions.RequestException:
                logger.info("No server responding or server without API endpoints")
                pass

            # Server not running properly, stop any existing process and start fresh
            if (
                self._persistent_flask_process
                and self._persistent_flask_process.poll() is None
            ):
                # Process exists but not responding properly, kill it
                logger.info("Stopping existing server process")
                self._persistent_flask_process.terminate()
                try:
                    self._persistent_flask_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._persistent_flask_process.kill()
                    self._persistent_flask_process.wait()

            # Start new server
            flask_app_path = Path(__file__).parent / "persistent_video_server.py"

            if not flask_app_path.exists():
                self.status_pane.object = f"<p style='color:red'>❌ Persistent server not found: {flask_app_path}</p>"
                return False

            self.status_pane.object = f"<p style='color:blue'>🚀 Starting persistent video server on port {self._persistent_flask_port}...</p>"

            # Start persistent server
            self._persistent_flask_process = subprocess.Popen(
                [
                    "python",
                    str(flask_app_path),
                    "--port",
                    str(self._persistent_flask_port),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Wait for server to start
            import time

            for attempt in range(10):
                time.sleep(0.5)
                try:
                    response = requests.get(
                        f"http://localhost:{self._persistent_flask_port}", timeout=1
                    )
                    if response.status_code == 200:
                        logger.info(
                            f"Persistent Flask server started on port {self._persistent_flask_port}"
                        )
                        return True
                except Exception:
                    continue

            # Server failed to start
            self.status_pane.object = (
                "<p style='color:red'>❌ Failed to start persistent server</p>"
            )
            return False

        except Exception as e:
            logger.error(f"Error ensuring persistent server: {e}")
            self.status_pane.object = (
                f"<p style='color:red'>❌ Error starting server: {e}</p>"
            )
            return False

    def _stop_all_bbox_viewers(self, _=None):
        """Stop the persistent Flask server and all related processes."""
        stopped_count = 0

        # First, try to get current server state and clear data via API
        api_success = False
        try:
            # Check what's currently on the server before clearing
            health_response = requests.get(
                f"http://localhost:{self._persistent_flask_port}/api/health", timeout=2
            )
            
            current_videos = 0
            current_meshes = 0
            if health_response.status_code == 200:
                health_data = health_response.json()
                current_videos = health_data.get('videos_count', 0)
                current_meshes = health_data.get('meshes_count', 0)
                logger.info(f"Server currently has {current_videos} videos and {current_meshes} meshes")
            
            # Now clear the data
            clear_response = requests.post(
                f"http://localhost:{self._persistent_flask_port}/api/clear", timeout=2
            )
            if clear_response.status_code == 200:
                data = clear_response.json()
                cleared_videos = data.get('cleared_videos', 0)
                cleared_meshes = data.get('cleared_meshes', 0)
                total_cleared = data.get('total_cleared', cleared_videos + cleared_meshes)
                
                logger.info(f"Cleared {cleared_videos} videos and {cleared_meshes} meshes from server (total: {total_cleared})")
                
                if total_cleared > 0:
                    self.status_pane.object = f"<p style='color:green'>✅ Cleared {cleared_videos} videos and {cleared_meshes} meshes from server</p>"
                else:
                    self.status_pane.object = "<p style='color:blue'>ℹ️ Server was already empty</p>"
                api_success = True
            else:
                logger.warning(f"Clear API failed with status {clear_response.status_code}")
                
        except Exception as e:
            logger.info(f"Could not clear server data (server may already be down): {e}")
            # Show fallback status based on what we detected before clearing
            if 'current_videos' in locals() or 'current_meshes' in locals():
                fallback_videos = locals().get('current_videos', 0)
                fallback_meshes = locals().get('current_meshes', 0) 
                if fallback_videos > 0 or fallback_meshes > 0:
                    self.status_pane.object = f"<p style='color:orange'>⚠️ Server had {fallback_videos} videos and {fallback_meshes} meshes but couldn't clear them (server may be down)</p>"

        # Stop our tracked process
        if self._persistent_flask_process:
            try:
                pid = self._persistent_flask_process.pid
                if self._persistent_flask_process.poll() is None:  # Still running
                    logger.info(f"Terminating Flask server process (PID: {pid})")

                    # Try graceful termination first
                    self._persistent_flask_process.terminate()
                    try:
                        self._persistent_flask_process.wait(timeout=3)
                        logger.info(f"Flask server terminated gracefully (PID: {pid})")
                        stopped_count += 1
                    except subprocess.TimeoutExpired:
                        # Force kill if needed
                        logger.warning(f"Force killing Flask server (PID: {pid})")
                        self._persistent_flask_process.kill()
                        self._persistent_flask_process.wait()
                        stopped_count += 1

            except Exception as e:
                logger.warning(f"Error stopping tracked Flask process: {e}")

        # Also kill any rogue Flask processes on our port (belt and suspenders approach)
        try:
            import psutil

            killed_rogue = 0
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    if proc.info["name"] == "python" and proc.info["cmdline"]:
                        cmdline = " ".join(proc.info["cmdline"])
                        if (
                            "persistent_video_server.py" in cmdline
                            and f"--port {self._persistent_flask_port}" in cmdline
                        ):
                            logger.warning(
                                f"Found rogue Flask server process (PID: {proc.info['pid']}), killing it"
                            )
                            proc.kill()
                            killed_rogue += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            if killed_rogue > 0:
                stopped_count += killed_rogue
                logger.info(f"Killed {killed_rogue} rogue Flask processes")

        except ImportError:
            logger.info("psutil not available, skipping rogue process cleanup")
        except Exception as e:
            logger.warning(f"Error during rogue process cleanup: {e}")

        # Clean up state
        self._persistent_flask_process = None

        # Hide stop button
        if hasattr(self, "stop_all_viewers_button"):
            self.stop_all_viewers_button.visible = False

        # Update status (only if we didn't already set it from API clear)
        if not api_success:
            if stopped_count > 0:
                self.status_pane.object = f"<p style='color:green'>✅ Stopped {stopped_count} Flask server process(es)</p>"
            else:
                self.status_pane.object = (
                    "<p style='color:blue'>ℹ️ No Flask server processes were running</p>"
                )

    def _update_file_management_buttons(self):
        """Update visibility and state of file management buttons based on current file."""
        has_file = bool(self.selected_file and self.selected_session)

        # Show the entire file management controls container when a file is loaded
        self.file_management_controls.visible = has_file

        if has_file:
            # Check if file is modified to show/hide buttons appropriately
            try:
                bucket, full_path = self.session_manager.get_file_path(
                    self.selected_session, self.current_bucket_type, self.selected_file
                )
                is_modified = self.file_manager.is_file_modified(bucket, full_path)

                # Show replace button only if file is modified
                self.replace_file_button.visible = is_modified
                if is_modified:
                    self.replace_file_button.name = "Replace in Cloud (Modified)"
                else:
                    self.replace_file_button.name = "Replace in Cloud"

                # Show download original button only if file is modified
                self.download_original_button.visible = is_modified

            except Exception as e:
                logger.warning(f"Error checking file modification status: {e}")
                # Default to visible if we can't check
                self.replace_file_button.visible = True
                self.replace_file_button.name = "Replace in Cloud"
                self.download_original_button.visible = False
        else:
            # Hide individual buttons when no file is selected
            self.replace_file_button.visible = False
            self.download_original_button.visible = False

    def _replace_file_in_cloud(self, _):
        """Replace the current file in cloud with the cached version."""
        if not self.selected_file or not self.selected_session:
            return

        try:
            # Show loading
            self._show_loading("Replacing file in cloud...")
            self.replace_file_button.disabled = True

            # Get bucket and full path
            bucket, full_path = self.session_manager.get_file_path(
                self.selected_session, self.current_bucket_type, self.selected_file
            )

            # Replace file
            success, message = self.file_manager.replace_file_from_cache(
                bucket, full_path
            )

            if success:
                self.status_pane.object = f"<p style='color:green'>✅ {message}</p>"
                # Refresh the file list to show the _old backup file
                self._refresh_file_list()
            else:
                self.status_pane.object = f"<p style='color:red'>❌ {message}</p>"

        except Exception as e:
            logger.error(f"Error replacing file in cloud: {e}")
            self.status_pane.object = f"<p style='color:red'>❌ Replace error: {e}</p>"
        finally:
            self.replace_file_button.disabled = False
            self._hide_loading()

    def _download_original_file(self, _):
        """Download the original file from cloud, overwriting local cache."""
        if not self.selected_file or not self.selected_session:
            return

        try:
            # Show loading
            self._show_loading("Downloading original file from cloud...")
            self.download_original_button.disabled = True

            # Get bucket and full path
            bucket, full_path = self.session_manager.get_file_path(
                self.selected_session, self.current_bucket_type, self.selected_file
            )

            # Download original file from cloud and overwrite cache
            try:
                content = self.file_manager.client.read_file(bucket, full_path)
                cache_path = self.file_manager._get_cache_path(bucket, full_path)
                cache_path.write_bytes(content)
                logger.info(f"Downloaded original file to cache: {cache_path}")

                # Reload the file to show original content
                self._on_file_select_by_path(self.selected_file)

                # Refresh the file list to show updated cache status
                self._refresh_file_list()

                self.status_pane.object = "<p style='color:green'>✅ Downloaded original version from cloud</p>"

            except Exception as e:
                logger.error(f"Error downloading original file: {e}")
                self.status_pane.object = (
                    f"<p style='color:red'>❌ Failed to download original: {e}</p>"
                )

        except Exception as e:
            logger.error(f"Error in download original: {e}")
            self.status_pane.object = f"<p style='color:red'>❌ Download error: {e}</p>"
        finally:
            self.download_original_button.disabled = False
            self._hide_loading()

    def _show_modal(self, file_name: str):
        """Show the deletion confirmation modal."""
        self.modal_message.object = f"<p style='margin: 15px 0;'>Are you sure you want to delete <strong>'{file_name}'</strong>?<br/><span style='color: #666; font-size: 0.9em;'>This will remove it from both local cache and cloud storage.</span></p>"
        self.modal_dialog.visible = True

    def _hide_modal(self):
        """Hide the deletion confirmation modal."""
        self.modal_dialog.visible = False

    def _confirm_delete(self, _):
        """Handle confirm delete button click."""
        self._hide_modal()
        self._perform_file_deletion()

    def _cancel_delete(self, _):
        """Handle cancel delete button click."""
        self._hide_modal()
        self.status_pane.object = "<p>Deletion cancelled</p>"

    def _delete_file(self, _):
        """Delete the current file from cloud and cache with confirmation."""
        if not self.selected_file or not self.selected_session:
            return

        # Show modal confirmation dialog
        self._show_modal(self.selected_file)

    def _perform_file_deletion(self):
        """Actually perform the file deletion after confirmation."""
        try:
            # Show loading
            self._show_loading("Deleting file...")
            self.delete_file_button.disabled = True

            # Get bucket and full path
            bucket, full_path = self.session_manager.get_file_path(
                self.selected_session, self.current_bucket_type, self.selected_file
            )

            # Delete file
            success, message = self.file_manager.delete_file(bucket, full_path)

            if success:
                self.status_pane.object = f"<p style='color:green'>✅ {message}</p>"
                # Clear the current file selection
                self.selected_file = ""
                self.file_viewer.object = "<p>File deleted</p>"
                self.file_name_header.object = "<h3>File deleted</h3>"
                # Hide file management buttons
                self._update_file_management_buttons()
                # Refresh the file list
                self._refresh_file_list()
            else:
                self.status_pane.object = f"<p style='color:red'>❌ {message}</p>"

        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            self.status_pane.object = f"<p style='color:red'>❌ Delete error: {e}</p>"
        finally:
            self.delete_file_button.disabled = False
            self._hide_loading()

    def _refresh_file_list(self):
        """Refresh the file list to show updated files (including _old backups)."""
        try:
            if self.selected_session:
                session = self.session_manager.load_session_files(self.selected_session)
                if session:
                    self._update_file_tree(session)
                    self.status_pane.object = "<p>📄 File list refreshed</p>"
        except Exception as e:
            logger.error(f"Error refreshing file list: {e}")

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

        # Video controls (conversion, bbox viewer, and stop buttons)
        self.video_controls = pn.Row(
            self.convert_video_button,
            self.bbox_viewer_button,
            self.mesh_3d_viewer_button,
            self.stop_all_viewers_button,
            margin=(10, 0),
        )

        # Main content area with file name header
        # Create a container for the view content (VTK container moved outside tabs)
        self.view_container = pn.Column(
            self.file_viewer,
            self.video_controls,
        )

        # Create tabs without VTK to avoid DOM manipulation issues
        viewer_tabs = pn.Tabs(
            ("View", self.view_container),
            (
                "Edit",
                pn.Column(
                    pn.Row(self.edit_button, self.save_button, self.cancel_edit_button),
                    self.file_editor,
                ),
            ),
            dynamic=True,  # Re-enable dynamic tabs since VTK is outside
        )

        content_panel = pn.Column(
            self.file_name_header,
            self.file_management_controls,
            viewer_tabs,
            self.persistent_vtk_container,  # VTK container outside tabs to avoid DOM issues
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

        # Create a simple modal dialog using Panel's overlay approach
        self.modal_message = pn.pane.HTML("")

        # Simple modal dialog content - no complex CSS positioning
        self.modal_dialog = pn.Column(
            pn.pane.HTML(
                "<h3>⚠️ Confirm Deletion</h3>",
                styles={"color": "#d32f2f", "margin": "0 0 10px 0"},
            ),
            self.modal_message,
            pn.Row(
                self.confirm_delete_button,
                pn.Spacer(width=20),
                self.cancel_delete_button,
                align="center",
            ),
            margin=20,
            width=400,
            height=180,
            styles={
                "background": "white",
                "border": "3px solid #d32f2f",
                "border-radius": "10px",
                "padding": "20px",
                "box-shadow": "0 8px 24px rgba(0, 0, 0, 0.3)",
            },
            visible=False,
        )

        return pn.template.MaterialTemplate(
            title="CIS Data Dashboard",
            sidebar=[nav_panel],
            main=[
                # Modal dialogs at top for visibility
                self.loading_modal,
                self.modal_dialog,
                status_panel,
                content_panel,
            ],
            header_background="#2596be",
            sidebar_width=600,  # Much wider sidebar for file tree
        )


def create_app(
    remote_name: Optional[str] = None,
    curated_bucket: Optional[str] = None,
    processed_bucket: Optional[str] = None,
):
    """Create and return the dashboard application."""
    try:
        dashboard = DataDashboard(
            remote_name=remote_name,
            curated_bucket=curated_bucket,
            processed_bucket=processed_bucket,
        )
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


def serve_dashboard(
    port: int = 5007,
    show: bool = True,
    autoreload: bool = True,
    remote_name: Optional[str] = None,
    curated_bucket: Optional[str] = None,
    processed_bucket: Optional[str] = None,
):
    """Serve the dashboard application."""
    app = create_app(
        remote_name=remote_name,
        curated_bucket=curated_bucket,
        processed_bucket=processed_bucket,
    )

    return pn.serve(
        app,
        port=port,
        show=show,
        title="GCS Data Dashboard",
        autoreload=autoreload,
    )
