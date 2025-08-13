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
            width=550,  # Width for container
            height=300,  # Height
            visible=False,  # Hidden initially until session selected
            stylesheets=["""
                select {
                    overflow-x: auto !important;
                    white-space: nowrap !important;
                }
                option {
                    white-space: nowrap !important;
                }
            """]
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
            visible=False  # Initially hidden until file is loaded
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
        self.convert_video_button.on_click(self._convert_video)
        self.replace_file_button.on_click(self._replace_file_in_cloud)
        self.download_original_button.on_click(self._download_original_file)
        self.delete_file_button.on_click(self._delete_file)

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
            self._update_file_management_buttons()
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
            render_info["cache_path"] = self.file_manager.get_cache_path(bucket, full_path)

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
        cache_status = "💾 Cached" if render_info.get("from_cache", False) else "📡 Downloaded"
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
            except Exception as e:
                modification_status = '<div><strong>Modified:</strong> <span style="color: #6c757d;">❓ Unknown</span></div>'
        
        return f"""
        <div style="background-color: #f0f0f0; padding: 10px; margin-bottom: 10px; border-radius: 4px; font-size: 12px;">
            <div><strong>Status:</strong> {cache_status}</div>
            {modification_status}
            <div><strong>Remote:</strong> <code>{remote_path}</code></div>
            <div><strong>Cache:</strong> <code>{cache_path}</code></div>
        </div>
        """

    def _update_file_viewer(self, render_info: Dict[str, Any]):
        """Update the file viewer based on render info."""
        if render_info["type"] == "text":
            # Text content with syntax highlighting
            content = render_info["content"]
            # language = render_info.get("language", "text")  # Available for future syntax highlighting
            path_info = self._get_file_path_info(render_info)

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
            needs_conversion = (browser_compatible in [False, "limited"] and 
                              codec_name.lower() not in safe_codecs)
            self.convert_video_button.visible = needs_conversion
            self.convert_video_button.disabled = False
            
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
                compatibility_warning = f"""
                <div style="background-color: #d4edda; border: 1px solid #c3e6cb; padding: 5px 10px; margin: 10px 0; border-radius: 4px; font-size: 12px;">
                    ✅ Widely supported codec
                </div>
                """
            
            if size_mb > 50:  # Large video warning
                self.file_viewer.object = f"""
                <div>
                    {path_info}
                    <h4>Video File ({size_mb:.1f} MB)</h4>
                    {codec_details}
                    {compatibility_warning}
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
                    <video controls width="100%" height="400">
                        <source src="{render_info["data_url"]}" type="{render_info["mime_type"]}">
                        Your browser does not support video playback.
                    </video>
                </div>
                """

        elif render_info["type"] == "error":
            path_info = self._get_file_path_info(render_info)
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
                content, render_info = self.file_manager.get_file_content_with_progress(bucket, full_path)
                self.current_file_content = content
                self.current_file_info = render_info
                self._update_file_viewer(render_info)

            # Exit edit mode
            self._exit_edit_mode()

            self.status_pane.object = (
                f"<p style='color:green'>💾 Saved to local cache: {self.selected_file} - Use 'Replace in Cloud' to upload</p>"
            )

        except Exception as e:
            logger.error(f"Error saving file to cache: {e}")
            self.status_pane.object = f"<p style='color:red'>Error saving file to cache: {e}</p>"

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
        is_cached = self.file_manager.is_file_cached(session_name, bucket_type, file_path)
        
        if not is_cached:
            return "📡"  # Remote only
        
        # File is cached, check if modified
        try:
            # Construct bucket name and full path for modification check
            bucket = f"fieldwork_{bucket_type}"
            if bucket_type == "curated":
                full_path = f"{session_name}/{file_path.strip('/')}"
            else:  # processed
                full_path = f"{session_name}/{file_path.strip('/')}"
            
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
            success, message, converted_path = self.file_manager.convert_video_to_h264(bucket, full_path)
            
            if success:
                self.status_pane.object = f"<p style='color:green'>✅ {message} - Use 'Replace in Cloud' button to upload</p>"
                # Replace the local cached file with the converted version  
                self._replace_local_video_and_reload(bucket, full_path, converted_path)
            else:
                self.status_pane.object = f"<p style='color:red'>❌ {message}</p>"
                self.convert_video_button.disabled = False
            
        except Exception as e:
            logger.error(f"Error in video conversion: {e}")
            self.status_pane.object = f"<p style='color:red'>❌ Conversion error: {e}</p>"
            self.convert_video_button.disabled = False
        finally:
            self._hide_loading()
    
    def _replace_local_video_and_reload(self, bucket: str, full_path: str, converted_path: str):
        """Replace local cached video with converted version and reload viewer."""
        try:
            # Get original cache path
            original_cache_path = self.file_manager.get_cache_path(bucket, full_path)
            
            # Replace the cached file with the converted version
            import shutil
            import os
            shutil.copy2(converted_path, original_cache_path)
            logger.info(f"Replaced cached video with H.264 version: {original_cache_path}")
            
            # Clean up the temporary _h264 file to avoid duplicates
            try:
                os.remove(converted_path)
                logger.info(f"Cleaned up temporary converted file: {converted_path}")
            except Exception as cleanup_error:
                logger.warning(f"Could not clean up temporary file {converted_path}: {cleanup_error}")
            
            # Reload the current file to show the converted version
            self._on_file_select_by_path(self.selected_file)
            
        except Exception as e:
            logger.error(f"Error replacing local video: {e}")
            self.status_pane.object = f"<p style='color:orange'>⚠️ Conversion successful but couldn't replace local file: {e}</p>"
    
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
            success, message = self.file_manager.replace_file_from_cache(bucket, full_path)
            
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
                
                self.status_pane.object = f"<p style='color:green'>✅ Downloaded original version from cloud</p>"
                
            except Exception as e:
                logger.error(f"Error downloading original file: {e}")
                self.status_pane.object = f"<p style='color:red'>❌ Failed to download original: {e}</p>"
                
        except Exception as e:
            logger.error(f"Error in download original: {e}")
            self.status_pane.object = f"<p style='color:red'>❌ Download error: {e}</p>"
        finally:
            self.download_original_button.disabled = False
            self._hide_loading()

    def _delete_file(self, _):
        """Delete the current file from cloud and cache with confirmation."""
        if not self.selected_file or not self.selected_session:
            return
        
        # Create confirmation dialog content
        confirm_html = f"""
        <div style="padding: 20px; text-align: center;">
            <h3 style="color: #d63384;">⚠️ Confirm File Deletion</h3>
            <p>Are you sure you want to <strong>permanently delete</strong>:</p>
            <p><code>{self.selected_file}</code></p>
            <p style="color: #6c757d; font-size: 12px;">This will remove the file from both cloud storage and local cache.</p>
        </div>
        """
        
        # Create confirmation dialog
        confirm_pane = pn.pane.HTML(confirm_html, width=400, height=150)
        confirm_button = pn.widgets.Button(name="Delete", button_type="danger", width=100)
        cancel_button = pn.widgets.Button(name="Cancel", button_type="primary", width=100)
        
        def confirm_delete(_):
            dialog.close()
            self._perform_file_deletion()
        
        def cancel_delete(_):
            dialog.close()
        
        confirm_button.on_click(confirm_delete)
        cancel_button.on_click(cancel_delete)
        
        button_row = pn.Row(cancel_button, confirm_button, margin=(10, 0))
        dialog_content = pn.Column(confirm_pane, button_row)
        
        # Show modal dialog
        dialog = pn.layout.Modal(dialog_content, title="Confirm Deletion")
        dialog.open()
    
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
                    self.status_pane.object = f"<p>📄 File list refreshed</p>"
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

        # Video conversion controls (only video-specific buttons)
        video_controls = pn.Row(
            self.convert_video_button,
            margin=(10, 0)
        )
        

        # Main content area with file name header
        viewer_tabs = pn.Tabs(
            ("View", pn.Column(self.file_viewer, video_controls)),
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
            self.file_management_controls,
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
