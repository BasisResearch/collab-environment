"""
File viewers for different file types in the dashboard.
"""

import io
import pandas as pd
import yaml
import os
import hashlib
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import base64
from loguru import logger

from .rclone_client import RcloneClient


class FileViewerRegistry:
    """Registry for file viewers by file extension."""

    def __init__(self):
        self.viewers = {}
        self._register_default_viewers()

    def _register_default_viewers(self):
        """Register default file viewers."""
        # Text viewers
        text_viewer = TextViewer()
        for ext in [".yml", ".yaml", ".txt", ".xml", ".json", ".md", ".rst"]:
            self.viewers[ext] = text_viewer

        # Table viewers
        table_viewer = TableViewer()
        for ext in [".csv", ".parquet"]:
            self.viewers[ext] = table_viewer

        # Video viewer
        video_viewer = VideoViewer()
        for ext in [".mp4", ".avi", ".mov", ".mkv"]:
            self.viewers[ext] = video_viewer

    def get_viewer(self, file_path: str) -> Optional["BaseViewer"]:
        """Get viewer for a file based on its extension."""
        ext = Path(file_path).suffix.lower()
        return self.viewers.get(ext)

    def register_viewer(self, extensions: list, viewer: "BaseViewer"):
        """Register a custom viewer for file extensions."""
        for ext in extensions:
            self.viewers[ext.lower()] = viewer


class BaseViewer:
    """Base class for file viewers."""

    def can_edit(self) -> bool:
        """Return True if this viewer supports editing."""
        return False

    def can_view(self) -> bool:
        """Return True if this viewer supports viewing."""
        return True

    def render_view(self, content: bytes, file_path: str) -> Dict[str, Any]:
        """
        Render file content for viewing.

        Args:
            content: Raw file content
            file_path: Path to the file

        Returns:
            Dictionary with rendering information
        """
        raise NotImplementedError

    def prepare_edit(self, content: bytes, file_path: str) -> str:
        """
        Prepare content for editing.

        Args:
            content: Raw file content
            file_path: Path to the file

        Returns:
            String content ready for editing
        """
        if not self.can_edit():
            raise NotImplementedError("This viewer does not support editing")
        return content.decode("utf-8")

    def process_edit(self, edited_content: str, file_path: str) -> bytes:
        """
        Process edited content back to bytes.

        Args:
            edited_content: Edited content as string
            file_path: Path to the file

        Returns:
            Processed content as bytes
        """
        if not self.can_edit():
            raise NotImplementedError("This viewer does not support editing")
        return edited_content.encode("utf-8")


class TextViewer(BaseViewer):
    """Viewer for text-based files (YAML, TXT, XML, etc.)."""

    def can_edit(self) -> bool:
        return True

    def render_view(self, content: bytes, file_path: str) -> Dict[str, Any]:
        """Render text content with syntax highlighting hints."""
        try:
            text_content = content.decode("utf-8")
            file_ext = Path(file_path).suffix.lower()

            # Determine language for syntax highlighting
            language_map = {
                ".yml": "yaml",
                ".yaml": "yaml",
                ".json": "json",
                ".xml": "xml",
                ".md": "markdown",
                ".rst": "rst",
                ".txt": "text",
            }

            language = language_map.get(file_ext, "text")

            # For YAML files, also parse and validate
            parsed_data = None
            if file_ext in [".yml", ".yaml"]:
                try:
                    parsed_data = yaml.safe_load(text_content)
                except yaml.YAMLError as e:
                    logger.warning(f"YAML parsing error in {file_path}: {e}")

            return {
                "type": "text",
                "content": text_content,
                "language": language,
                "parsed_data": parsed_data,
                "size": len(content),
                "lines": text_content.count("\n") + 1,
            }

        except UnicodeDecodeError as e:
            return {
                "type": "error",
                "message": f"Cannot decode file as text: {e}",
                "size": len(content),
            }


class TableViewer(BaseViewer):
    """Viewer for tabular data files (CSV, Parquet)."""

    def render_view(self, content: bytes, file_path: str) -> Dict[str, Any]:
        """Render tabular data as HTML table."""
        try:
            file_ext = Path(file_path).suffix.lower()

            # Read data based on file type
            if file_ext == ".csv":
                df = pd.read_csv(io.BytesIO(content))
            elif file_ext == ".parquet":
                df = pd.read_parquet(io.BytesIO(content))
            else:
                raise ValueError(f"Unsupported table format: {file_ext}")

            # Basic statistics
            stats = {
                "rows": len(df),
                "columns": len(df.columns),
                "size": len(content),
                "column_names": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
            }

            # Convert to HTML (limit rows for display)
            display_limit = 1000
            if len(df) > display_limit:
                display_df = df.head(display_limit)
                truncated = True
            else:
                display_df = df
                truncated = False

            html_table = display_df.to_html(
                classes="table table-striped table-hover",
                table_id="data-table",
                escape=False,
                max_rows=display_limit,
            )

            return {
                "type": "table",
                "html": html_table,
                "stats": stats,
                "truncated": truncated,
                "display_limit": display_limit if truncated else len(df),
            }

        except Exception as e:
            logger.error(f"Error rendering table {file_path}: {e}")
            return {
                "type": "error",
                "message": f"Cannot render table: {e}",
                "size": len(content),
            }


class VideoViewer(BaseViewer):
    """Viewer for video files."""

    def render_view(self, content: bytes, file_path: str) -> Dict[str, Any]:
        """Render video using HTML5 video element."""
        try:
            file_ext = Path(file_path).suffix.lower()

            # Determine MIME type
            mime_types = {
                ".mp4": "video/mp4",
                ".avi": "video/x-msvideo",
                ".mov": "video/quicktime",
                ".mkv": "video/x-matroska",
            }

            mime_type = mime_types.get(file_ext, "video/mp4")

            # Encode video content as base64 for embedding
            # Note: This is not ideal for large videos - consider using rclone serve http
            video_b64 = base64.b64encode(content).decode("utf-8")
            data_url = f"data:{mime_type};base64,{video_b64}"

            return {
                "type": "video",
                "data_url": data_url,
                "mime_type": mime_type,
                "size": len(content),
                "size_mb": len(content) / (1024 * 1024),
            }

        except Exception as e:
            logger.error(f"Error rendering video {file_path}: {e}")
            return {
                "type": "error",
                "message": f"Cannot render video: {e}",
                "size": len(content),
            }


class FileContentManager:
    """Manages file viewing and editing operations."""

    def __init__(self, rclone_client: RcloneClient, cache_dir: Optional[str] = None):
        self.client = rclone_client
        self.viewer_registry = FileViewerRegistry()
        
        # Setup cache directory
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/collab_env_dashboard")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Dashboard cache directory: {self.cache_dir}")
    
    def _get_cache_path(self, bucket: str, file_path: str) -> Path:
        """Get cache file path for a bucket/file combination."""
        # Create a safe filename from bucket and file path
        safe_name = hashlib.md5(f"{bucket}/{file_path}".encode()).hexdigest()
        file_ext = Path(file_path).suffix
        cache_filename = f"{safe_name}{file_ext}"
        return self.cache_dir / cache_filename
    
    def is_file_cached(self, session_name: str, bucket_type: str, file_path: str) -> bool:
        """Check if a file is cached locally."""
        # Construct bucket name from session and type info
        bucket = f"fieldwork_{bucket_type}"  # Simplified for now
        cache_path = self._get_cache_path(bucket, file_path)
        return cache_path.exists()
    
    def get_cache_location(self) -> str:
        """Get the cache directory location."""
        return str(self.cache_dir)
    
    def is_file_supported(self, file_path: str) -> bool:
        """Check if a file type is supported by the dashboard."""
        viewer = self.viewer_registry.get_viewer(file_path)
        return viewer is not None

    def get_file_content_with_progress(
        self, bucket: str, file_path: str
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Get file content with caching and progress tracking.

        Args:
            bucket: Bucket name
            file_path: Path to file

        Returns:
            Tuple of (raw_content, render_info)
        """
        cache_path = self._get_cache_path(bucket, file_path)
        from_cache = False
        
        try:
            # Try to load from cache first
            if cache_path.exists():
                logger.info(f"Loading {file_path} from cache")
                content = cache_path.read_bytes()
                from_cache = True
            else:
                # Download from remote and cache
                logger.info(f"Downloading {file_path} from {bucket}")
                content = self.client.read_file(bucket, file_path)
                
                # Save to cache
                cache_path.write_bytes(content)
                logger.info(f"Cached {file_path} to {cache_path}")
            
            viewer = self.viewer_registry.get_viewer(file_path)

            if viewer:
                render_info = viewer.render_view(content, file_path)
                render_info["viewer_available"] = True
                render_info["can_edit"] = viewer.can_edit()
                render_info["from_cache"] = from_cache
            else:
                # Unknown file type
                render_info = {
                    "type": "unknown",
                    "size": len(content),
                    "viewer_available": False,
                    "can_edit": False,
                    "from_cache": from_cache,
                    "message": f"No viewer available for {Path(file_path).suffix}",
                }

            return content, render_info

        except Exception as e:
            logger.error(f"Error getting file content {bucket}/{file_path}: {e}")
            raise

    def get_file_content(
        self, bucket: str, file_path: str
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Get file content and render information (backward compatibility).

        Args:
            bucket: Bucket name
            file_path: Path to file

        Returns:
            Tuple of (raw_content, render_info)
        """
        return self.get_file_content_with_progress(bucket, file_path)

    def prepare_file_for_edit(self, bucket: str, file_path: str) -> str:
        """
        Prepare file content for editing.

        Args:
            bucket: Bucket name
            file_path: Path to file

        Returns:
            String content ready for editing
        """
        content = self.client.read_file(bucket, file_path)
        viewer = self.viewer_registry.get_viewer(file_path)

        if not viewer or not viewer.can_edit():
            raise ValueError(f"File {file_path} is not editable")

        return viewer.prepare_edit(content, file_path)

    def save_edited_file(self, bucket: str, file_path: str, edited_content: str):
        """
        Save edited file content back to storage.

        Args:
            bucket: Bucket name
            file_path: Path to file
            edited_content: Edited content as string
        """
        viewer = self.viewer_registry.get_viewer(file_path)

        if not viewer or not viewer.can_edit():
            raise ValueError(f"File {file_path} is not editable")

        processed_content = viewer.process_edit(edited_content, file_path)
        self.client.write_file(bucket, file_path, processed_content)

        logger.info(f"Saved edited file {bucket}/{file_path}")

    def get_supported_extensions(self) -> Dict[str, str]:
        """Get mapping of supported file extensions to viewer types."""
        extension_map = {}
        for ext, viewer in self.viewer_registry.viewers.items():
            viewer_type = viewer.__class__.__name__.replace("Viewer", "").lower()
            extension_map[ext] = viewer_type
        return extension_map
