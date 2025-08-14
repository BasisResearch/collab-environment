"""
File viewers for different file types in the dashboard.
"""

import io
import pandas as pd
import yaml
import os
import hashlib
import subprocess
import json
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

        # PLY 3D mesh viewer
        ply_viewer = PLYViewer()
        self.viewers[".ply"] = ply_viewer

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

    def _analyze_video_codec(self, temp_path: str) -> Dict[str, Any]:
        """Analyze video codec using ffprobe."""
        try:
            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                temp_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                probe_data = json.loads(result.stdout)

                # Find video stream
                video_stream = None
                for stream in probe_data.get("streams", []):
                    if stream.get("codec_type") == "video":
                        video_stream = stream
                        break

                if video_stream:
                    codec_name = video_stream.get("codec_name", "unknown")
                    profile = video_stream.get("profile", "")

                    # Check browser compatibility
                    browser_compatible = self._is_browser_compatible(
                        codec_name, profile
                    )

                    return {
                        "codec_name": codec_name,
                        "profile": profile,
                        "browser_compatible": browser_compatible,
                        "width": video_stream.get("width"),
                        "height": video_stream.get("height"),
                        "duration": video_stream.get("duration"),
                    }

            return {"codec_name": "unknown", "browser_compatible": False}

        except Exception as e:
            logger.warning(f"Could not analyze video codec: {e}")
            return {"codec_name": "unknown", "browser_compatible": None}

    def _is_browser_compatible(self, codec_name: str, profile: str = "") -> bool:
        """Check if video codec is compatible with modern browsers."""
        # Well-supported codecs across major browsers
        well_supported = {
            "h264": True,  # Most widely supported (AVC)
            "avc": True,  # Same as h264
            "vp8": True,  # WebM - good support
            "vp9": True,  # WebM - good support
            "av1": True,  # Modern codec, growing support
        }

        # Limited support codecs (work in some browsers)
        limited_support = {
            "hevc": "limited",  # Safari, some Chrome versions
            "h265": "limited",  # Same as hevc
            "mpeg4": "limited",  # MPEG-4 Visual - very limited (Firefox 3GP only, ChromeOS)
        }

        # Clearly unsupported codecs
        unsupported = {
            "msmpeg4": False,  # Microsoft MPEG-4
            "flv1": False,  # Flash Video
            "wmv": False,  # Windows Media Video
            "mpeg1video": False,  # MPEG-1
            "mpeg2video": False,  # MPEG-2
        }

        codec_lower = codec_name.lower()

        if codec_lower in unsupported:
            return False
        elif codec_lower in well_supported:
            return True
        elif codec_lower in limited_support:
            return False  # Treat limited support as unsupported for safety
        else:
            # Unknown codec - return False to indicate uncertainty
            return False

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

            # Analyze codec compatibility by writing to a temporary file
            import tempfile

            codec_info = {"codec_name": "unknown", "browser_compatible": None}

            try:
                with tempfile.NamedTemporaryFile(
                    suffix=file_ext, delete=False
                ) as temp_file:
                    temp_file.write(content)
                    temp_path = temp_file.name

                codec_info = self._analyze_video_codec(temp_path)
                os.unlink(temp_path)  # Clean up temp file

            except Exception as e:
                logger.warning(f"Could not analyze video codec for {file_path}: {e}")

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
                "codec_info": codec_info,
            }

        except Exception as e:
            logger.error(f"Error rendering video {file_path}: {e}")
            return {
                "type": "error",
                "message": f"Cannot render video: {e}",
                "size": len(content),
            }


class PLYViewer(BaseViewer):
    """Viewer for PLY 3D mesh files using PyVista."""

    # Track if VTK has been used - only allow one VTK viewer per session
    _vtk_used = False

    def render_view(self, content: bytes, file_path: str) -> Dict[str, Any]:
        """Render PLY file using PyVista and Panel's VTK integration."""
        try:
            import pyvista as pv
            import tempfile
            import os

            # Create a temporary file to save the PLY content
            with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_file:
                tmp_file.write(content)
                temp_path = tmp_file.name

            try:
                # Load the PLY file using PyVista
                mesh = pv.read(temp_path)

                # Check if this is a point cloud or mesh
                is_point_cloud = mesh.n_cells == 0 or mesh.n_cells == mesh.n_points
                has_faces = mesh.n_cells > 0 and not is_point_cloud

                # Create VTK pane that will be inserted into the persistent container
                try:
                    # Create PyVista plotter for Panel VTK
                    plotter = pv.Plotter(
                        window_size=[800, 600],
                        off_screen=False,  # Keep on-screen for Panel VTK
                        notebook=False,  # Disable notebook mode
                    )

                    # Configure plotter appearance
                    plotter.background_color = (0.95, 0.95, 0.95)  # type: ignore

                    if is_point_cloud:
                        # Render as points for point clouds
                        plotter.add_mesh(
                            mesh,  # type: ignore
                            color="lightblue",
                            point_size=3.0,
                            render_points_as_spheres=True,
                            opacity=0.8,
                        )
                    else:
                        # Render as mesh with edges for surfaces
                        plotter.add_mesh(
                            mesh,  # type: ignore
                            color="lightblue",
                            show_edges=True,
                            edge_color="gray",
                            smooth_shading=True,
                            opacity=0.9,
                        )

                    plotter.camera_position = "iso"
                    plotter.reset_camera()  # type: ignore

                    # Store mesh bounds for camera reset in app
                    mesh_bounds = mesh.bounds if hasattr(mesh, "bounds") else None
                    mesh_center = mesh.center if hasattr(mesh, "center") else None

                    # Return the render window instead of creating a VTK pane
                    render_window = plotter.ren_win

                    logger.info(
                        f"🎮 CREATED RENDER WINDOW for reusable VTK pane: {file_path}"
                    )

                except Exception as vtk_error:
                    logger.error(f"VTK render window creation failed: {vtk_error}")
                    render_window = None

                # Get mesh/point cloud statistics
                stats = {
                    "points": mesh.n_points,
                    "cells": mesh.n_cells,
                    "bounds": list(mesh.bounds) if hasattr(mesh, "bounds") else None,
                    "is_point_cloud": is_point_cloud,
                }

                # Only compute surface properties for meshes with faces
                if has_faces:
                    try:
                        stats["area"] = float(mesh.area)
                        stats["volume"] = float(mesh.volume)
                    except Exception as e:
                        logger.debug(f"Could not compute mesh properties: {e}")
                        stats["area"] = None
                        stats["volume"] = None
                else:
                    stats["area"] = None
                    stats["volume"] = None

                return {
                    "type": "ply_3d",
                    "render_window": render_window,
                    "stats": stats,
                    "mesh_bounds": mesh_bounds,
                    "mesh_center": mesh_center,
                    "success": True,
                    "error": None,
                }

            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

        except ImportError as e:
            return {
                "type": "error",
                "error": f"PyVista not available: {e}",
                "success": False,
            }
        except Exception as e:
            return {
                "type": "error",
                "error": f"Failed to render PLY file: {e}",
                "success": False,
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

    def is_file_cached(self, bucket: str, file_path: str) -> bool:
        """Check if a file is cached locally.

        Args:
            bucket: Actual bucket name
            file_path: Full path to file in bucket

        Returns:
            True if file is cached, False otherwise
        """
        cache_path = self._get_cache_path(bucket, file_path)
        return cache_path.exists()

    def get_cache_location(self) -> str:
        """Get the cache directory location."""
        return str(self.cache_dir)

    def get_cache_path(self, bucket: str, file_path: str) -> str:
        """Get the cache file path for a bucket/file combination (public method)."""
        return str(self._get_cache_path(bucket, file_path))

    def convert_video_to_h264(
        self, bucket: str, file_path: str
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Convert a video file to H.264 format.

        Args:
            bucket: Bucket name
            file_path: Path to file

        Returns:
            Tuple of (success, message, converted_file_path)
        """
        try:
            # Get the cached file path
            cache_path = self._get_cache_path(bucket, file_path)
            if not cache_path.exists():
                return False, "File not cached locally", None

            # Create output path for converted file
            input_path = Path(cache_path)
            output_path = input_path.parent / f"{input_path.stem}_h264.mp4"

            # Check if conversion already exists
            if output_path.exists():
                return True, "H.264 version already exists", str(output_path)

            # Run ffmpeg conversion
            cmd = [
                "ffmpeg",
                "-i",
                str(cache_path),
                "-c:v",
                "libx264",  # H.264 video codec
                "-crf",
                "23",  # Constant rate factor (good quality)
                "-preset",
                "medium",  # Encoding speed vs compression
                "-c:a",
                "aac",  # AAC audio codec
                "-movflags",
                "+faststart",  # Web optimization
                "-y",  # Overwrite output file
                str(output_path),
            ]

            logger.info(f"Converting video: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                logger.info(f"Successfully converted video to H.264: {output_path}")
                return (
                    True,
                    f"Successfully converted to H.264 ({self._format_file_size(output_path.stat().st_size)})",
                    str(output_path),
                )
            else:
                error_msg = (
                    result.stderr[-500:] if result.stderr else "Unknown ffmpeg error"
                )
                logger.error(f"Video conversion failed: {error_msg}")
                return False, f"Conversion failed: {error_msg}", None

        except subprocess.TimeoutExpired:
            return False, "Conversion timed out (>5 minutes)", None
        except Exception as e:
            logger.error(f"Error converting video: {e}")
            return False, f"Conversion error: {e}", None

    def replace_file_from_cache(self, bucket: str, file_path: str) -> Tuple[bool, str]:
        """
        Replace a file in the cloud with the cached version, backing up the original.

        Args:
            bucket: Bucket name
            file_path: Path to file in bucket

        Returns:
            Tuple of (success, message)
        """
        try:
            # Get cached file path
            cache_path = self._get_cache_path(bucket, file_path)
            if not cache_path.exists():
                return False, "File not cached locally"

            # Create backup filename with _old suffix
            original_path = Path(file_path)
            backup_filename = f"{original_path.stem}_old{original_path.suffix}"
            backup_file_path = str(original_path.parent / backup_filename)

            # Step 1: Backup original file using rclone copyto
            try:
                backup_success = self.client.copy_file(
                    bucket, file_path, bucket, backup_file_path
                )
                if backup_success:
                    logger.info(f"Backed up original file: {bucket}/{backup_file_path}")
                else:
                    logger.warning("Could not backup original file using rclone copyto")
            except Exception as e:
                logger.warning(f"Could not backup original file: {e}")
                # Continue anyway - backup failure shouldn't stop the upload

            # Step 2: Upload cached file directly using rclone copyto
            success = self.client.copy_local_to_remote(
                str(cache_path), bucket, file_path
            )
            if not success:
                return False, "Failed to upload cached file to remote"

            logger.info(
                f"Replaced cloud file with cached version: {bucket}/{file_path}"
            )
            return (
                True,
                f"Successfully replaced cloud file with cached version (backup saved as {backup_filename})",
            )

        except Exception as e:
            logger.error(f"Error replacing file from cache: {e}")
            return False, f"Replace failed: {e}"

    def delete_file(self, bucket: str, file_path: str) -> Tuple[bool, str]:
        """
        Delete a file from the cloud storage.

        Args:
            bucket: Bucket name
            file_path: Path to file in bucket

        Returns:
            Tuple of (success, message)
        """
        try:
            # Use rclone to delete the file
            cmd = [
                "rclone",
                "delete",
                f"{self.client.remote_name}:{bucket}/{file_path}",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                logger.info(f"Deleted file: {bucket}/{file_path}")

                # Also remove from local cache
                cache_path = self._get_cache_path(bucket, file_path)
                if cache_path.exists():
                    cache_path.unlink()
                    logger.info(f"Removed from cache: {cache_path}")

                return True, f"Successfully deleted {file_path}"
            else:
                error_msg = result.stderr or "Unknown rclone error"
                return False, f"Delete failed: {error_msg}"

        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return False, f"Delete failed: {e}"

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

    def _get_local_file_checksum(self, file_path: Path, hash_type: str = "md5") -> str:
        """Get checksum of a local file."""
        try:
            import hashlib

            hash_func = getattr(hashlib, hash_type)()

            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)

            checksum = hash_func.hexdigest()
            logger.debug(f"Local {hash_type} checksum for {file_path}: {checksum}")
            return checksum

        except Exception as e:
            logger.error(f"Error calculating local checksum: {e}")
            return ""

    def is_file_modified(self, bucket: str, file_path: str) -> bool:
        """
        Check if local cached file differs from remote file using checksums.

        Args:
            bucket: Bucket name
            file_path: Path to file

        Returns:
            True if local file differs from remote, False if same or no cache
        """
        cache_path = self._get_cache_path(bucket, file_path)
        if not cache_path.exists():
            return False  # No local cache, so not modified

        try:
            # Get local checksum
            local_checksum = self._get_local_file_checksum(cache_path)
            if not local_checksum:
                return False

            # Get remote checksum
            remote_checksum = self.client.get_file_checksum(bucket, file_path)
            if not remote_checksum:
                return False  # Can't compare, assume not modified

            is_different = local_checksum != remote_checksum
            if is_different:
                logger.info(
                    f"File modified: {bucket}/{file_path} (local: {local_checksum[:8]}..., remote: {remote_checksum[:8]}...)"
                )
            else:
                logger.debug(f"File unchanged: {bucket}/{file_path}")

            return is_different

        except Exception as e:
            logger.error(f"Error checking if file is modified: {e}")
            return False

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
        Prepare file content for editing from cached version.

        Args:
            bucket: Bucket name
            file_path: Path to file

        Returns:
            String content ready for editing
        """
        # Read from cache if available, otherwise from remote
        cache_path = self._get_cache_path(bucket, file_path)
        if cache_path.exists():
            content = cache_path.read_bytes()
        else:
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
