"""
File viewers for different file types in the dashboard.
"""

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

    def __init__(self, rclone_client: Optional[RcloneClient] = None):
        self.viewers = {}
        self.rclone_client = rclone_client
        self._register_default_viewers()

    def _register_default_viewers(self):
        """Register default file viewers."""
        # Text viewers - disabled for now

        # text_viewer = TextViewer()
        # for ext in [".yml", ".yaml", ".txt", ".xml", ".json", ".md", ".rst"]:
        #     self.viewers[ext] = text_viewer

        # Table viewers
        table_viewer = TableViewer()
        for ext in [".csv", ".parquet"]:
            self.viewers[ext] = table_viewer

        # Video viewer - pass rclone_client
        video_viewer = VideoViewer(rclone_client=self.rclone_client)
        for ext in [".mp4", ".avi", ".mov", ".mkv"]:
            self.viewers[ext] = video_viewer

        # PLY 3D mesh viewer - pass rclone_client
        ply_viewer = PLYViewer(rclone_client=self.rclone_client)
        self.viewers[".ply"] = ply_viewer

        # PKL pickle file viewer
        pkl_viewer = PKLViewer()
        for ext in [".pkl", ".pickle"]:
            self.viewers[ext] = pkl_viewer

    def get_viewer(self, file_path: str) -> Optional["BaseViewer"]:
        """Get viewer for a file based on its extension and file-specific criteria."""
        ext = Path(file_path).suffix.lower()
        viewer = self.viewers.get(ext)

        # If viewer has a can_handle_file method, check if it can handle this specific file
        if viewer and hasattr(viewer, "can_handle_file"):
            if not viewer.can_handle_file(file_path):
                return None

        return viewer

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

    def render_view(
        self, local_file_path: str, original_file_path: str
    ) -> Dict[str, Any]:
        """
        Render file content for viewing.

        Args:
            local_file_path: Path to the local cached file
            original_file_path: Original path to the file (for reference)

        Returns:
            Dictionary with rendering information
        """
        raise NotImplementedError

    def prepare_edit(self, local_file_path: str, original_file_path: str) -> str:
        """
        Prepare content for editing.

        Args:
            local_file_path: Path to the local cached file
            original_file_path: Original path to the file (for reference)

        Returns:
            String content ready for editing
        """
        if not self.can_edit():
            raise NotImplementedError("This viewer does not support editing")
        return Path(local_file_path).read_text("utf-8")

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

    def render_view(
        self, local_file_path: str, original_file_path: str
    ) -> Dict[str, Any]:
        """Render text content with syntax highlighting hints."""
        try:
            text_content = Path(local_file_path).read_text("utf-8")
            file_ext = Path(original_file_path).suffix.lower()

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
                    logger.warning(f"YAML parsing error in {original_file_path}: {e}")

            return {
                "type": "text",
                "content": text_content,
                "language": language,
                "parsed_data": parsed_data,
                "size": Path(local_file_path).stat().st_size,
                "lines": text_content.count("\n") + 1,
            }

        except UnicodeDecodeError as e:
            return {
                "type": "error",
                "message": f"Cannot decode file as text: {e}",
                "size": Path(local_file_path).stat().st_size,
            }


class TableViewer(BaseViewer):
    """Viewer for tabular data files (CSV, Parquet)."""

    def render_view(
        self, local_file_path: str, original_file_path: str
    ) -> Dict[str, Any]:
        """Render tabular data as HTML table."""
        try:
            file_ext = Path(original_file_path).suffix.lower()

            # Read data based on file type
            if file_ext == ".csv":
                df = pd.read_csv(local_file_path)
            elif file_ext == ".parquet":
                df = pd.read_parquet(local_file_path)
            else:
                raise ValueError(f"Unsupported table format: {file_ext}")

            # Basic statistics
            stats = {
                "rows": len(df),
                "columns": len(df.columns),
                "size": Path(local_file_path).stat().st_size,
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
            logger.error(f"Error rendering table {original_file_path}: {e}")
            return {
                "type": "error",
                "message": f"Cannot render table: {e}",
                "size": Path(local_file_path).stat().st_size,
            }


class VideoViewer(BaseViewer):
    """Viewer for video files."""

    def __init__(self, rclone_client: Optional[RcloneClient] = None):
        self._current_bucket = None
        self._rclone_client = rclone_client

    def _find_bbox_csv_files_with_bucket(self, bucket: str, video_path: str) -> list:
        """Find matching bbox CSV files in the same directory as the video, and for thermal videos, also check corresponding rgb folders."""
        try:
            import re

            # Get directory path from video_path (without bucket)
            video_path_obj = Path(video_path)
            dir_path = (
                str(video_path_obj.parent) if video_path_obj.parent != Path(".") else ""
            )

            # Use the provided rclone client, or create a default one if not available
            rclone_client = self._rclone_client or RcloneClient()

            bbox_csvs = []
            directories_to_search = [dir_path]  # Always search the same directory first

            # For thermal videos, also search corresponding rgb directories
            # Check if this is a thermal_XX directory pattern
            if dir_path:
                dir_name = Path(dir_path).name
                thermal_match = re.match(r"^thermal_(\d+)$", dir_name, re.IGNORECASE)
                if thermal_match:
                    camera_number = thermal_match.group(1)
                    corresponding_rgb_dir = str(
                        Path(dir_path).parent / f"rgb_{camera_number}"
                    )
                    directories_to_search.append(corresponding_rgb_dir)
                    logger.info(
                        f"Thermal video detected: {dir_path}, will also search {corresponding_rgb_dir} for bbox CSV files"
                    )

            # Search all directories
            for search_dir in directories_to_search:
                try:
                    # List files in the directory
                    files = rclone_client.list_directory(bucket, search_dir)

                    # Filter for CSV files ending with '_bboxes.csv'
                    for file_info in files:
                        file_name = file_info.get(
                            "Name", ""
                        )  # rclone uses 'Name' field

                        # Only look for CSV files that contain "bbox" or "bboxes" in the filename
                        if file_name.lower().endswith(".csv") and (
                            "bbox" in file_name.lower()
                        ):
                            # Construct full path for validation (without bucket prefix)
                            full_csv_path = (
                                str(Path(search_dir) / file_name)
                                if search_dir
                                else file_name
                            )

                            # Check if it's actually a bbox CSV by examining columns
                            if self._is_bbox_csv_with_bucket(bucket, full_csv_path):
                                # Add source information to help identify cross-folder CSVs
                                csv_entry = {
                                    "name": file_name,
                                    "path": full_csv_path,  # Path without bucket prefix
                                    "size": file_info.get("Size", 0),
                                    "source_dir": search_dir,  # Track which directory this came from
                                    "is_cross_folder": search_dir
                                    != dir_path,  # Flag for cross-folder detection
                                }

                                # Add cross-folder indicator to name for clarity
                                if search_dir != dir_path:
                                    csv_entry["display_name"] = (
                                        f"{file_name} (from {Path(search_dir).name})"
                                    )
                                else:
                                    csv_entry["display_name"] = file_name

                                bbox_csvs.append(csv_entry)

                except Exception as e:
                    if search_dir == dir_path:
                        logger.warning(
                            f"Could not list primary directory {search_dir} for bbox detection: {e}"
                        )
                    else:
                        logger.info(
                            f"Could not list cross-folder directory {search_dir} for bbox detection (this is normal if folder doesn't exist): {e}"
                        )

            logger.info(
                f"Found {len(bbox_csvs)} bbox CSV files for video {bucket}/{video_path} (searched directories: {directories_to_search})"
            )
            return bbox_csvs

        except Exception as e:
            logger.error(f"Error finding bbox CSV files: {e}")
            return []

    def _is_bbox_csv_with_bucket(self, bucket: str, csv_path: str) -> bool:
        """Check if a CSV file contains bbox/tracking data."""
        try:
            import pandas as pd
            import tempfile
            import os

            rclone_client = self._rclone_client or RcloneClient()

            # Download a small sample of the CSV to check columns
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
                temp_path = temp_file.name

            try:
                # Download the file using read_file method
                csv_content = rclone_client.read_file(bucket, csv_path)

                # Write to temp file
                with open(temp_path, "wb") as f:
                    f.write(csv_content)

                # Read just the first few rows to check columns
                df = pd.read_csv(temp_path, nrows=5)

                # Check for required columns for bbox data - must have x1,y1,x2,y2 format
                required_cols = {"track_id", "frame"}
                bbox_cols = {"x1", "y1", "x2", "y2"}

                df_cols = set(df.columns)

                # Must have track_id, frame, and x1,y1,x2,y2 bbox columns
                has_required = required_cols.issubset(df_cols)
                has_bbox = bbox_cols.issubset(df_cols)

                is_valid = has_required and has_bbox

                if is_valid:
                    logger.info(
                        f"Valid bbox CSV detected: {bucket}/{csv_path} (bbox format with x1,y1,x2,y2)"
                    )

                return is_valid

            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            logger.warning(f"Could not validate CSV file {bucket}/{csv_path}: {e}")
            return False

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

    def render_view(
        self, local_file_path: str, original_file_path: str
    ) -> Dict[str, Any]:
        """Render video using HTML5 video element."""
        try:
            file_ext = Path(original_file_path).suffix.lower()

            # Determine MIME type
            mime_types = {
                ".mp4": "video/mp4",
                ".avi": "video/x-msvideo",
                ".mov": "video/quicktime",
                ".mkv": "video/x-matroska",
            }

            mime_type = mime_types.get(file_ext, "video/mp4")

            # Analyze codec compatibility using the local cached file directly
            codec_info = {"codec_name": "unknown", "browser_compatible": None}

            try:
                codec_info = self._analyze_video_codec(local_file_path)
            except Exception as e:
                logger.warning(
                    f"Could not analyze video codec for {original_file_path}: {e}"
                )

            # Read and encode video content as base64 for embedding
            # Note: This is not ideal for large videos - consider using rclone serve http
            content = Path(local_file_path).read_bytes()
            video_b64 = base64.b64encode(content).decode("utf-8")
            data_url = f"data:{mime_type};base64,{video_b64}"

            file_size = Path(local_file_path).stat().st_size

            # Check for matching bbox CSV files in the same directory
            bbox_csvs: list[
                dict
            ] = []  # Will be set after render_info is created with bucket info

            return {
                "type": "video",
                "data_url": data_url,
                "mime_type": mime_type,
                "size": file_size,
                "size_mb": file_size / (1024 * 1024),
                "codec_info": codec_info,
                "bbox_csvs": bbox_csvs,  # List of available bbox CSV files
            }

        except Exception as e:
            logger.error(f"Error rendering video {original_file_path}: {e}")
            return {
                "type": "error",
                "message": f"Cannot render video: {e}",
                "size": Path(local_file_path).stat().st_size,
            }


class PKLViewer(BaseViewer):
    """Viewer for camera parameter pickle files only (filtered by name)."""

    def get_supported_extensions(self) -> list[str]:
        return [".pkl", ".pickle"]

    def can_handle_file(self, file_path: str) -> bool:
        """Only handle camera parameter files based on naming pattern."""
        filename = Path(file_path).name.lower()

        # Specific patterns for camera parameter files we've seen:
        # - *_mesh-aligned.pkl (most common)
        # - *camera*.pkl
        # - *intrinsic*.pkl, *extrinsic*.pkl
        camera_patterns = [
            "mesh-aligned",  # rgb_1_mesh-aligned.pkl
            "camera",  # camera_params.pkl, camera_intrinsics.pkl
            "intrinsic",  # intrinsics.pkl
            "extrinsic",  # extrinsics.pkl
            "calibration",  # calibration.pkl
        ]

        return any(pattern in filename for pattern in camera_patterns)

    def render_view(
        self, local_file_path: str, original_file_path: str
    ) -> Dict[str, Any]:
        """Render pickle file content by displaying its structure."""
        import pickle
        import json

        try:
            # Load pickle data from local file
            with open(local_file_path, "rb") as f:
                data = pickle.load(f)

            # Create a display of the pickle structure
            def describe_object(obj, max_depth=3, current_depth=0):
                if current_depth > max_depth:
                    return "..."

                if obj is None:
                    return "None"
                elif isinstance(obj, (int, float, str, bool)):
                    return f"{type(obj).__name__}: {repr(obj)}"
                elif isinstance(obj, dict):
                    if not obj:
                        return "dict: {}"
                    items = {}
                    for k, v in list(obj.items())[:10]:  # Show first 10 items
                        items[str(k)] = describe_object(v, max_depth, current_depth + 1)
                    if len(obj) > 10:
                        items["..."] = f"({len(obj) - 10} more items)"
                    return items
                elif isinstance(obj, (list, tuple)):
                    if not obj:
                        return f"{type(obj).__name__}: []"
                    items = [
                        describe_object(item, max_depth, current_depth + 1)
                        for item in obj[:5]
                    ]  # Show first 5 items
                    if len(obj) > 5:
                        items.append(f"... ({len(obj) - 5} more items)")
                    return {f"{type(obj).__name__}[{len(obj)}]": items}
                elif hasattr(obj, "shape"):  # NumPy arrays, tensors
                    return f"{type(obj).__name__}: shape {getattr(obj, 'shape', '?')}, dtype {getattr(obj, 'dtype', '?')}"
                else:
                    return f"{type(obj).__name__}: {str(obj)[:100]}..."

            structure = describe_object(data)

            # Convert to JSON for display
            json_str = json.dumps(structure, indent=2, default=str)

            file_size = Path(local_file_path).stat().st_size

            # Create display content
            display_content = f"""# Pickle File Structure

**File type:** Python pickle (.pkl)  
**Root type:** {type(data).__name__}  
**File size:** {file_size:,} bytes  

## Content Structure:
```json
{json_str}
```

## Raw Type Information:
- **Type**: `{type(data)}`
- **Size**: {file_size:,} bytes
"""

            # If it looks like camera parameters, add specific info
            if isinstance(data, dict) and any(
                key in data for key in ["K", "c2w", "fx", "fy"]
            ):
                display_content += "\n## Camera Parameters Detected\n"
                display_content += "This appears to be a camera parameter file with intrinsic/extrinsic data.\n"

                if "K" in data:
                    display_content += f"- **Intrinsic Matrix (K)**: {getattr(data['K'], 'shape', 'unknown shape')}\n"
                if "c2w" in data:
                    display_content += f"- **Camera-to-World (c2w)**: {getattr(data['c2w'], 'shape', 'unknown shape')}\n"
                if "width" in data and "height" in data:
                    display_content += f"- **Image Size**: {data.get('width', '?')} x {data.get('height', '?')}\n"

            return {
                "type": "text",
                "content": display_content,
                "language": "markdown",
                "parsed_data": data,
                "size": file_size,
                "lines": display_content.count("\n") + 1,
                "root_type": type(data).__name__,
            }

        except Exception as e:
            file_size = Path(local_file_path).stat().st_size
            error_content = f"""# Pickle File Load Error

**Error**: {str(e)}  
**File size**: {file_size:,} bytes  

This pickle file could not be loaded. It may be corrupted or created with a different Python version.

```
{str(e)}
```
"""
            return {
                "type": "error",
                "content": error_content,
                "language": "markdown",
                "error": str(e),
                "size": file_size,
                "lines": error_content.count("\n") + 1,
            }


class PLYViewer(BaseViewer):
    """Viewer for PLY 3D mesh files using PyVista."""

    # Track if VTK has been used - only allow one VTK viewer per session
    _vtk_used = False

    def __init__(self, rclone_client: Optional[RcloneClient] = None):
        self._rclone_client = rclone_client

    def _find_3d_csv_files_with_bucket(self, bucket: str, ply_path: str) -> list:
        """Find matching 3D centroids CSV files by searching the entire session folder."""
        try:
            from pathlib import Path

            # Extract session folder from PLY path
            # Example: "2024_02_06-session_0001/environment/C0043/rade-features/mesh/mesh.ply"
            # We want to search in: "2024_02_06-session_0001/"
            ply_path_obj = Path(ply_path)
            path_parts = ply_path_obj.parts

            if len(path_parts) == 0:
                logger.warning(f"Invalid PLY path: {ply_path}")
                return []

            # Session folder is typically the first part of the path
            session_folder = path_parts[0]

            # Use the provided rclone client, or create a default one if not available
            rclone_client = self._rclone_client or RcloneClient()

            csv_3d_files = []

            def search_directory_recursive(search_path: str):
                """Recursively search for 3D CSV files in a directory."""
                try:
                    files = rclone_client.list_directory(bucket, search_path)

                    for file_info in files:
                        file_name = file_info.get("Name", "")
                        is_dir = file_info.get("IsDir", False)

                        if is_dir:
                            # Recursively search subdirectories
                            subdir_path = (
                                f"{search_path}/{file_name}"
                                if search_path
                                else file_name
                            )
                            search_directory_recursive(subdir_path)
                        elif file_name.endswith("_centroids_3d.csv"):
                            # Found a 3D CSV file, store with full relative path
                            full_path = (
                                f"{search_path}/{file_name}"
                                if search_path
                                else file_name
                            )

                            # Extract camera info for better identification
                            camera_info = "unknown"
                            if "rgb_1" in full_path:
                                camera_info = "rgb_1"
                            elif "rgb_2" in full_path:
                                camera_info = "rgb_2"
                            elif "thermal_1" in full_path:
                                camera_info = "thermal_1"
                            elif "thermal_2" in full_path:
                                camera_info = "thermal_2"

                            csv_3d_files.append(
                                {
                                    "path": full_path,
                                    "name": file_name,
                                    "camera": camera_info,
                                    "display_name": f"{file_name} ({camera_info})",
                                }
                            )

                except Exception as e:
                    logger.debug(f"Could not search directory {search_path}: {e}")

            # Start recursive search from session folder
            logger.info(
                f"Searching for 3D CSV files in session folder: {session_folder}"
            )
            search_directory_recursive(session_folder)

            logger.info(
                f"Found {len(csv_3d_files)} 3D CSV files: {[f['display_name'] for f in csv_3d_files]}"
            )
            return csv_3d_files

        except Exception as e:
            logger.error(f"Error finding 3D CSV files: {e}")
            return []

    def _find_camera_params_with_bucket(
        self, bucket: str, ply_path: str
    ) -> Optional[str]:
        """Find matching camera parameters pickle file in aligned_splat/[CAMERA_ID]/ subfolders."""
        try:
            from .rclone_client import RcloneClient
            from pathlib import Path

            # Extract session folder from PLY path
            # Example: "2024_02_06-session_0001/environment/C0043/rade-features/mesh/mesh.ply"
            # We want to search in: "2024_02_06-session_0001/aligned_splat/[CAMERA_ID]/"
            ply_path_obj = Path(ply_path)
            path_parts = ply_path_obj.parts

            if len(path_parts) == 0:
                logger.warning(f"Invalid PLY path for camera params search: {ply_path}")
                return None

            # Session folder is typically the first part of the path
            session_folder = path_parts[0]
            aligned_splat_folder = f"{session_folder}/aligned_splat"

            # Use the provided rclone client, or create a default one if not available
            rclone_client = self._rclone_client or RcloneClient()

            logger.info(
                f"Searching for camera params in subfolders of: {aligned_splat_folder}"
            )

            try:
                # First, list the aligned_splat directory to find camera subfolders
                aligned_splat_contents = rclone_client.list_directory(
                    bucket, aligned_splat_folder
                )

                camera_folders = []
                for item in aligned_splat_contents:
                    if item.get("IsDir", False):  # Only directories
                        camera_id = item.get("Name", "")
                        if camera_id:  # Valid camera folder name
                            camera_folders.append(camera_id)

                logger.info(
                    f"Found camera folders in {aligned_splat_folder}: {camera_folders}"
                )

                # Search each camera subfolder for _mesh-aligned.pkl files
                for camera_id in camera_folders:
                    camera_folder = f"{aligned_splat_folder}/{camera_id}"

                    try:
                        camera_files = rclone_client.list_directory(
                            bucket, camera_folder
                        )

                        # Log files in this camera folder
                        camera_file_names = [f.get("Name", "") for f in camera_files]
                        logger.info(f"Files in {camera_folder}: {camera_file_names}")

                        for file_info in camera_files:
                            file_name = file_info.get("Name", "")
                            if file_name.endswith(
                                "_mesh-aligned.pkl"
                            ) or file_name.endswith("_mesh_aligned.pkl"):
                                full_path = f"{camera_folder}/{file_name}"
                                logger.info(f"Found camera params file: {full_path}")
                                return full_path

                    except Exception as e:
                        logger.debug(
                            f"Could not access camera folder {camera_folder}: {e}"
                        )
                        continue

                logger.info(
                    f"No camera params files found in any camera subfolder of {aligned_splat_folder}"
                )
                return None

            except Exception as e:
                logger.info(f"Could not access {aligned_splat_folder}: {e}")
                return None

        except Exception as e:
            logger.error(f"Error finding camera params: {e}")
            return None

    def render_view(
        self, local_file_path: str, original_file_path: str
    ) -> Dict[str, Any]:
        """Render PLY file using PyVista and Panel's VTK integration."""
        try:
            import pyvista as pv

            # Load the PLY file using PyVista directly from the cached file
            mesh = pv.read(local_file_path)

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
                        # color="lightblue",
                        # point_size=3.0,
                        # render_points_as_spheres=True,
                        # opacity=0.8,
                    )
                else:
                    # Render as mesh with edges for surfaces
                    plotter.add_mesh(
                        mesh,  # type: ignore
                        # color="lightblue",
                        # show_edges=True,
                        # edge_color="gray",
                        # smooth_shading=True,
                        # opacity=0.9,
                    )

                # Set camera position with better default view
                # Use explicit camera positioning instead of "iso" to avoid issues
                if hasattr(mesh, "bounds") and hasattr(mesh, "center"):
                    bounds = mesh.bounds
                    center = mesh.center

                    # Calculate a good viewing distance based on mesh size
                    x_range = bounds[1] - bounds[0]
                    y_range = bounds[3] - bounds[2]
                    z_range = bounds[5] - bounds[4]
                    max_range = max(x_range, y_range, z_range)
                    distance = max_range * 2.5

                    # Set camera to a nice isometric view
                    plotter.camera.position = (
                        center[0] + distance,
                        center[1] + distance,
                        center[2] + distance,
                    )
                    plotter.camera.focal_point = center
                    plotter.camera.up = (0, 0, 1)  # Z-up
                else:
                    # Fallback to standard iso view
                    plotter.camera_position = "iso"

                plotter.reset_camera()  # type: ignore
                plotter.reset_camera_clipping_range()  # Ensure proper clipping

                # Store mesh bounds for camera reset in app
                mesh_bounds = mesh.bounds if hasattr(mesh, "bounds") else None
                mesh_center = mesh.center if hasattr(mesh, "center") else None

                # Return the render window instead of creating a VTK pane
                render_window = plotter.ren_win

                logger.info(
                    f"🎮 CREATED RENDER WINDOW for reusable VTK pane: {original_file_path}"
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
        self.viewer_registry = FileViewerRegistry(rclone_client=rclone_client)

        # Setup cache directory
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/collab_env_dashboard")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Dashboard cache directory: {self.cache_dir}")

    def _get_cache_path(self, bucket: str, file_path: str) -> Path:
        """Get cache file path for a bucket/file combination."""
        # Create a safe filename from bucket and file path
        hash_input = f"{bucket}/{file_path}"
        safe_name = hashlib.md5(hash_input.encode()).hexdigest()
        file_ext = Path(file_path).suffix
        cache_filename = f"{safe_name}{file_ext}"
        cache_path = self.cache_dir / cache_filename

        # Debug logging for PKL files
        if file_ext.lower() == ".pkl":
            logger.info("🔍 PKL cache path generation:")
            logger.info(f"  📂 Bucket: {bucket}")
            logger.info(f"  📄 File path: {file_path}")
            logger.info(f"  🔐 Hash input: {hash_input}")
            logger.info(f"  🏷️ Hash: {safe_name}")
            logger.info(f"  💾 Cache path: {cache_path}")

        return cache_path

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
                # Pass local cache path to viewer instead of content
                render_info = viewer.render_view(str(cache_path), file_path)
                render_info["viewer_available"] = True
                render_info["can_edit"] = viewer.can_edit()
                render_info["from_cache"] = from_cache
                # Add bucket info for viewers that need it (like video bbox detection)
                render_info["bucket"] = bucket
                render_info["full_path"] = file_path

                # For video files, check for bbox CSV files now that we have bucket info
                if render_info.get("type") == "video" and hasattr(
                    viewer, "_find_bbox_csv_files_with_bucket"
                ):
                    try:
                        logger.info(
                            f"Checking for bbox CSV files in {bucket}/{file_path}"
                        )
                        bbox_csvs = viewer._find_bbox_csv_files_with_bucket(
                            bucket, file_path
                        )
                        render_info["bbox_csvs"] = bbox_csvs
                        logger.info(
                            f"Bbox CSV detection complete: found {len(bbox_csvs)} files"
                        )
                    except Exception as e:
                        logger.warning(f"Error detecting bbox CSV files: {e}")
                        render_info["bbox_csvs"] = []

                # For PLY files, check for 3D CSV files now that we have bucket info
                if render_info.get("type") == "ply_3d" and hasattr(
                    viewer, "_find_3d_csv_files_with_bucket"
                ):
                    try:
                        logger.info(
                            f"Checking for 3D CSV files in {bucket}/{file_path}"
                        )
                        csv_3d_files = viewer._find_3d_csv_files_with_bucket(
                            bucket, file_path
                        )
                        render_info["csv_3d_files"] = csv_3d_files

                        # Also check for camera params
                        if hasattr(viewer, "_find_camera_params_with_bucket"):
                            camera_params_file = viewer._find_camera_params_with_bucket(
                                bucket, file_path
                            )
                            render_info["camera_params_file"] = camera_params_file

                        logger.info(
                            f"3D CSV detection complete: found {len(csv_3d_files)} files"
                        )
                    except Exception as e:
                        logger.warning(f"Error detecting 3D CSV files: {e}")
                        render_info["csv_3d_files"] = []
                        render_info["camera_params_file"] = None
            else:
                # Unknown file type
                file_size = cache_path.stat().st_size
                render_info = {
                    "type": "unknown",
                    "size": file_size,
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
        # Ensure file is cached
        cache_path = self._get_cache_path(bucket, file_path)
        if not cache_path.exists():
            # Download and cache if not available
            content = self.client.read_file(bucket, file_path)
            cache_path.write_bytes(content)

        viewer = self.viewer_registry.get_viewer(file_path)

        if not viewer or not viewer.can_edit():
            raise ValueError(f"File {file_path} is not editable")

        # Pass cache path to viewer instead of content
        return viewer.prepare_edit(str(cache_path), file_path)

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
