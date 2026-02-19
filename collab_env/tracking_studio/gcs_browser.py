"""
GCS Video Browser Component

Provides interface for browsing and downloading videos from Google Cloud Storage.
"""

from typing import List, Dict
from loguru import logger

from collab_env.data.gcs_utils import GCSClient


class GCSVideoBrowser:
    """Browser for selecting and downloading videos from GCS buckets"""

    def __init__(self, credentials_path: str):
        """
        Initialize GCS browser.

        Args:
            credentials_path: Path to GCS service account credentials JSON
        """
        self.gcs = GCSClient(credentials_path=credentials_path)
        logger.info("GCS Video Browser initialized")

    def list_buckets(self) -> List[str]:
        """
        List all available GCS buckets.

        Returns:
            List of bucket names
        """
        try:
            buckets = self.gcs.list_buckets()
            logger.info(f"Found {len(buckets)} buckets")
            return buckets
        except Exception as e:
            logger.error(f"Failed to list buckets: {e}")
            return []

    def list_folders(self, bucket: str, prefix: str = "") -> List[str]:
        """
        List immediate subfolders in a bucket path.

        Note: GCS doesn't have real folders - they're just prefixes in object names.
        This function extracts unique first-level directory prefixes.

        Args:
            bucket: GCS bucket name
            prefix: Path prefix within bucket (should end with / if not empty)

        Returns:
            List of folder names (relative to prefix)
        """
        try:
            # Ensure prefix ends with / if not empty
            if prefix and not prefix.endswith("/"):
                prefix = prefix + "/"

            # Get all objects recursively to find folder-like structures
            pattern = f"{bucket}/{prefix}**" if prefix else f"{bucket}/**"
            all_paths = self.gcs.glob(pattern)

            # Extract unique immediate subdirectories
            unique_folders = set()
            for path in all_paths:
                # Remove bucket prefix
                rel_path = path.replace(f"{bucket}/", "")

                # Remove the current prefix if any
                if prefix:
                    if not rel_path.startswith(prefix):
                        continue
                    rel_path = rel_path[len(prefix) :]

                # Get first directory component after prefix
                if "/" in rel_path:
                    folder = rel_path.split("/")[0]
                    if folder:  # Skip empty strings
                        unique_folders.add(folder)

            folder_list = sorted(list(unique_folders))
            logger.info(
                f"Found {len(folder_list)} folder prefixes in {bucket}/{prefix}"
            )
            return folder_list

        except Exception as e:
            logger.error(f"Failed to list folders in {bucket}/{prefix}: {e}")
            return []

    def list_videos(self, bucket: str, prefix: str = "") -> List[Dict[str, str]]:
        """
        List video files (.mp4, .mov, .avi) in a bucket path.

        Args:
            bucket: GCS bucket name
            prefix: Path prefix within bucket

        Returns:
            List of dicts with video metadata: {name, path, rel_path}
        """
        try:
            # Build pattern for video files - ensure prefix ends with / if not empty
            if prefix and not prefix.endswith("/"):
                prefix = prefix + "/"

            # Search for multiple video formats
            video_extensions = ["*.mp4", "*.mov", "*.avi", "*.MP4", "*.MOV", "*.AVI"]
            all_files = []

            for ext in video_extensions:
                pattern = (
                    f"{bucket}/{prefix}**/{ext}" if prefix else f"{bucket}/**/{ext}"
                )
                files = self.gcs.glob(pattern)
                all_files.extend(files)

            videos = []
            seen_paths = set()  # Avoid duplicates from case-insensitive extensions

            for file_path in all_files:
                if file_path in seen_paths:
                    continue
                seen_paths.add(file_path)

                # Extract filename
                filename = file_path.split("/")[-1]

                # Get relative path from bucket
                rel_path = file_path.replace(f"{bucket}/", "")

                videos.append(
                    {
                        "name": filename,
                        "path": file_path,
                        "rel_path": rel_path,
                    }
                )

            logger.info(f"Found {len(videos)} videos in {bucket}/{prefix}")
            return sorted(videos, key=lambda x: x["name"])

        except Exception as e:
            logger.error(f"Failed to list videos in {bucket}/{prefix}: {e}")
            return []

    def download_video(self, gcs_path: str, local_path: str) -> str:
        """
        Download video from GCS to local path.

        Args:
            gcs_path: Full GCS path (e.g., "bucket/path/video.mp4" or "gs://bucket/path/video.mp4")
            local_path: Local destination path

        Returns:
            Local path to downloaded video
        """
        try:
            # Remove gs:// prefix if present
            if gcs_path.startswith("gs://"):
                gcs_path = gcs_path[5:]

            logger.info(f"Downloading {gcs_path} to {local_path}")
            self.gcs.download_file(gcs_path, local_path, overwrite=True)
            logger.info(f"Successfully downloaded video to {local_path}")
            return local_path

        except Exception as e:
            logger.error(f"Failed to download video from {gcs_path}: {e}")
            raise
