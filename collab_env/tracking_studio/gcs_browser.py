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

            # Use a single delimited listing of immediate children. This returns
            # only the first-level entries (folders show up as "directory"),
            # instead of recursively walking the whole bucket and extracting
            # prefixes client-side (which is O(all objects) and very slow on
            # large buckets).
            path = f"{bucket}/{prefix}".rstrip("/")
            entries = self.gcs.gcs.ls(path, detail=True)

            unique_folders = set()
            for entry in entries:
                if entry.get("type") != "directory":
                    continue
                folder = entry["name"].rstrip("/").split("/")[-1]
                if folder:  # Skip empty strings
                    unique_folders.add(folder)

            folder_list = sorted(unique_folders)
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
            # Ensure prefix ends with / if not empty
            if prefix and not prefix.endswith("/"):
                prefix = prefix + "/"

            # Do a single recursive listing and filter by extension client-side.
            # Previously this ran one full recursive glob per extension (6x),
            # each of which walked the entire bucket subtree -> 6 full listings
            # per call, the cause of the long hang on large buckets.
            video_extensions = (".mp4", ".mov", ".avi")
            path = f"{bucket}/{prefix}".rstrip("/")
            all_files = self.gcs.gcs.find(path)

            videos = []
            for file_path in all_files:
                if not file_path.lower().endswith(video_extensions):
                    continue

                # Extract filename
                filename = file_path.split("/")[-1]

                # Get relative path from bucket
                rel_path = file_path.replace(f"{bucket}/", "", 1)

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

    def exists(self, gcs_path: str) -> bool:
        """Check whether a GCS object exists.

        Args:
            gcs_path: Full GCS path (e.g., "bucket/path/file.csv" or "gs://...")
        """
        if gcs_path.startswith("gs://"):
            gcs_path = gcs_path[5:]
        try:
            return bool(self.gcs.gcs.exists(gcs_path))
        except Exception as e:
            logger.warning(f"exists() check failed for {gcs_path}: {e}")
            return False

    def upload_file(self, local_path: str, gcs_path: str) -> str:
        """Upload a local file to GCS.

        Args:
            local_path: Local source path
            gcs_path: Destination GCS path (e.g., "bucket/path/file.csv")

        Returns:
            The destination GCS path.
        """
        if gcs_path.startswith("gs://"):
            gcs_path = gcs_path[5:]
        self.gcs.upload_file(local_path, gcs_path)
        return gcs_path

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
