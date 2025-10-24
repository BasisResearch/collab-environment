"""
Session discovery and management logic for the dashboard.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from .rclone_client import RcloneClient


@dataclass
class SessionInfo:
    """Information about a data session."""

    name: str
    curated_path: Optional[str] = None
    processed_path: Optional[str] = None
    curated_files: Optional[List[Dict]] = None
    processed_files: Optional[List[Dict]] = None

    def __post_init__(self):
        if self.curated_files is None:
            self.curated_files = []
        if self.processed_files is None:
            self.processed_files = []

    @property
    def has_curated(self) -> bool:
        return self.curated_path is not None

    @property
    def has_processed(self) -> bool:
        return self.processed_path is not None

    @property
    def is_complete(self) -> bool:
        return self.has_curated and self.has_processed


class SessionManager:
    """Manages session discovery and organization across curated and processed buckets."""

    def __init__(
        self,
        rclone_client: RcloneClient,
        curated_bucket: str = "fieldwork_curated",
        processed_bucket: str = "fieldwork_processed",
    ):
        """
        Initialize session manager.

        Args:
            rclone_client: Rclone client instance
            curated_bucket: Name of the curated data bucket
            processed_bucket: Name of the processed data bucket
        """
        self.client = rclone_client
        self.curated_bucket = curated_bucket
        self.processed_bucket = processed_bucket
        self._sessions_cache: Dict[str, SessionInfo] = {}
        self._cache_valid = False

    def discover_sessions(self, force_refresh: bool = False) -> Dict[str, SessionInfo]:
        """
        Discover all sessions across both buckets.

        Args:
            force_refresh: Force refresh of cached session data

        Returns:
            Dictionary mapping session names to SessionInfo objects
        """
        if not force_refresh and self._cache_valid:
            return self._sessions_cache

        logger.info("Discovering sessions across buckets...")

        # Get top-level directories from both buckets
        curated_dirs = self._get_top_level_directories(self.curated_bucket)
        processed_dirs = self._get_top_level_directories(self.processed_bucket)

        # Find session names that exist in both buckets
        all_session_names = set(curated_dirs.keys()) | set(processed_dirs.keys())

        sessions = {}
        for session_name in all_session_names:
            session = SessionInfo(name=session_name)

            # Set paths if they exist
            if session_name in curated_dirs:
                session.curated_path = curated_dirs[session_name]

            if session_name in processed_dirs:
                session.processed_path = processed_dirs[session_name]

            sessions[session_name] = session

        self._sessions_cache = sessions
        self._cache_valid = True

        logger.info(f"Discovered {len(sessions)} sessions")
        logger.info(
            f"Complete sessions (both curated & processed): {sum(1 for s in sessions.values() if s.is_complete)}"
        )

        return sessions

    def _get_top_level_directories(self, bucket: str) -> Dict[str, str]:
        """
        Get top-level directories in a bucket.

        Args:
            bucket: Bucket name

        Returns:
            Dictionary mapping directory names to their paths
        """
        try:
            items = self.client.list_directory(bucket, "")
            directories = {}

            for item in items:
                if item.get("IsDir", False):
                    name = item["Name"]
                    directories[name] = name

            return directories

        except Exception as e:
            logger.error(f"Failed to list directories in bucket {bucket}: {e}")
            return {}

    def get_session(
        self, session_name: str, force_refresh: bool = False
    ) -> Optional[SessionInfo]:
        """
        Get information about a specific session.

        Args:
            session_name: Name of the session
            force_refresh: Force refresh of session data

        Returns:
            SessionInfo object or None if not found
        """
        sessions = self.discover_sessions(force_refresh)
        return sessions.get(session_name)

    def load_session_files(self, session_name: str) -> Optional[SessionInfo]:
        """
        Load detailed file information for a session.

        Args:
            session_name: Name of the session

        Returns:
            SessionInfo with file listings populated
        """
        session = self.get_session(session_name)
        if not session:
            return None

        # Load curated files
        if session.has_curated and session.curated_path:
            try:
                session.curated_files = self._load_directory_tree(
                    self.curated_bucket, session.curated_path
                )
            except Exception as e:
                logger.error(f"Failed to load curated files for {session_name}: {e}")

        # Load processed files
        if session.has_processed and session.processed_path:
            try:
                session.processed_files = self._load_directory_tree(
                    self.processed_bucket, session.processed_path
                )
            except Exception as e:
                logger.error(f"Failed to load processed files for {session_name}: {e}")

        return session

    def _load_directory_tree(self, bucket: str, path: str) -> List[Dict]:
        """
        Recursively load all files in a directory tree.

        Args:
            bucket: Bucket name
            path: Directory path

        Returns:
            List of file/directory information with full paths
        """
        all_items = []

        def _recursive_list(current_path: str, relative_path: str = ""):
            items = self.client.list_directory(bucket, current_path)

            for item in items:
                # Add relative path information
                item_copy = item.copy()
                item_copy["RelativePath"] = (
                    f"{relative_path}/{item['Name']}" if relative_path else item["Name"]
                )
                item_copy["FullPath"] = (
                    f"{current_path}/{item['Name']}" if current_path else item["Name"]
                )
                item_copy["Bucket"] = bucket

                all_items.append(item_copy)

                # Recursively process directories
                if item.get("IsDir", False):
                    next_path = (
                        f"{current_path}/{item['Name']}"
                        if current_path
                        else item["Name"]
                    )
                    next_relative = (
                        f"{relative_path}/{item['Name']}"
                        if relative_path
                        else item["Name"]
                    )
                    _recursive_list(next_path, next_relative)

        _recursive_list(path)
        return all_items

    def get_file_path(
        self, session_name: str, file_type: str, relative_path: str
    ) -> Tuple[str, str]:
        """
        Get the full bucket and path for a file in a session.

        Args:
            session_name: Name of the session
            file_type: Either "curated" or "processed"
            relative_path: Relative path within the session

        Returns:
            Tuple of (bucket_name, full_file_path)
        """
        session = self.get_session(session_name)
        if not session:
            raise ValueError(f"Session {session_name} not found")

        if file_type == "curated":
            if not session.has_curated:
                raise ValueError(f"Session {session_name} has no curated data")
            bucket = self.curated_bucket
            base_path = session.curated_path
        elif file_type == "processed":
            if not session.has_processed:
                raise ValueError(f"Session {session_name} has no processed data")
            bucket = self.processed_bucket
            base_path = session.processed_path
        else:
            raise ValueError(f"Invalid file_type: {file_type}")

        full_path = f"{base_path}/{relative_path.strip('/')}"
        return bucket, full_path

    def invalidate_cache(self):
        """Invalidate the sessions cache to force refresh on next access."""
        self._cache_valid = False
        self._sessions_cache.clear()
