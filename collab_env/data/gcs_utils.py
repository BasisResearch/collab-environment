import os
from pathlib import Path
import gcsfs
from dotenv import load_dotenv
from google.cloud import storage
from google.oauth2 import service_account
from loguru import logger

from collab_env.data.file_utils import expand_path, get_project_root

DEFAULT_PROJECT_ID = "collab-data-463313"
DEFAULT_GCS_CREDENTIALS_PATH = "config-local/collab-data-463313-c340ad86b28e.json"

class GCSClient:
    is_initialized = False

    def __init__(
        self,
        project_id: str = DEFAULT_PROJECT_ID,
        credentials_path: str | Path | None = None,
    ):
        """
        Args:
            credentials_path: Path to GCS credentials file. If not provided, will use the default path.
        """
        if credentials_path is None:
            credentials_path = expand_path(
                DEFAULT_GCS_CREDENTIALS_PATH, get_project_root()
            )

        self.credentials_path = credentials_path
        assert os.path.exists(self.credentials_path), (
            f"Credentials file {self.credentials_path} does not exist"
        )
        logger.info(f"Using credentials from {self.credentials_path}")
        self.credentials = service_account.Credentials.from_service_account_file(
            self.credentials_path
        )

        self.project_id = project_id
        logger.info(f"Using project {self.project_id}")

        self._gcs = gcsfs.GCSFileSystem(
            self.project_id, token=str(self.credentials_path)
        )
        self._storage_client = storage.Client(
            self.project_id, credentials=self.credentials
        )

        self.is_initialized = True

    @property
    def gcs(self):
        assert self.is_initialized, (
            "GCSClient must be initialized before accessing the GCSFileSystem"
        )
        return self._gcs

    def create_bucket(self, bucket_name: str):
        """
        Create a bucket in GCS.
        """
        assert self.is_initialized, (
            "GCSClient must be initialized before creating a bucket"
        )
        self._gcs.mkdir(bucket_name)
        logger.info(f"Created bucket {bucket_name}.")

    def list_buckets(self) -> list[str]:
        """
        List all buckets in GCS.
        """
        assert self.is_initialized, (
            "GCSClient must be initialized before listing buckets"
        )
        return [b.name for b in self._storage_client.list_buckets()]

    def delete_bucket(self, bucket_name: str):
        """
        Delete a bucket in GCS.
        """
        assert self.is_initialized, (
            "GCSClient must be initialized before deleting a bucket"
        )
        bucket = self._storage_client.bucket(bucket_name)
        bucket.delete()
        logger.info(f"Deleted bucket {bucket_name}.")

    def create_folder(self, folder_path: str):
        """
        Create a folder in GCS.

        Args:
            folder_path: The GCS folder path, e.g. "bucket_name/path/to/folder/"
        """
        assert self.is_initialized, (
            "GCSClient must be initialized before creating a folder"
        )

        # Ensure folder_path ends with a slash
        if not folder_path.endswith("/"):
            folder_path += "/"

        # Split bucket and path
        if "/" not in folder_path:
            raise ValueError(
                "folder_path must be in the form 'bucket_name/path/to/folder/'"
            )
        bucket, *path_parts = folder_path.split("/", 1)
        if not path_parts:
            raise ValueError("folder_path must include a path after the bucket name")
        path = path_parts[0]

        folder_gcs_path = f"{bucket}/{path}"
        # GCS does not have real folders, but we can create a zero-length blob as a marker
        marker_blob = folder_gcs_path.rstrip("/") + "/.folder_marker"
        if not self._gcs.exists(marker_blob):
            with self._gcs.open(marker_blob, "wb"):
                pass  # create empty file as folder marker
            logger.info(f"Created folder marker at gs://{marker_blob}")
        else:
            logger.warning(f"Folder marker already exists at gs://{marker_blob}")

    def remove_folder(self, folder_path: str, require_marker: bool = True):
        """
        Remove a folder in GCS.
        """
        assert self.is_initialized, (
            "GCSClient must be initialized before removing a folder"
        )

        # check if the folder path ends with a slash
        if not folder_path.endswith("/"):
            folder_path += "/"

        # check if the folder exists
        if not self._gcs.exists(folder_path):
            logger.error(f"Folder {folder_path} does not exist")
            raise ValueError(f"Folder {folder_path} does not exist")

        # check if files besides the marker exist
        marker_blob = folder_path.rstrip("/") + "/.folder_marker"
        marker_exists = self._gcs.exists(marker_blob)
        files = self.glob(f"{folder_path}/*")

        if require_marker and not marker_exists:
            logger.error("Marker does not exist at gs://{marker_blob}")
            raise ValueError(f"Marker does not exist at gs://{marker_blob}")

        if len(files) > 1 - int(marker_exists):
            logger.error(f"Folder {folder_path} is not empty")
            raise ValueError(f"Folder {folder_path} is not empty")

        if marker_exists:
            self._gcs.rm(marker_blob)
            logger.info(f"Removed folder marker at gs://{marker_blob}")

        # by now, the folder should not exist
        assert not self._gcs.exists(folder_path), f"Folder {folder_path} still exists"

        logger.info(f"Removed folder {folder_path}")

    def glob(self, pattern: str):
        """
        Glob for files in a bucket.
        """
        assert self.is_initialized, (
            "GCSClient must be initialized before creating a folder"
        )

        return self._gcs.glob(pattern)

    def upload_file(self, local_path: str, gcs_path: str):
        """
        Upload a file to GCS.
        """
        assert self.is_initialized, (
            "GCSClient must be initialized before uploading a file"
        )
        logger.info(f"Uploading file {local_path} to {gcs_path}.")
        self._gcs.put(local_path, gcs_path)
        logger.info(f"Uploaded file {local_path} to {gcs_path}.")

    def delete_path(self, gcs_path: str):
        """
        Delete a path in GCS.
        """
        assert self.is_initialized, (
            "GCSClient must be initialized before deleting a path"
        )

        self._gcs.rm(gcs_path)
        logger.info(f"Deleted path {gcs_path}.")

    def is_folder(self, path: str) -> bool:
        """
        Check if a path represents a folder in GCS.
        
        In GCS, folders are logical constructs. This method checks:
        1. If a folder marker exists (.folder_marker file)
        2. If any objects exist with the path as a prefix
        3. If the path itself exists as an object
        
        Args:
            path: The GCS path to check
            
        Returns:
            bool: True if the path represents a folder, False otherwise
        """
        assert self.is_initialized, (
            "GCSClient must be initialized before checking if path is folder"
        )
        
        # Ensure path ends with slash for folder check
        if not path.endswith("/"):
            path += "/"
        
        # Method 1: Check for folder marker
        marker_blob = path.rstrip("/") + "/.folder_marker"
        if self._gcs.exists(marker_blob):
            return True
        
        # Method 2: Check if any objects exist with this path prefix
        objects = self.glob(f"{path}*")
        if len(objects) > 0:
            return True
        
        # Method 3: Check if the path itself exists as an object
        if self._gcs.exists(path.rstrip("/")):
            return True
        
        return False

    def list_folder_contents(self, folder_path: str) -> list[str]:
        """
        List contents of a folder in GCS.
        
        Args:
            folder_path: The GCS folder path
            
        Returns:
            list[str]: List of object paths in the folder
        """
        assert self.is_initialized, (
            "GCSClient must be initialized before listing folder contents"
        )
        
        if not folder_path.endswith("/"):
            folder_path += "/"
        
        return self.glob(f"{folder_path}*")
