"""
Rclone integration utilities for GCS bucket access.
"""

import subprocess
import json
from typing import List, Dict, Any
from loguru import logger


class RcloneClient:
    """Client for interacting with GCS buckets via rclone."""

    def __init__(self, remote_name: str = "collab-data"):
        """
        Initialize rclone client.

        Args:
            remote_name: Name of the rclone remote configured for GCS access
        """
        self.remote_name = remote_name
        self._verify_rclone()
        self._verify_remote()

    def _verify_rclone(self):
        """Verify rclone is installed and accessible."""
        try:
            result = subprocess.run(
                ["rclone", "version"], capture_output=True, text=True, check=True
            )
            logger.info(f"Rclone version: {result.stdout.split()[1]}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                "rclone not found. Please install rclone and ensure it's in PATH."
            ) from e

    def _verify_remote(self):
        """Verify the remote is configured."""
        try:
            result = subprocess.run(
                ["rclone", "listremotes"], capture_output=True, text=True, check=True
            )
            remotes = [
                line.strip().rstrip(":")
                for line in result.stdout.strip().split("\n")
                if line.strip()
            ]
            if self.remote_name not in remotes:
                raise RuntimeError(
                    f"Remote '{self.remote_name}' not found. Available remotes: {remotes}"
                )
            logger.info(f"Using rclone remote: {self.remote_name}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to list rclone remotes: {e}") from e

    def list_buckets(self) -> List[str]:
        """List all buckets in the remote."""
        try:
            result = subprocess.run(
                ["rclone", "lsd", f"{self.remote_name}:"],
                capture_output=True,
                text=True,
                check=True,
            )
            buckets = []
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    # Extract bucket name from rclone lsd output
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        bucket_name = parts[-1]
                        buckets.append(bucket_name)
            return buckets
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to list buckets: {e}")
            return []

    def list_directory(self, bucket: str, path: str = "") -> List[Dict[str, Any]]:
        """
        List contents of a directory in a bucket.

        Args:
            bucket: Bucket name
            path: Path within bucket (empty for root)

        Returns:
            List of dictionaries with file/directory information
        """
        remote_path = f"{self.remote_name}:{bucket}"
        if path:
            remote_path += f"/{path.strip('/')}"

        try:
            # Use rclone lsjson for structured output
            result = subprocess.run(
                ["rclone", "lsjson", remote_path],
                capture_output=True,
                text=True,
                check=True,
            )

            if not result.stdout.strip():
                return []

            items = json.loads(result.stdout)
            return items

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to list directory {remote_path}: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse rclone output: {e}")
            return []

    def read_file(self, bucket: str, file_path: str) -> bytes:
        """
        Read a file from GCS bucket.

        Args:
            bucket: Bucket name
            file_path: Path to file within bucket

        Returns:
            File contents as bytes
        """
        remote_path = f"{self.remote_name}:{bucket}/{file_path.strip('/')}"

        try:
            result = subprocess.run(
                ["rclone", "cat", remote_path], capture_output=True, check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to read file {remote_path}: {e}")
            raise

    def read_text_file(
        self, bucket: str, file_path: str, encoding: str = "utf-8"
    ) -> str:
        """
        Read a text file from GCS bucket.

        Args:
            bucket: Bucket name
            file_path: Path to file within bucket
            encoding: Text encoding

        Returns:
            File contents as string
        """
        content = self.read_file(bucket, file_path)
        return content.decode(encoding)

    def write_file(self, bucket: str, file_path: str, content: bytes):
        """
        Write a file to GCS bucket.

        Args:
            bucket: Bucket name
            file_path: Path to file within bucket
            content: File contents as bytes
        """
        remote_path = f"{self.remote_name}:{bucket}/{file_path.strip('/')}"

        try:
            # Use rclone rcat to write content
            subprocess.run(["rclone", "rcat", "--gcs-bucket-policy-only", remote_path], input=content, check=True)
            logger.info(f"Successfully wrote file to {remote_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to write file {remote_path}: {e}")
            raise

    def write_text_file(
        self, bucket: str, file_path: str, content: str, encoding: str = "utf-8"
    ):
        """
        Write a text file to GCS bucket.

        Args:
            bucket: Bucket name
            file_path: Path to file within bucket
            content: File contents as string
            encoding: Text encoding
        """
        self.write_file(bucket, file_path, content.encode(encoding))

    def start_http_serve(self, bucket: str, port: int = 8080) -> subprocess.Popen:
        """
        Start rclone serve http for read-only access to a bucket.

        Args:
            bucket: Bucket name to serve
            port: HTTP port to serve on

        Returns:
            Process object for the rclone serve command
        """
        remote_path = f"{self.remote_name}:{bucket}"

        try:
            process = subprocess.Popen(
                ["rclone", "serve", "http", remote_path, "--addr", f":{port}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            logger.info(f"Started rclone HTTP server for {bucket} on port {port}")
            return process
        except Exception as e:
            logger.error(f"Failed to start rclone HTTP server: {e}")
            raise

    def file_exists(self, bucket: str, file_path: str) -> bool:
        """
        Check if a file exists in the bucket.

        Args:
            bucket: Bucket name
            file_path: Path to file within bucket

        Returns:
            True if file exists, False otherwise
        """
        remote_path = f"{self.remote_name}:{bucket}/{file_path.strip('/')}"

        try:
            subprocess.run(
                ["rclone", "lsl", remote_path], capture_output=True, check=True
            )
            return True
        except subprocess.CalledProcessError:
            return False
    
    def copy_file(self, source_bucket: str, source_path: str, dest_bucket: str, dest_path: str) -> bool:
        """
        Copy a file from one location to another using rclone copyto.
        
        Args:
            source_bucket: Source bucket name
            source_path: Source file path within bucket
            dest_bucket: Destination bucket name  
            dest_path: Destination file path within bucket
            
        Returns:
            True if copy successful, False otherwise
        """
        source_remote_path = f"{self.remote_name}:{source_bucket}/{source_path.strip('/')}"
        dest_remote_path = f"{self.remote_name}:{dest_bucket}/{dest_path.strip('/')}"
        
        try:
            result = subprocess.run(
                ["rclone", "copyto", "--gcs-bucket-policy-only", source_remote_path, dest_remote_path],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully copied file: {source_bucket}/{source_path} -> {dest_bucket}/{dest_path}")
                return True
            else:
                logger.error(f"Failed to copy file: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error copying file: {e}")
            return False
    
    def copy_local_to_remote(self, local_path: str, bucket: str, remote_path: str) -> bool:
        """
        Copy a local file to remote bucket using rclone copyto.
        
        Args:
            local_path: Local file path
            bucket: Destination bucket name
            remote_path: Destination file path within bucket
            
        Returns:
            True if copy successful, False otherwise
        """
        dest_remote_path = f"{self.remote_name}:{bucket}/{remote_path.strip('/')}"
        
        try:
            result = subprocess.run(
                ["rclone", "copyto", "--gcs-bucket-policy-only", local_path, dest_remote_path],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully copied local file to remote: {local_path} -> {bucket}/{remote_path}")
                return True
            else:
                logger.error(f"Failed to copy local file to remote: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error copying local file to remote: {e}")
            return False
    
    def get_file_checksum(self, bucket: str, file_path: str, hash_type: str = "md5") -> str:
        """
        Get checksum of a remote file using rclone hashsum.
        
        Args:
            bucket: Bucket name
            file_path: Path to file within bucket
            hash_type: Hash algorithm (md5, sha1, sha256, etc.)
            
        Returns:
            Checksum string, or empty string if failed
        """
        remote_path = f"{self.remote_name}:{bucket}/{file_path.strip('/')}"
        
        try:
            result = subprocess.run(
                ["rclone", "hashsum", hash_type, remote_path],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                # rclone hashsum output format: "checksum filename"
                output = result.stdout.strip()
                if output:
                    checksum = output.split()[0]
                    logger.debug(f"Remote {hash_type} checksum for {bucket}/{file_path}: {checksum}")
                    return checksum
                    
            logger.warning(f"Failed to get remote checksum: {result.stderr}")
            return ""
            
        except Exception as e:
            logger.error(f"Error getting remote checksum: {e}")
            return ""
