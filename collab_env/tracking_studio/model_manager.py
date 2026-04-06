"""
Model Manager Component

Handles loading and managing detection models (YOLO and Roboflow).
"""

import os
from pathlib import Path
from typing import List, Optional
from loguru import logger

from ultralytics import YOLO


class ModelManager:
    """Manager for detection models (YOLO and Roboflow)"""

    def __init__(self, roboflow_api_key: Optional[str] = None):
        """
        Initialize model manager.

        Args:
            roboflow_api_key: Roboflow API key (or read from env)
        """
        self.roboflow_api_key = roboflow_api_key or os.getenv("ROBOFLOW_API_KEY")
        self.local_models_dir = Path("/workspace/models")

        # Check if running locally (models in ~/.cache/ultralytics)
        if not self.local_models_dir.exists():
            # Use default Ultralytics cache directory
            self.local_models_dir = Path.home() / ".cache" / "ultralytics"

        logger.info(f"Model directory: {self.local_models_dir}")

    def list_local_yolo_models(self) -> List[str]:
        """
        Return available YOLO models (YOLO11 and YOLO26 variants).

        Returns:
            List of model filenames
        """
        # Auto-downloadable models (Ultralytics will download them)
        auto_downloadable = [
            "yolo11n.pt",
            "yolo11s.pt",
            "yolo11m.pt",
        ]

        # Models that must exist locally (not auto-downloadable)
        local_only = [
            "yolo26n-fast.pt",
            "yolo26s-fast.pt",
            "yolo26m-fast.pt",
        ]

        available = []

        # Add auto-downloadable models (always available)
        available.extend(auto_downloadable)

        # Add local-only models only if they exist
        for model in local_only:
            model_path = self.local_models_dir / model
            if model_path.exists():
                available.append(model)
                logger.debug(f"Found local YOLO26 model: {model}")

        logger.info(f"Available YOLO models: {available}")
        return available

    def load_yolo_model(self, model_name: str) -> YOLO:
        """
        Load YOLO model - will download automatically if available.

        Args:
            model_name: Model filename (e.g., "yolo11n.pt", "yolo26n-fast.pt")

        Returns:
            Loaded YOLO model
        """
        try:
            logger.info(f"Loading YOLO model: {model_name}")
            # Pass directly to YOLO - it will handle local files or auto-download
            model = YOLO(model_name)
            logger.info(f"Successfully loaded YOLO model: {model_name}")
            return model

        except Exception as e:
            logger.error(f"Failed to load YOLO model {model_name}: {e}")
            raise ValueError(
                f"Failed to load model '{model_name}'.\n\n"
                f"Possible solutions:\n"
                f"- Check the model name is correct\n"
                f"- Download manually and place in {self.local_models_dir}\n"
                f"- Use the 'Custom' upload option to upload your .pt file"
            ) from e

    def _validate_roboflow_model_id(self, model_id: str) -> str:
        """
        Validate and format Roboflow model ID.

        Accepts:
        - project/version (e.g., "ratsmerged20260211/1")
        - workspace/project/version (e.g., "myworkspace/ratsmerged20260211/1")

        Returns properly formatted model ID.
        """
        parts = model_id.split("/")

        if len(parts) == 2:
            # project/version format
            logger.info(f"Model ID format: project/version ({model_id})")
            return model_id
        elif len(parts) == 3:
            # workspace/project/version format
            logger.info(f"Model ID format: workspace/project/version ({model_id})")
            return model_id
        else:
            raise ValueError(
                f"Invalid model ID format: {model_id}\n"
                f"Expected: 'project/version' or 'workspace/project/version'"
            )

    def load_roboflow_model(self, model_id: str):
        """
        Load Roboflow model using Inference SDK or local file path.

        Args:
            model_id: Model ID in format "project/version", "workspace/project/version",
                     or a local file path to a .pt file

        Returns:
            Loaded Roboflow model or YOLO model from local file
        """
        # Check if model_id is a local file path
        if (
            model_id.startswith("/")
            or model_id.startswith("~")
            or model_id.endswith(".pt")
        ):
            logger.info(f"Loading Roboflow model from local file: {model_id}")
            model_path = Path(model_id).expanduser()

            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            logger.info(f"Loading YOLO model from: {model_path}")
            model = YOLO(str(model_path))
            logger.info(
                f"Successfully loaded Roboflow model from local file: {model_id}"
            )
            return model

        if not self.roboflow_api_key:
            raise ValueError(
                "ROBOFLOW_API_KEY not set. Please provide API key in environment or constructor."
            )

        # Validate model ID format
        model_id = self._validate_roboflow_model_id(model_id)

        # Try downloading .pt file first (for YOLO native tracking)
        # This provides better performance and supports all ByteTrack parameters
        try:
            logger.info(
                f"Downloading Roboflow model weights for YOLO native tracking: {model_id}"
            )
            model = self._load_roboflow_with_pipeline(model_id)
            logger.info(
                f"Successfully loaded Roboflow model with native tracking: {model_id}"
            )
            return model

        except Exception as download_error:
            # Fallback to get_model() (inference API) if download fails
            logger.warning(f"Download failed: {download_error}")
            logger.info(
                "Attempting fallback: loading with inference API (Supervision tracking)"
            )

            try:
                from inference import get_model

                # Extract project/version from workspace/project/version if needed
                parts = model_id.split("/")
                if len(parts) == 3:
                    # workspace/project/version -> project/version
                    project_version = f"{parts[1]}/{parts[2]}"
                    logger.info(
                        f"Trying to load Roboflow model with get_model(): {project_version}"
                    )
                    model = get_model(
                        model_id=project_version, api_key=self.roboflow_api_key
                    )
                else:
                    # Already project/version format
                    logger.info(
                        f"Trying to load Roboflow model with get_model(): {model_id}"
                    )
                    model = get_model(model_id=model_id, api_key=self.roboflow_api_key)

                logger.info(
                    f"Successfully loaded Roboflow model via inference API: {model_id}"
                )
                return model

            except ImportError:
                logger.error(
                    "inference library not installed. Install with: pip install inference"
                )
                raise
            except Exception as inference_error:
                # Both methods failed
                logger.error(f"Inference API also failed: {inference_error}")
                error_msg = (
                    f"Failed to load Roboflow model '{model_id}'.\n\n"
                    f"Tried:\n"
                    f"1. Downloading model weights (.pt file): {str(download_error)}\n"
                    f"2. Loading via inference API: {str(inference_error)}\n\n"
                    f"Possible solutions:\n"
                    f"- Verify model ID format: workspace/project/version (e.g., 'dima-sdrkv/ratsmerged20260211/1')\n"
                    f"- Check model exists at https://app.roboflow.com/\n"
                    f"- Ensure ROBOFLOW_API_KEY has access to this model\n"
                    f"- Try uploading the .pt file directly using 'Custom' option"
                )
                raise ValueError(error_msg) from inference_error

    def _load_roboflow_with_pipeline(self, model_id: str):
        """
        Fallback: Download Roboflow model weights via /ptFile endpoint.

        This downloads the model weights once via API, then runs inference locally.
        Much faster than HTTP inference for every frame.
        """
        import requests

        logger.info(
            f"Downloading Roboflow model weights for local inference: {model_id}"
        )

        # Parse model ID to get workspace/project/version
        parts = model_id.split("/")
        if len(parts) == 2:
            # project/version format - need workspace
            raise ValueError(
                f"Model ID '{model_id}' missing workspace.\n"
                f"For model download, use format: workspace/project/version"
            )
        elif len(parts) == 3:
            # workspace/project/version format
            workspace, project, version = parts
        else:
            raise ValueError(f"Invalid model ID format: {model_id}")

        # Create cache directory for downloaded models
        cache_dir = self.local_models_dir / "roboflow_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Check if model already downloaded
        model_cache_name = f"{workspace}_{project}_v{version}.pt"
        cached_model_path = cache_dir / model_cache_name

        if cached_model_path.exists():
            logger.info(f"Using cached Roboflow model: {cached_model_path}")
            return YOLO(str(cached_model_path))

        # Download model weights from Roboflow using /ptFile endpoint
        logger.info("Fetching model weights URL from Roboflow API...")

        try:
            # Call /ptFile endpoint to get signed download URL
            ptfile_url = (
                f"https://api.roboflow.com/{workspace}/{project}/{version}/ptFile"
            )
            logger.info(f"Requesting weights URL from: {ptfile_url}")

            response = requests.get(
                ptfile_url, params={"api_key": self.roboflow_api_key}, timeout=10
            )
            response.raise_for_status()

            # Parse response to get weightsUrl
            data = response.json()
            if "weightsUrl" not in data:
                raise ValueError(f"No weightsUrl in response: {data}")

            weights_url = data["weightsUrl"]
            logger.info("Got weights URL, downloading...")

            # Download the .pt file from signed URL
            response = requests.get(weights_url, stream=True, timeout=120)
            response.raise_for_status()

            # Save to cache
            with open(cached_model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(
                f"Downloaded model weights: {cached_model_path} ({cached_model_path.stat().st_size} bytes)"
            )

            # Load with Ultralytics YOLO
            model = YOLO(str(cached_model_path))
            logger.info(
                f"Successfully loaded Roboflow model for local inference: {model_id}"
            )
            return model

        except requests.exceptions.HTTPError as e:
            error_msg = (
                f"Failed to download Roboflow model weights for '{model_id}'.\n\n"
                f"HTTP Error {e.response.status_code}: {e.response.text[:200]}\n\n"
                f"Possible solutions:\n"
                f"1. Verify model ID format is workspace/project/version\n"
                f"2. Check ROBOFLOW_API_KEY has access to this model\n"
                f"3. Ensure model exists at https://app.roboflow.com/\n"
                f"4. Upload model weights manually via 'Custom' option"
            )
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        except Exception as e:
            error_msg = (
                f"Failed to download Roboflow model weights for '{model_id}'.\n\n"
                f"Error: {str(e)}\n\n"
                f"Try uploading model weights manually via 'Custom' option."
            )
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    def list_roboflow_projects(self) -> List[str]:
        """
        Query Roboflow API for all projects in the workspace tied to the API key.

        Returns:
            List of project IDs in "workspace/project" format, sorted alphabetically.
        """
        import requests

        if not self.roboflow_api_key:
            raise ValueError("ROBOFLOW_API_KEY not set")

        try:
            # Root endpoint with API key returns workspace info (may include
            # workspace name and/or a nested workspace object with projects).
            root = requests.get(
                "https://api.roboflow.com/",
                params={"api_key": self.roboflow_api_key},
                timeout=10,
            )
            root.raise_for_status()
            root_data = root.json()
            logger.debug(f"Roboflow root response keys: {list(root_data.keys())}")

            # Collect candidate workspace names from various possible shapes
            workspace_names: List[str] = []
            ws_field = root_data.get("workspace")
            if isinstance(ws_field, str):
                workspace_names.append(ws_field)
            elif isinstance(ws_field, dict):
                name = ws_field.get("url") or ws_field.get("name")
                if name:
                    workspace_names.append(name)
            for w in root_data.get("workspaces", []) or []:
                if isinstance(w, str):
                    workspace_names.append(w)
                elif isinstance(w, dict):
                    name = w.get("url") or w.get("name")
                    if name:
                        workspace_names.append(name)

            if not workspace_names:
                raise ValueError(
                    f"Could not resolve any workspace from API key. "
                    f"Root response: {root_data}"
                )

            project_ids: List[str] = []
            for workspace in workspace_names:
                ws = requests.get(
                    f"https://api.roboflow.com/{workspace}",
                    params={"api_key": self.roboflow_api_key},
                    timeout=10,
                )
                ws.raise_for_status()
                data = ws.json()
                projects = data.get("workspace", {}).get("projects") or data.get(
                    "projects"
                ) or []
                logger.info(
                    f"Roboflow workspace '{workspace}': {len(projects)} projects"
                )
                for p in projects:
                    if isinstance(p, str):
                        pid = p
                    else:
                        pid = p.get("id") or p.get("url") or p.get("name") or ""
                    if not pid:
                        continue
                    if "/" not in pid:
                        pid = f"{workspace}/{pid}"
                    project_ids.append(pid)

            project_ids = sorted(set(project_ids))
            logger.info(
                f"Found {len(project_ids)} total Roboflow projects across "
                f"{len(workspace_names)} workspace(s)"
            )
            return project_ids
        except requests.exceptions.HTTPError as e:
            error_msg = f"Failed to list Roboflow projects: HTTP {e.response.status_code}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to list Roboflow projects: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    def list_roboflow_project_models(self, project_id: str) -> List[dict]:
        """
        Query Roboflow API for available model versions in a project.

        Args:
            project_id: Project ID in format "workspace/project" (e.g., "dima-sdrkv/ratsmerged20260211")

        Returns:
            List of dicts with keys: version, name, images, map
        """
        import requests

        if not self.roboflow_api_key:
            raise ValueError("ROBOFLOW_API_KEY not set")

        parts = project_id.split("/")
        if len(parts) != 2:
            raise ValueError("Project ID must be in format: workspace/project")

        workspace, project = parts

        try:
            url = f"https://api.roboflow.com/{workspace}/{project}"
            logger.info(f"Querying Roboflow project models: {url}")

            response = requests.get(
                url, params={"api_key": self.roboflow_api_key}, timeout=10
            )
            response.raise_for_status()

            data = response.json()

            versions = []
            if "versions" in data:
                for vd in data["versions"]:
                    version_num = vd.get("id", "")
                    if isinstance(version_num, str) and "/" in version_num:
                        version_num = version_num.split("/")[-1]
                    if not version_num:
                        version_num = vd.get("version")
                    if not version_num:
                        continue

                    map_val = vd.get("model", {}).get("map", "")
                    if map_val and str(map_val) != "NaN":
                        map_str = f"{float(map_val):.1f}%"
                    else:
                        map_str = ""

                    versions.append(
                        {
                            "version": str(version_num),
                            "name": vd.get("name", ""),
                            "images": vd.get("images", 0),
                            "map": map_str,
                            "raw": vd,
                        }
                    )

            versions.sort(
                key=lambda x: int(x["version"]) if x["version"].isdigit() else 0,
                reverse=True,
            )
            logger.info(
                f"Found {len(versions)} versions: {[v['version'] for v in versions]}"
            )
            return versions

        except requests.exceptions.HTTPError as e:
            error_msg = (
                f"Failed to query Roboflow project: HTTP {e.response.status_code}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to query Roboflow project: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
