"""
Simulation data loader for boid simulation parquet files and config.yaml.
"""

import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

from collab_env.data.file_utils import expand_path, get_project_root

logger = logging.getLogger(__name__)


class SimulationDataLoader:
    """Loads and manages boid simulation data from parquet files and config.yaml."""

    def __init__(self):
        self.registered_simulations: Dict[str, Dict[str, Any]] = {}
        self.loaded_episodes: Dict[
            str, Dict[str, Any]
        ] = {}  # simulation_id -> episode_data

    def register_simulation(
        self, simulation_id: str, folder_path: str, config_path: str
    ) -> Dict[str, Any]:
        """Register a simulation folder and parse its config."""
        try:
            folder_path = Path(folder_path)
            config_path = Path(config_path)

            if not folder_path.exists():
                raise FileNotFoundError(f"Simulation folder not found: {folder_path}")
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            # Parse config.yaml
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            # Discover episode files
            episode_files = self._discover_episodes(folder_path)

            # Resolve mesh paths
            mesh_status = self._check_mesh_files(config)

            # Extract key parameters
            sim_info = {
                "id": simulation_id,
                "name": self._generate_name(folder_path.name),
                "folder_path": str(folder_path),
                "config_path": str(config_path),
                "config": config,
                "episode_files": episode_files,
                "num_episodes": len(episode_files),
                "num_agents": config.get("simulator", {}).get("num_agents", 0),
                "num_frames": config.get("simulator", {}).get("num_frames", 0),
                "mesh_status": mesh_status,
            }

            self.registered_simulations[simulation_id] = sim_info
            logger.info(f"Registered simulation {simulation_id}: {sim_info['name']}")

            return sim_info

        except Exception as e:
            logger.error(f"Error registering simulation {simulation_id}: {e}")
            raise

    def get_simulations(self) -> List[Dict[str, Any]]:
        """Get list of all registered simulations."""
        return list(self.registered_simulations.values())

    def get_episodes(self, simulation_id: str) -> List[Dict[str, Any]]:
        """Get list of episodes for a simulation."""
        if simulation_id not in self.registered_simulations:
            raise ValueError(f"Simulation not found: {simulation_id}")

        sim_info = self.registered_simulations[simulation_id]
        episodes = []

        for i, episode_file in enumerate(sim_info["episode_files"]):
            episodes.append(
                {
                    "id": i,
                    "name": f"Episode {i}",
                    "filename": Path(episode_file).name,
                    "path": episode_file,
                }
            )

        return episodes

    def load_episode(self, simulation_id: str, episode_id: int) -> Dict[str, Any]:
        """Load episode data and convert to track format."""
        if simulation_id not in self.registered_simulations:
            raise ValueError(f"Simulation not found: {simulation_id}")

        sim_info = self.registered_simulations[simulation_id]

        if episode_id >= len(sim_info["episode_files"]):
            raise ValueError(f"Episode {episode_id} not found")

        episode_file = sim_info["episode_files"][episode_id]

        try:
            # Load parquet file
            df = pd.read_parquet(episode_file)
            logger.info(
                f"Loaded episode {episode_id} from {Path(episode_file).name}: {df.shape}"
            )

            # Convert to track format
            tracks_by_frame = self._convert_to_tracks(df)

            # Get mesh paths
            mesh_paths = self._get_mesh_paths(sim_info["config"])

            episode_data = {
                "frames": tracks_by_frame,
                "config": {
                    "num_agents": sim_info["num_agents"],
                    "num_frames": sim_info["num_frames"],
                    "meshes": mesh_paths,
                    "scene_scale": sim_info["config"]
                    .get("environment", {})
                    .get("scene_scale", 300.0),
                    "scene_position": sim_info["config"]
                    .get("environment", {})
                    .get("scene_position", [0, 0, 0]),
                    "scene_angle": sim_info["config"]
                    .get("meshes", {})
                    .get("scene_angle", [0, 0, 0]),
                },
                "num_frames": len(tracks_by_frame),
                "num_tracks": len(df["id"].unique()) if "id" in df.columns else 0,
            }

            # Cache the loaded episode
            cache_key = f"{simulation_id}_{episode_id}"
            self.loaded_episodes[cache_key] = episode_data

            return episode_data

        except Exception as e:
            logger.error(f"Error loading episode {episode_id} from {episode_file}: {e}")
            raise

    def _discover_episodes(self, folder_path: Path) -> List[str]:
        """Discover episode parquet files in simulation folder."""
        episode_files = []

        for parquet_file in folder_path.glob("episode-*.parquet"):
            episode_files.append(str(parquet_file))

        # Sort by episode number
        episode_files.sort(key=lambda f: self._extract_episode_number(Path(f).name))

        return episode_files

    def _extract_episode_number(self, filename: str) -> int:
        """Extract episode number from filename like 'episode-0-completed-*.parquet'."""
        try:
            parts = filename.split("-")
            if len(parts) >= 2 and parts[0] == "episode":
                return int(parts[1])
        except (ValueError, IndexError):
            pass
        return 0

    def _generate_name(self, folder_name: str) -> str:
        """Generate a human-readable name from folder name."""
        # Convert folder name like "hackathon-boid-small-200-align-cohesion_sim_run-started-20250926-214330"
        # to "Hackathon Boid Small 200 Align Cohesion"
        name = folder_name.replace("_sim_run", "").replace("-started-", " ")
        # Remove timestamp at the end
        parts = name.split("-")
        if len(parts) > 0 and parts[-1].isdigit():
            parts = parts[:-1]
        if len(parts) > 0 and parts[-1].isdigit():
            parts = parts[:-1]

        return " ".join(word.capitalize() for word in parts)

    def _check_mesh_files(self, config: Dict[str, Any]) -> Dict[str, bool]:
        """Check if mesh files exist."""
        mesh_status = {"scene_found": False, "target_found": False}

        try:
            meshes_config = config.get("meshes", {})

            # Check scene mesh
            scene_mesh = meshes_config.get("mesh_scene")
            if scene_mesh:
                scene_path = expand_path(scene_mesh, get_project_root())
                mesh_status["scene_found"] = Path(scene_path).exists()

            # Check target submesh
            sub_mesh_target = meshes_config.get("sub_mesh_target")
            if sub_mesh_target:
                if isinstance(sub_mesh_target, list) and len(sub_mesh_target) > 0:
                    target_path = expand_path(sub_mesh_target[0], get_project_root())
                    mesh_status["target_found"] = Path(target_path).exists()
                elif isinstance(sub_mesh_target, str):
                    target_path = expand_path(sub_mesh_target, get_project_root())
                    mesh_status["target_found"] = Path(target_path).exists()

        except Exception as e:
            logger.warning(f"Error checking mesh files: {e}")

        return mesh_status

    def _get_mesh_paths(self, config: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """Get resolved mesh file paths."""
        mesh_paths = {"scene_path": None, "target_path": None}

        try:
            meshes_config = config.get("meshes", {})

            # Scene mesh
            scene_mesh = meshes_config.get("mesh_scene")
            if scene_mesh:
                mesh_paths["scene_path"] = str(
                    expand_path(scene_mesh, get_project_root())
                )

            # Target submesh
            sub_mesh_target = meshes_config.get("sub_mesh_target")
            if sub_mesh_target:
                if isinstance(sub_mesh_target, list) and len(sub_mesh_target) > 0:
                    mesh_paths["target_path"] = str(
                        expand_path(sub_mesh_target[0], get_project_root())
                    )
                elif isinstance(sub_mesh_target, str):
                    mesh_paths["target_path"] = str(
                        expand_path(sub_mesh_target, get_project_root())
                    )

        except Exception as e:
            logger.warning(f"Error resolving mesh paths: {e}")

        return mesh_paths

    def _convert_to_tracks(self, df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """Convert parquet DataFrame to track format compatible with mesh viewer."""
        tracks_by_frame = {}

        for _, row in df.iterrows():
            frame = int(row["time"])  # "time" column contains frame number

            if frame not in tracks_by_frame:
                tracks_by_frame[frame] = []

            track_point = {
                "track_id": int(row["id"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
                "type": str(row.get("type", "agent")),
            }

            # Add velocity if available
            if "v_x" in row and pd.notna(row["v_x"]):
                track_point["v_x"] = float(row["v_x"])
                track_point["v_y"] = float(row["v_y"])
                track_point["v_z"] = float(row["v_z"])

            tracks_by_frame[frame].append(track_point)

        return tracks_by_frame
