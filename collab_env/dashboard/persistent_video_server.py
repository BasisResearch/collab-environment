#!/usr/bin/env python3
"""
Persistent Flask server for video and mesh viewing with proper templates.
Refactored version with clean separation of concerns.
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
import pandas as pd
from pathlib import Path
from threading import Lock
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app with proper template and static folders
app = Flask(__name__, template_folder="templates", static_folder="static")

# Thread-safe storage for videos and meshes
videos_data: dict[str, dict] = {}
meshes_data: dict[str, dict] = {}
data_lock = Lock()


def convert_camera_params_to_json(params):
    """Convert camera parameters to JSON-serializable format."""
    try:
        import numpy as np

        def convert_value(obj):
            """Recursively convert numpy arrays and other types to JSON-serializable format."""
            if obj is None:
                return None
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_value(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_value(v) for v in obj]
            elif hasattr(obj, "cpu") and hasattr(obj, "numpy"):
                # PyTorch tensor
                return convert_value(obj.cpu().numpy())
            elif isinstance(obj, (int, float, str)):
                return obj
            else:
                logger.warning(f"Unknown type {type(obj)}, converting to string")
                return str(obj)

        if params is None:
            return None

        result = convert_value(params)
        return result

    except Exception as e:
        logger.error(f"Error converting camera params: {e}")
        return None


# Routes for serving HTML templates
@app.route("/")
def index():
    """Main page showing available viewers."""
    return render_template("index.html")


@app.route("/video")
def video_viewer():
    """2D video viewer with bbox overlays."""
    return render_template("video_viewer.html")


@app.route("/3d")
def mesh_viewer():
    """3D mesh viewer with track animations."""
    return render_template("mesh_viewer.html")


@app.route("/sync")
def sync_viewer():
    """Synchronized 2D + 3D viewer."""
    return render_template("sync_viewer.html")


# API Routes for data management
@app.route("/api/videos", methods=["GET"])
def api_list_videos():
    """List all available videos."""
    with data_lock:
        return jsonify(
            [
                {
                    "id": vid_id,
                    "name": data["name"],
                    "has_bboxes": data.get("has_bboxes", False),
                }
                for vid_id, data in videos_data.items()
            ]
        )


@app.route("/api/meshes", methods=["GET"])
def api_list_meshes():
    """List all available meshes."""
    with data_lock:
        return jsonify(
            [
                {
                    "id": mesh_id,
                    "name": data["name"],
                    "frames": data.get("num_frames", 0),
                    "tracks": data.get("num_tracks", 0),
                }
                for mesh_id, data in meshes_data.items()
            ]
        )


@app.route("/api/add_video", methods=["POST"])
def api_add_video():
    """Add a video with optional bbox CSV."""
    try:
        data = request.get_json()
        video_id = data.get("video_id")
        video_path = data.get("video_path")
        csv_path = data.get("csv_path")
        remote_path = data.get("remote_path")  # Full remote path for display

        if not all([video_id, video_path]):
            return jsonify({"error": "Missing required fields"}), 400

        # Process bbox CSV if provided
        bbox_data = {}
        if csv_path and Path(csv_path).exists():
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                frame = int(row["frame"])
                if frame not in bbox_data:
                    bbox_data[frame] = []
                bbox_data[frame].append(
                    {
                        "track_id": int(row.get("track_id", 0)),
                        "x1": float(row.get("x1", 0)),
                        "y1": float(row.get("y1", 0)),
                        "x2": float(row.get("x2", 0)),
                        "y2": float(row.get("y2", 0)),
                    }
                )

        with data_lock:
            videos_data[video_id] = {
                "name": remote_path if remote_path else Path(video_path).name,
                "video_path": video_path,
                "csv_path": csv_path,
                "remote_path": remote_path,
                "bbox_data": bbox_data,
                "has_bboxes": bool(bbox_data),
            }

        return jsonify({"success": True, "video_id": video_id})

    except Exception as e:
        logger.error(f"Error adding video: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/add_mesh", methods=["POST"])
def api_add_mesh():
    """Add a mesh with 3D tracking CSV."""
    try:
        data = request.get_json()
        mesh_id = data.get("mesh_id")
        mesh_path = data.get("mesh_path")
        csv_3d_path = data.get("csv_3d_path")
        camera_params_path = data.get("camera_params_path")
        remote_path = data.get("remote_path")  # Full remote path for display

        if not all([mesh_id, mesh_path, csv_3d_path]):
            return jsonify({"error": "Missing required fields"}), 400

        # Process 3D tracking CSV
        tracks_by_frame = {}
        df = None
        if Path(csv_3d_path).exists():
            df = pd.read_csv(csv_3d_path)
            for _, row in df.iterrows():
                frame = int(row["frame"])
                if frame not in tracks_by_frame:
                    tracks_by_frame[frame] = []
                tracks_by_frame[frame].append(
                    {
                        "track_id": int(row["track_id"]),
                        "x": float(row["x"]) if pd.notna(row["x"]) else None,
                        "y": float(row["y"]) if pd.notna(row["y"]) else None,
                        "z": float(row["z"]) if pd.notna(row["z"]) else None,
                    }
                )

        # Load camera parameters if provided
        camera_params = None
        if camera_params_path and Path(camera_params_path).exists():
            import pickle

            try:
                with open(camera_params_path, "rb") as f:
                    raw_params = pickle.load(f)
                camera_params = convert_camera_params_to_json(raw_params)
            except Exception as e:
                logger.error(f"Error loading camera params: {e}")

        with data_lock:
            meshes_data[mesh_id] = {
                "name": remote_path if remote_path else Path(mesh_path).name,
                "mesh_path": mesh_path,
                "csv_3d_path": csv_3d_path,
                "remote_path": remote_path,
                "tracks_by_frame": tracks_by_frame,
                "camera_params": camera_params,
                "num_frames": len(tracks_by_frame),
                "num_tracks": len(df["track_id"].unique())
                if df is not None and "track_id" in df.columns
                else 0,
            }

        return jsonify({"success": True, "mesh_id": mesh_id})

    except Exception as e:
        logger.error(f"Error adding mesh: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/video/<video_id>")
def api_get_video_data(video_id):
    """Get video data including bbox tracks."""
    with data_lock:
        if video_id not in videos_data:
            return jsonify({"error": "Video not found"}), 404

        data = videos_data[video_id]
        return jsonify(
            {
                "name": data["name"],
                "video_path": data["video_path"],
                "has_bboxes": data["has_bboxes"],
                "bbox_data": data.get("bbox_data", {}),
            }
        )


@app.route("/api/mesh/<mesh_id>")
def api_get_mesh_data(mesh_id):
    """Get mesh data including 3D tracks and camera params."""
    with data_lock:
        if mesh_id not in meshes_data:
            return jsonify({"error": "Mesh not found"}), 404

        data = meshes_data[mesh_id]
        return jsonify(
            {
                "name": data["name"],
                "mesh_path": data["mesh_path"],
                "num_frames": data["num_frames"],
                "num_tracks": data["num_tracks"],
                "tracks_by_frame": data["tracks_by_frame"],
                "camera_params": data["camera_params"],
            }
        )


@app.route("/api/mesh/<mesh_id>/file")
def api_get_mesh_file(mesh_id):
    """Serve the actual mesh file."""
    with data_lock:
        if mesh_id not in meshes_data:
            return jsonify({"error": "Mesh not found"}), 404

        mesh_path = meshes_data[mesh_id]["mesh_path"]
        if not Path(mesh_path).exists():
            return jsonify({"error": "Mesh file not found"}), 404

        directory = Path(mesh_path).parent
        filename = Path(mesh_path).name
        return send_from_directory(directory, filename)


@app.route("/api/video/<video_id>/file")
def api_get_video_file(video_id):
    """Serve the actual video file."""
    with data_lock:
        if video_id not in videos_data:
            return jsonify({"error": "Video not found"}), 404

        video_path = videos_data[video_id]["video_path"]
        if not Path(video_path).exists():
            return jsonify({"error": "Video file not found"}), 404

        directory = Path(video_path).parent
        filename = Path(video_path).name
        return send_from_directory(directory, filename)


@app.route("/api/clear", methods=["POST"])
def api_clear():
    """Clear all videos and meshes."""
    with data_lock:
        num_videos = len(videos_data)
        num_meshes = len(meshes_data)
        videos_data.clear()
        meshes_data.clear()

    return jsonify(
        {
            "success": True,
            "cleared_videos": num_videos,
            "cleared_meshes": num_meshes,
            "total_cleared": num_videos + num_meshes,
        }
    )


@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check endpoint with current server status."""
    with data_lock:
        videos_count = len(videos_data)
        meshes_count = len(meshes_data)

    return jsonify(
        {
            "status": "healthy",
            "videos_count": videos_count,
            "meshes_count": meshes_count,
            "server": "persistent_video_server",
        }
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Persistent Video Server")
    parser.add_argument("--port", type=int, default=5050, help="Port to run server on")
    args = parser.parse_args()

    port = args.port
    logger.info(f"Starting persistent server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
