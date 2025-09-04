import json
import os
from pathlib import Path
from typing import Optional
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.engine.results import Boxes


def video_to_frames(video_path, out_dir, prefix="frame"):
    """
    Extracts frames from a video and saves them as .jpg.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = 0
    with tqdm(total=total, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            fname = f"{prefix}_{idx:06d}.jpg"
            cv2.imwrite(os.path.join(out_dir, fname), frame)
            idx += 1
            pbar.update(1)
    cap.release()
    print(f"Saved {idx} frames to {out_dir}")


def track_objects(csv_path: Path) -> dict:
    """
    Use ByteTracker to track objects based on detection results.
    Args:
        csv_path: Path to the CSV file containing detection results.
    Returns:
        A dictionary with track IDs and their corresponding frame and coordinates.
    """
    df = pd.read_csv(csv_path)

    # Parse predictions from the CSV
    def safe_json_loads(x):
        try:
            return json.loads(x) if pd.notnull(x) and x.strip() else {"predictions": []}
        except Exception:
            return {"predictions": []}

    df["parsed_predictions"] = df["predictions"].apply(safe_json_loads)
    all_frames = []

    for _, row in df.iterrows():
        detections = []
        for obj in row["parsed_predictions"]["predictions"]:
            x, y, w, h, conf = (
                obj["x"],
                obj["y"],
                obj["width"],
                obj["height"],
                obj.get("confidence", 1.0),
            )
            x1, y1 = x - w / 2, y - h / 2
            x2, y2 = x + w / 2, y + h / 2
            detections.append([x1, y1, x2, y2, conf])
        all_frames.append(detections)

    # Create proper args object for ByteTracker with official defaults
    class ByteTrackerArgs:
        track_high_thresh = 0.25
        track_low_thresh = 0.1
        new_track_thresh = 0.25
        track_buffer = 30
        match_thresh = 0.8
        fuse_score = True

    # Initialize ByteTracker
    tracker = BYTETracker(ByteTrackerArgs(), frame_rate=30)
    track_history: dict[str, list[tuple[int, tuple[int, int]]]] = {}

    # Perform tracking
    for frame_idx, detections in enumerate(all_frames):
        if not detections:
            continue
        dets_np = np.array(detections, dtype=np.float32)
        if dets_np.ndim != 2 or dets_np.shape[1] < 5:
            continue

        # Use Ultralytics Boxes class which is properly subscriptable
        # Boxes expects: x1, y1, x2, y2, conf, class
        # Add dummy class column if not present
        if dets_np.shape[1] == 5:
            dets_np = np.hstack([dets_np, np.zeros((dets_np.shape[0], 1))])

        # Create Boxes object (orig_shape doesn't matter for tracking)
        boxes = Boxes(dets_np, orig_shape=(1080, 1920))

        # Call update with Boxes object
        tracked = tracker.update(boxes, img=None)

        # Process tracked detections
        if tracked.size > 0:
            for det in tracked:
                x1, y1, x2, y2, track_id = det[:5]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                track_id = int(track_id)
                if track_id not in track_history:
                    track_history[str(track_id)] = []
                track_history[str(track_id)].append((frame_idx, (cx, cy)))

    return track_history


def get_detections_from_video(
    csv_path: Path,
    video_path: Path,
    output_video_path: Path,
    fps: Optional[int] = None,
    output_frames_dir: Optional[Path] = None,
):
    """
    Visualize detections from a CSV file on a video and save the annotated video.
    Optionally, save annotated frames to a directory.

    Args:
        csv_path (Path): Path to the CSV file containing detections.
        video_path (Path): Path to the input video.
        output_video_path (Path): Path to save the annotated video.
        fps (int, optional): Frames per second for the output video. If None, use the input video's FPS.
        output_frames_dir (Path, optional): Directory to save annotated frames.
    """

    # Load detections from CSV
    df = pd.read_csv(csv_path)
    df["parsed_predictions"] = df["predictions"].apply(json.loads)

    # Open the video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"⚠️ Unable to open video: {video_path}")
        return

    # Get video properties
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Use input FPS if not provided
    if fps is None:
        fps = int(input_fps)

    # Initialize video writer
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
        fps,
        (width, height),
    )

    # Create output frames directory if needed
    if output_frames_dir is not None:
        output_frames_dir.mkdir(parents=True, exist_ok=True)

    # Process video frame by frame
    for frame_idx in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ Unable to read frame {frame_idx}")
            break

        # Get predictions for the current frame
        if frame_idx < len(df):
            preds = df.iloc[frame_idx]["parsed_predictions"]["predictions"]
            for obj in preds:
                x, y, w_box, h_box = obj["x"], obj["y"], obj["width"], obj["height"]
                conf = obj.get("confidence", None)

                # Calculate bounding box coordinates
                x1 = int(x - w_box / 2)
                y1 = int(y - h_box / 2)
                x2 = int(x + w_box / 2)
                y2 = int(y + h_box / 2)

                # Draw bounding box and label
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = (
                    f"{obj.get('class', 'bird')}: {conf:.2f}"
                    if conf is not None
                    else "bird"
                )
                cv2.putText(
                    frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

        # Write the annotated frame to the video
        writer.write(frame)

        # Optionally save the annotated frame
        if output_frames_dir is not None:
            frame_path = output_frames_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)

    # Release resources
    cap.release()
    writer.release()

    print(f"✅ Annotated video saved to: {output_video_path}")
    if output_frames_dir is not None:
        print(f"✅ Annotated frames saved to: {output_frames_dir}")


def run_tracking(
    session_path: Path,
    camera_type: str = "thermal",
    camera_number: int = 1,
    detections_csv: Optional[Path] = None,
):
    """
    Run tracking on either thermal or RGB videos for a given session.
    Args:
        session_path: Path to session directory (e.g., .../2024_05_19-session_0001)
        camera_type: "thermal" or "rgb"
        camera_number: 1 or 2
    """
    print(
        f"Running tracking for {camera_type} camera {camera_number} in session: {session_path}"
    )
    if camera_type == "thermal":
        subfolder = f"thermal_{camera_number}"
        frame_dir = session_path / f"thermal_{camera_number}"
        if detections_csv is None:
            detections_csv = frame_dir / f"detections_{camera_number}.csv"
        subfolder_path = session_path / subfolder
    elif camera_type == "rgb":
        subfolder = f"rgb_{camera_number}"
        print("using detections from thermal camera")
        frame_dir = session_path / "thermal_{camera_number}"
        if detections_csv is None:
            detections_csv = frame_dir / f"detections_{camera_number}.csv"
        subfolder_path = session_path / subfolder
    else:
        print("Invalid camera_type. Use 'thermal' or 'rgb'.")
        return
    if not detections_csv.exists():
        print(f"⚠️ No detections CSV found at: {detections_csv}")
        return
    if not subfolder_path.exists():
        print(f"⚠️ Subfolder not found: {subfolder_path}")
        return
    if not frame_dir.exists():
        print(f"⚠️ Missing frames at: {frame_dir}")
        return

    # Run tracking
    track_history = track_objects(detections_csv)

    # Save tracks
    output_csv = subfolder_path / f"{camera_type}_{camera_number}_tracks.csv"
    with open(output_csv, "w") as f:
        f.write("track_id,frame,x,y\n")
        for tid, points in track_history.items():
            for frame_idx, (x, y) in points:
                f.write(f"{tid},{frame_idx},{x},{y}\n")
    print(f"✅ Tracking results saved to {output_csv}")


def output_tracked_bboxes_csv(
    track_csv: Path,
    detect_csv: Path,
    output_csv: Path,
    iou_threshold: float = 0.01,
    use_nearest: bool = True,
):
    tracks_df = pd.read_csv(track_csv)
    detects_df = pd.read_csv(detect_csv)
    if "predictions" in detects_df.columns:
        detects_df["parsed_predictions"] = detects_df["predictions"].apply(json.loads)
    else:
        detects_df["parsed_predictions"] = detects_df.apply(
            lambda row: [
                {
                    "x": row["x"],
                    "y": row["y"],
                    "width": row["width"],
                    "height": row["height"],
                    "confidence": row.get("confidence", 1.0),
                    "class": row.get("class", "bird"),
                }
            ],
            axis=1,
        )

    output_rows = []
    for _, track_row in tracks_df.iterrows():
        tid, frame_idx, x, y = (
            int(track_row["track_id"]),
            int(track_row["frame"]),
            int(track_row["x"]),
            int(track_row["y"]),
        )
        if frame_idx >= len(detects_df):
            continue
        det_row = detects_df.iloc[frame_idx]
        preds = (
            det_row["parsed_predictions"]["predictions"]
            if isinstance(det_row["parsed_predictions"], dict)
            else det_row["parsed_predictions"]
        )
        best_iou = 0.0
        best_bbox = None
        best_dist = float("inf")
        best_bbox_dist = None
        for obj in preds:
            x_c, y_c, w, h = obj["x"], obj["y"], obj["width"], obj["height"]
            x1, y1 = int(x_c - w / 2), int(y_c - h / 2)
            x2, y2 = int(x_c + w / 2), int(y_c + h / 2)
            # IoU matching
            track_box = [x - 2, y - 2, x + 2, y + 2]
            det_box = [x1, y1, x2, y2]
            x_center, y_center = int(x_c), int(y_c)
            iou = float(
                max(0, min(track_box[2], det_box[2]) - max(track_box[0], det_box[0]))
                * max(0, min(track_box[3], det_box[3]) - max(track_box[1], det_box[1]))
            )
            iou /= (
                (track_box[2] - track_box[0]) * (track_box[3] - track_box[1])
                + (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
                - iou
                + 1e-6
            )
            if iou > best_iou:
                best_iou = iou
                best_bbox = (
                    x1,
                    y1,
                    x2,
                    y2,
                    obj.get("confidence", 1.0),
                    obj.get("class", "bird"),
                )
            # Nearest center matching
            dist = np.hypot(x - x_center, y - y_center)
            if dist < best_dist:
                best_dist = dist
                best_bbox_dist = (
                    x1,
                    y1,
                    x2,
                    y2,
                    obj.get("confidence", 1.0),
                    obj.get("class", "bird"),
                )
        # Prefer IoU, fallback to nearest if enabled
        if best_bbox and best_iou > iou_threshold:
            output_rows.append(
                {
                    "track_id": tid,
                    "frame": frame_idx,
                    "x1": best_bbox[0],
                    "y1": best_bbox[1],
                    "x2": best_bbox[2],
                    "y2": best_bbox[3],
                    "confidence": best_bbox[4],
                    "class": best_bbox[5],
                }
            )
        elif use_nearest and best_bbox_dist:
            output_rows.append(
                {
                    "track_id": tid,
                    "frame": frame_idx,
                    "x1": best_bbox_dist[0],
                    "y1": best_bbox_dist[1],
                    "x2": best_bbox_dist[2],
                    "y2": best_bbox_dist[3],
                    "confidence": best_bbox_dist[4],
                    "class": best_bbox_dist[5],
                }
            )

    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(output_csv, index=False)
    print(f"✅ Tracked bounding box CSV saved to: {output_csv}")


def generate_thermal_masks_from_bboxes(
    bbox_csv: Path,
    video_path: Path,
    output_mask_dir: Path,
    temp_threshold: int = 128,  # adjust based on your thermal encoding
    mask_value: int = 255,
):
    """
    Generate binary masks for thermal video frames, masking pixels within each bounding box
    that are warmer than temp_threshold.

    Args:
        bbox_csv (Path): CSV file with columns ['frame', 'x1', 'y1', 'x2', 'y2'].
        video_path (Path): Path to the thermal video file.
        output_mask_dir (Path): Directory to save mask images.
        temp_threshold (int): Pixel intensity threshold for "warm" pixels.
        mask_value (int): Value to assign to mask pixels.
    """

    output_mask_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(bbox_csv)
    df["frame"] = df["frame"].astype(int)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ Unable to read frame {frame_idx} from {video_path}")
            continue

        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame

        mask = np.zeros(frame_gray.shape, dtype=np.uint8)

        # Get all bboxes for this frame
        group = df[df["frame"] == frame_idx]
        for _, row in group.iterrows():
            x1, y1, x2, y2 = (
                int(row["x1"]),
                int(row["y1"]),
                int(row["x2"]),
                int(row["y2"]),
            )
            roi = frame_gray[y1:y2, x1:x2]
            roi_mask = (roi > temp_threshold).astype(np.uint8) * mask_value
            mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], roi_mask)

        mask_path = output_mask_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(mask_path), mask)

    cap.release()
    print(f"✅ Thermal masks saved to: {output_mask_dir}")


def plot_tracks_at_frame_bbox_from_video(
    tracked_bboxes_csv: Path,
    video_path: Path,
    output_image: Path,
    frame_number: int = 1000,
    max_frame: int = 1000,
):
    """
    Plot all track traces up to max_frame on top of a given frame extracted from a video.
    Args:
        tracked_bboxes_csv: CSV with columns track_id, frame, x1, y1, x2, y2
        video_path: Path to the video file
        output_image: Path to save the output image
        frame_number: Frame number to extract as background
        max_frame: Only show motion up to this frame
    """
    # Read tracks
    df = pd.read_csv(tracked_bboxes_csv)
    df = df[df["frame"] <= max_frame]
    df["track_id"] = df["track_id"].astype(int)
    df["x_center"] = (df["x1"] + df["x2"]) / 2
    df["y_center"] = (df["y1"] + df["y2"]) / 2

    # Extract frame from video
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, img = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not read frame {frame_number} from {video_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)

    track_ids = df["track_id"].unique()
    colors = plt.cm.get_cmap("tab20", len(track_ids))

    for i, tid in enumerate(track_ids):
        track = df[df["track_id"] == tid].sort_values("frame")
        ax.plot(
            track["x_center"],
            track["y_center"],
            "-",
            color=colors(i),
            linewidth=2,
            alpha=0.8,
        )
        if not track.empty:
            ax.plot(
                track["x_center"].iloc[-1],
                track["y_center"].iloc[-1],
                "o",
                color=colors(i),
                markersize=8,
            )
            ax.text(
                track["x_center"].iloc[-1] + 5,
                track["y_center"].iloc[-1],
                f"ID {tid}",
                color=colors(i),
                fontsize=10,
            )

    ax.set_title(f"Tracks up to frame {max_frame}")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_image, dpi=200)
    plt.close(fig)
    print(f"✅ Track plot saved to: {output_image}")
