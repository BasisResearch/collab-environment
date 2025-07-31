import os
import cv2
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Optional
from pathlib import Path
from ultralytics.trackers.byte_tracker import BYTETracker
import argparse


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

    # Initialize ByteTracker
    tracker = BYTETracker()
    track_history: dict[str, list[tuple[int, tuple[int, int]]]] = {}

    # Perform tracking
    for frame_idx, detections in enumerate(all_frames):
        if not detections:
            continue
        dets_np = np.array(detections)
        if dets_np.ndim != 2 or dets_np.shape[1] < 5:
            continue
        class_ids = np.zeros((dets_np.shape[0], 1))  # Add dummy class IDs
        formatted_dets = np.hstack((dets_np[:, :5], class_ids))
        tracked = tracker.update(formatted_dets)

        for det in tracked:
            x1, y1, x2, y2, track_id, _ = det[:6]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append((frame_idx, (cx, cy)))

    return track_history


def visualize_detections_from_video(
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


def overlay_tracks_on_video(
    csv_path: Path, frame_dir: Path, output_video: Path, fps: int = 5
):
    df = pd.read_csv(csv_path)
    df["track_id"] = df["track_id"].astype(int)

    frame_paths = sorted(frame_dir.glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {frame_dir}")

    # Get frame size
    sample_frame = cv2.imread(str(frame_paths[0]))
    h, w = sample_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
        fps,
        (w, h),
    )

    for frame_path in frame_paths:
        frame_idx = int(frame_path.stem.split("_")[-1])
        frame = cv2.imread(str(frame_path))

        frame_tracks = df[df["frame"] == frame_idx]
        for _, row in frame_tracks.iterrows():
            _, x, y = int(row["track_id"]), int(row["x"]), int(row["y"])
            cv2.circle(frame, (x, y), 5, (255, 255, 0), -1)

        writer.write(frame)

    writer.release()
    print(f"✅ Saved annotated video to: {output_video}")


def run_tracking(
    session_path: Path, camera_type: str = "thermal", camera_number: int = 1
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
        frame_dir = session_path / f"thermal_{camera_number}" / "annotated_frames"
        csv_pattern = "*.csv"
        subfolder_path = session_path / f"thermal_{camera_number}"
    elif camera_type == "rgb":
        subfolder = f"rgb_{camera_number}"
        print("using detections from thermal camera")
        frame_dir = session_path / f"thermal_{camera_number}" / "cropped" / "frames"
        csv_pattern = "*.csv"
        subfolder_path = session_path / subfolder
    else:
        print("Invalid camera_type. Use 'thermal' or 'rgb'.")
        return

    if not subfolder_path.exists():
        print(f"⚠️ Subfolder not found: {subfolder_path}")
        return

    try:
        csv_file = next(subfolder_path.glob(csv_pattern))
    except StopIteration:
        print(f"⚠️ No CSV found in {subfolder_path}")
        return

    if not frame_dir.exists():
        print(f"⚠️ Missing frames at: {frame_dir}")
        return

    # Run tracking
    track_history = track_objects(csv_file)

    # Save tracks
    output_csv = subfolder_path / f"{camera_type}_{camera_number}_tracks.csv"
    with open(output_csv, "w") as f:
        f.write("track_id,frame,x,y\n")
        for tid, points in track_history.items():
            for frame_idx, (x, y) in points:
                f.write(f"{tid},{frame_idx},{x},{y}\n")
    print(f"✅ Tracking results saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Run object tracking on thermal or RGB videos."
    )
    parser.add_argument(
        "--target_root_dir",
        type=str,
        required=True,
        help="Path to the root directory containing data.",
    )
    parser.add_argument(
        "--session_root_dir",
        type=str,
        required=True,
        help="Session directory name (e.g., '2024_02_06-session_0001').",
    )
    parser.add_argument(
        "--camera_type",
        type=str,
        choices=["thermal", "rgb"],
        required=True,
        help="Type of camera ('thermal' or 'rgb').",
    )
    parser.add_argument(
        "--camera_number",
        type=int,
        choices=[1, 2],
        required=True,
        help="Camera number (1 or 2).",
    )
    args = parser.parse_args()

    # Parse arguments
    target_root_dir = Path(args.target_root_dir)
    session_root_dir = args.session_root_dir
    camera_type = args.camera_type
    camera_number = args.camera_number

    # Construct session path
    session_path = target_root_dir / session_root_dir / "aligned_videos"

    # Run tracking
    run_tracking(session_path, camera_type=camera_type, camera_number=camera_number)


if __name__ == "__main__":
    main()
