from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def export_tracks_with_masks(
    tracked_bboxes_csv: Path, mask_dir: Path, output_csv: Path
):
    """
    Export a CSV with track info and corresponding binary mask (flattened) for each track and frame.

    Args:
        tracked_bboxes_csv (Path): CSV with columns ['track_id', 'frame', 'x1', 'y1', 'x2', 'y2'].
        mask_dir (Path): Directory containing mask images named as frame_XXXXXX.jpg.
        output_csv (Path): Path to save the output CSV.
    """
    df = pd.read_csv(tracked_bboxes_csv)
    rows = []
    for _, row in df.iterrows():
        frame_idx = int(row["frame"])
        mask_path = mask_dir / f"frame_{frame_idx:06d}.jpg"
        if not mask_path.exists():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        mask_crop = mask[y1:y2, x1:x2]
        mask_flat = mask_crop.flatten()
        mask_str = ",".join(map(str, mask_flat))
        rows.append(
            {
                "track_id": int(row["track_id"]),
                "frame": frame_idx,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "mask": mask_str,
            }
        )
    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)
    print(f"✅ Exported tracks with masks to: {output_csv}")


# visualization!


def overlay_tracks_on_video(
    csv_path: Path, frame_dir: Path, output_video: Path, fps: int = 30
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
            tid, x, y = int(row["track_id"]), int(row["x"]), int(row["y"])
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            cv2.circle(frame, (x, y), 5, (255, 255, 0), -1)
            cv2.putText(
                frame, f"ID {tid}", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        writer.write(frame)

    writer.release()
    print(f"✅ Saved annotated video to: {output_video}")


def plot_tracks(
    csv_path: Path,
    output_figure_path: Path,
    title: str = "Object Tracks Over Time",
    frame_range: tuple = (0, 100),  # Specify a range of frames (start, end)
    specific_frames: Optional[
        list[int]
    ] = None,  # Specify a list of specific frames to include
):
    """
    Plot object tracks across time, showing trails for each tracked object.

    Args:
        csv_path (Path): Path to the CSV file containing tracking data.
        output_figure_path (Path): Path to save the output figure.
        title (str): Title of the plot.
        frame_range (tuple, optional): Range of frames to include (start, end).
        specific_frames (list, optional): List of specific frames to include.
    """
    # Load tracking data
    df = pd.read_csv(csv_path)

    # Ensure the required columns exist
    if not {"track_id", "frame", "x", "y"}.issubset(df.columns):
        raise ValueError(
            "CSV file must contain 'track_id', 'frame', 'x', and 'y' columns."
        )

    # Filter by frame range or specific frames
    if frame_range:
        start, end = frame_range
        df = df[(df["frame"] >= start) & (df["frame"] <= end)]
    elif specific_frames:
        df = df[df["frame"].isin(specific_frames)]

    # Create a figure
    plt.figure(figsize=(10, 8))
    plt.title(title, fontsize=16)
    plt.xlabel("X Coordinate", fontsize=14)
    plt.ylabel("Y Coordinate", fontsize=14)

    # Plot each track
    # only plot points where the class is bird?

    # only plot tracks that have more than 1 point
    df = df.groupby("track_id").filter(lambda x: len(x) > 10)
    if df.empty:
        plt.text(0.5, 0.5, "No valid tracks to display", fontsize=12, ha="center")
        return
    track_ids = df["track_id"].unique()
    for track_id in track_ids:
        track_data = df[df["track_id"] == track_id]
        plt.plot(
            track_data["x"],
            track_data["y"],
            marker="o",
            label=f"Track {track_id}",
            alpha=0.7,
        )

    # Add legend and grid
    # plt.legend(title="Track ID", fontsize=10, loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)

    # Invert the y-axis to match image coordinates
    plt.gca().invert_yaxis()

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_figure_path, dpi=300)
    print(f"✅ Figure saved to: {output_figure_path}")

    # Show the plot
    plt.show()


# # Example usage
# csv_file = Path(f"{TARGET_ROOT_DIR}/processed_data/{SESSION_ROOT_DIR}/aligned_frames/rgb_2/rgb_2_tracks.csv")
# output_figure = Path(f"{TARGET_ROOT_DIR}/processed_data/{SESSION_ROOT_DIR}/tracks_plot.png")

# # Plot with a frame range
# plot_tracks(csv_file, output_figure, title="Bird Tracks Over Time", frame_range=(100, 500))

# Plot with specific frames
# plot_tracks(csv_file, output_figure, title="Bird Tracks Over Time", specific_frames=[10, 20, 30, 40])
