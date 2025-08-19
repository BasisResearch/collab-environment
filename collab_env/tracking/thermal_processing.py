"""
Script: thermal_processing.py
Description:
    This script provides utilities for processing FLIR .csq thermal video files. It includes functionality for:
    - Extracting thermal frames and converting them to temperature values.
    - Rendering frames with colorbars.
    - Exporting thermal frames to MP4 videos with a fixed colormap.
    - Processing directories of .csq files.

Usage:
    Use the provided functions to process .csq files and export them as videos or analyze their content.

Example:
    process_directory("data/thermal", "output/videos", color="hot", preview=True, max_frames=100)
"""

import os
import re
import subprocess
import tempfile
from io import BytesIO
from math import exp, sqrt
from pathlib import Path
from typing import List, Optional, Tuple
import argparse

import cv2
import exiftool
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from libjpeg.utils import decode
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from tqdm import tqdm

# Constants
MAGIC_SEQ = re.compile(b"\x46\x46\x46\x00\x52\x54")
EXIFTOOL_PATH = subprocess.check_output(["which", "exiftool"]).decode("utf-8").strip()


class CSQReader:
    """
    Reader for FLIR .csq thermal video files.
    Extracts thermal image frames using ExifTool and raw temperature conversion.
    """

    def __init__(self, filename, blocksize=1_000_000):
        self.reader = open(filename, "rb")
        self.blocksize = blocksize
        self.leftover = b""
        self.imgs = []
        self.index = 0
        self.nframes = None

        if not os.path.exists(EXIFTOOL_PATH):
            raise FileNotFoundError(f"ExifTool not found at {EXIFTOOL_PATH}")

        self.et = exiftool.ExifTool(executable=EXIFTOOL_PATH)
        self.etHelper = exiftool.ExifToolHelper(executable=EXIFTOOL_PATH)
        self.et.run()

    def _populate_list(self):
        """Populate the list of frames from the .csq file."""
        self.imgs = []
        self.index = 0
        x = self.reader.read(self.blocksize)
        if not x:
            return
        matches = list(MAGIC_SEQ.finditer(x))
        if not matches:
            return
        start = matches[0].start()
        if self.leftover:
            self.imgs.append(self.leftover + x[:start])
        if len(matches) < 2:
            return
        for m1, m2 in zip(matches, matches[1:]):
            self.imgs.append(x[m1.start() : m2.start()])
        self.leftover = x[matches[-1].start() :]

    def next_frame(self):
        """Retrieve the next thermal frame as a numpy array."""
        if self.index >= len(self.imgs):
            self._populate_list()
            if not self.imgs:
                return None
        im = self.imgs[self.index]
        raw, metadata = extract_data(im, self.etHelper)
        thermal_im = raw2temp(raw, metadata[0])
        self.index += 1
        return thermal_im

    def skip_frame(self):
        """Skip the current frame."""
        if self.index >= len(self.imgs):
            self._populate_list()
            if not self.imgs:
                return False
        self.index += 1
        return True

    def count_frames(self):
        """Count the total number of frames in the .csq file."""
        self.nframes = 0
        while self.skip_frame():
            self.nframes += 1
        self.reset()
        return self.nframes

    def get_metadata(self):
        """Get metadata of the current frame."""
        if self.index >= len(self.imgs):
            self._populate_list()
            if not self.imgs:
                return None
        im = self.imgs[self.index]
        _, metadata = extract_data(im, self.etHelper)
        return metadata

    def reset(self):
        """Reset the reader to the beginning of the file."""
        self.reader.seek(0)
        self.leftover = b""
        self.imgs = []
        self.index = 0

    def close(self):
        """Close the file reader."""
        self.reader.close()


def extract_data(bin_data, etHelper):
    """Extract raw thermal data and metadata from a binary frame."""
    with tempfile.NamedTemporaryFile() as fp:
        fp.write(bin_data)
        fp.flush()
        metadata = etHelper.get_metadata(fp.name)
        binary = subprocess.check_output(
            [EXIFTOOL_PATH, "-b", "-RawThermalImage", fp.name]
        )
        raw = decode(binary)
    return raw, metadata


def raw2temp(raw, metadata):
    """Convert raw thermal data to temperature values using metadata."""
    E = metadata["FLIR:Emissivity"]
    OD = metadata["FLIR:ObjectDistance"]
    RTemp = metadata["FLIR:ReflectedApparentTemperature"]
    ATemp = metadata["FLIR:AtmosphericTemperature"]
    IRWTemp = metadata["FLIR:IRWindowTemperature"]
    IRT = metadata["FLIR:IRWindowTransmission"]
    RH = metadata["FLIR:RelativeHumidity"]
    PR1 = metadata["FLIR:PlanckR1"]
    PB = metadata["FLIR:PlanckB"]
    PF = metadata["FLIR:PlanckF"]
    PO = metadata["FLIR:PlanckO"]
    PR2 = metadata["FLIR:PlanckR2"]
    ATA1 = float(metadata["FLIR:AtmosphericTransAlpha1"])
    ATA2 = float(metadata["FLIR:AtmosphericTransAlpha2"])
    ATB1 = float(metadata["FLIR:AtmosphericTransBeta1"])
    ATB2 = float(metadata["FLIR:AtmosphericTransBeta2"])
    ATX = metadata["FLIR:AtmosphericTransX"]

    emiss_wind = 1 - IRT
    refl_wind = 0
    h2o = (RH / 100) * exp(
        1.5587 + 0.06939 * ATemp - 0.00027816 * ATemp**2 + 0.00000068455 * ATemp**3
    )

    def tau(d):
        return ATX * exp(-sqrt(d) * (ATA1 + ATB1 * sqrt(h2o))) + (1 - ATX) * exp(
            -sqrt(d) * (ATA2 + ATB2 * sqrt(h2o))
        )

    tau1 = tau(OD / 2)
    tau2 = tau(OD / 2)

    def radiance(T):
        return PR1 / (PR2 * (exp(PB / (T + 273.15)) - PF)) - PO

    raw_obj = (
        raw / E / tau1 / IRT / tau2
        - (1 - tau1) / E / tau1 * radiance(ATemp)
        - (1 - tau2) / E / tau1 / IRT / tau2 * radiance(ATemp)
        - emiss_wind / E / tau1 / IRT * radiance(IRWTemp)
        - (1 - E) / E * radiance(RTemp)
        - refl_wind / E / tau1 / IRT * radiance(RTemp)
    )
    temp_C = PB / np.log(PR1 / (PR2 * (raw_obj + PO)) + PF) - 273.15

    return temp_C


def process_frame(frame, vmin, vmax):
    frame = np.clip(frame, vmin, vmax)
    return np.uint8((frame - vmin) / (vmax - vmin) * 255)


def choose_vmin_vmax(path, default_vmin=-5, default_vmax=37):
    frame_collection = []

    path = Path(path)
    if path.is_file() and path.suffix == ".csq":
        reader = CSQReader(str(path))
        frame = reader.next_frame()
        if frame is not None:
            frame_collection.append(frame.flatten())
    else:
        for folder in os.listdir(path):
            if folder.startswith("FLIR"):
                full_path = os.path.join(path, folder)
                for f_name in os.listdir(full_path):
                    if f_name.endswith(".csq"):
                        reader = CSQReader(os.path.join(full_path, f_name))
                        frame = reader.next_frame()
                        if frame is not None:
                            frame_collection.append(frame.flatten())

    if not frame_collection:
        return default_vmin, default_vmax

    all_pixels = np.concatenate(frame_collection)
    vmin = max(default_vmin, np.round(np.percentile(all_pixels, 0.1)))
    vmax = min(default_vmax, np.round(np.percentile(all_pixels, 99.999)))
    return vmin, vmax


def estimate_duration(reader: CSQReader, fps: float = 30.0) -> float:
    """
    Estimates duration in seconds by counting frames and dividing by fps.
    """
    print("⏱ Counting frames for duration estimate...")
    total_frames = reader.count_frames()
    duration = total_frames / fps
    print(f"→ {total_frames} frames ≈ {duration:.2f} seconds at {fps:.1f} fps")
    reader.reset()
    return duration


def render_frame_with_colorbar(
    frame: np.ndarray,
    vmin: float,
    vmax: float,
    color: str,
    figsize: Tuple[int, int] = (5, 5),
) -> np.ndarray:
    """Render a thermal frame with a colorbar and return it as an RGB image."""
    if frame is None:
        raise ValueError(
            "Frame is None. Cannot render colorbar."
        )  # Fix: Handle None frame

    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=figsize)
    plt.axis("off")
    ax.imshow(frame, cmap=color, vmin=vmin, vmax=vmax)
    make_axes_locatable(ax)
    fig.tight_layout(pad=0)

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img.convert("RGB"))
    plt.close(fig)
    return img_array


def export_thermal_video(
    reader: CSQReader,
    out_path: Path,
    vmin: float,
    vmax: float,
    color: str,
    max_frames: Optional[int] = None,
    fps: int = 30,
):
    """Convert thermal frames into an MP4 video with a fixed colormap."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Always reset before reading frames
    reader.reset()

    # Determine total frames to process
    if max_frames is None or max_frames == 0:
        total_frames = reader.count_frames()
        print(f"🖼️ Total frames in video: {total_frames}")
        reader.reset()
        if total_frames == 0:
            print("⚠️ No frames found in CSQ file.")
            return
        frames_to_process = total_frames
    else:
        frames_to_process = max_frames

    frame_iterator = tqdm(
        range(frames_to_process),
        desc="🔄 Writing frames",
        unit="frame",
        total=frames_to_process,
    )

    frame_count = 0
    out = None
    for _ in frame_iterator:
        frame = reader.next_frame()
        if frame is None:
            break
        img = render_frame_with_colorbar(frame, vmin, vmax, color)
        height, width, _ = img.shape
        if out is None:
            out = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
                fps,
                (width, height),
            )
        out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        frame_count += 1

    if out is not None and frame_count > 0:
        out.release()
        duration_sec = frame_count / fps
        print(f"\n✅ Exported {frame_count} frames to: {out_path}")
        print(f"⏱ Approx. duration: {duration_sec:.2f} seconds at {fps} fps")
    else:
        print("⚠️ No frames written to video.")


def process_directory(
    folder_path, out_path, color="binary", preview=True, max_frames=None, fps=30
):
    """Process the thermal_1 and thermal_2 subfolders in the given folder_path and export their .csq files as MP4 videos."""
    folder_path = Path(folder_path)
    thermal_folders = ["thermal_1", "thermal_2"]

    for thermal_folder in thermal_folders:
        thermal_path = folder_path / thermal_folder
        if not thermal_path.exists():
            print(
                f"⚠️ Folder {thermal_folder} does not exist in {folder_path}. Skipping..."
            )
            continue

        csq_files = sorted(thermal_path.glob("*.csq"))
        print(f"Found {len(csq_files)} .csq files in {thermal_path}")

        for idx, file_path in enumerate(csq_files):
            print(f"\nProcessing [{idx}] in {thermal_folder}: {file_path.name}")

            reader = CSQReader(str(file_path))

            if preview:
                frame = reader.next_frame()
                if frame is None:
                    print(f"⚠️ No frames found in {file_path}")
                    continue

                sns.set_style("ticks")
                fig, ax = plt.subplots()
                plt.axis("off")
                im = ax.imshow(frame, cmap=color)
                plt.colorbar(im, ax=ax)
                plt.title(f"Preview for {file_path.name}")
                plt.show()

                vmin = float(input("Enter vmin: "))
                vmax = float(input("Enter vmax: "))
            else:
                print("Auto-detecting vmin/vmax...")
                vmin, vmax = choose_vmin_vmax(file_path)
                print(f"→ Using vmin={vmin}, vmax={vmax}")

            reader.reset()
            out_file = (
                Path(out_path) / thermal_folder / f"thermal_{int(vmin)}_{int(vmax)}.mp4"
            )  # Ensure output is organized by thermal folder
            out_file.parent.mkdir(
                parents=True, exist_ok=True
            )  # Create the output directory if it doesn't exist
            export_thermal_video(
                reader,
                out_file,
                vmin,
                vmax,
                color=color,
                max_frames=max_frames or 0,
                fps=fps,
            )  # Handle None for max_frames


def validate_session_structure(session_path: Path) -> List[str]:
    """Validate that session has expected structure"""
    issues = []

    # Check for required thermal directories
    for thermal_dir in ["thermal_1", "thermal_2"]:
        thermal_path = session_path / thermal_dir
        if not thermal_path.exists():
            issues.append(f"Missing {thermal_dir} directory")
            continue

        # Check for CSQ files
        csq_files = list(thermal_path.glob("*.csq"))
        if not csq_files:
            issues.append(f"No CSQ files found in {thermal_dir}")

    # Check for RGB directories
    for rgb_dir in ["rgb_1", "rgb_2"]:
        rgb_path = session_path / rgb_dir
        if not rgb_path.exists():
            issues.append(f"Missing {rgb_dir} directory")
            continue

        # Check for MP4 files
        mp4_files = list(rgb_path.glob("*.MP4"))
        if not mp4_files:
            issues.append(f"No MP4 files found in {rgb_dir}")
    if not issues:
        print("Session structure is valid.")
    else:
        print("Session structure issues found:")
        for issue in issues:
            print(f"- {issue}")
    return issues


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder_path",
        type=str,
        required=True,
        help="Path to the folder containing thermal data",
    )
    parser.add_argument(
        "--out_path", type=str, required=True, help="Path to save the processed videos"
    )
    parser.add_argument(
        "--color",
        type=str,
        default="hot",
        help="Colormap to use for thermal video rendering (default: 'hot')",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Minimum temperature value for colormap scaling (optional)",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Maximum temperature value for colormap scaling (optional)",
    )
    parser.add_argument(
        "--preview",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Preview the first frame and allow user to set vmin/vmax (True/False)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: None, process all frames)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for the output video (default: 30)",
    )
    args = parser.parse_args()

    process_directory(
        args.folder_path,
        args.out_path,
        color=args.color,
        preview=args.preview,
        max_frames=args.max_frames,
        fps=args.fps,
    )
