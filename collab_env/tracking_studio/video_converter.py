"""
Video Format Converter Component

Ensures videos are in H.264 format for browser compatibility.
"""

import subprocess
from pathlib import Path
from loguru import logger


def needs_conversion(video_path: Path) -> bool:
    """
    Check if video needs H.264 conversion.

    Args:
        video_path: Path to video file

    Returns:
        True if conversion needed, False otherwise
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        codec = result.stdout.strip()

        logger.info(f"Video codec: {codec}")
        if codec != "h264":
            return True
        # H.264 in non-MP4 container (e.g. .mov) may not play in all browsers
        ext = Path(video_path).suffix.lower()
        if ext not in (".mp4", ".m4v"):
            logger.info(f"H.264 in {ext} container — will remux to .mp4 for browser compatibility")
            return True
        return False

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to check video codec: {e}")
        # Assume conversion needed if check fails
        return True
    except FileNotFoundError:
        logger.error("ffprobe not found. Please install ffmpeg.")
        raise


def convert_to_h264(input_path: Path, output_path: Path, remux_only: bool = False) -> Path:
    """
    Convert video to H.264 format using ffmpeg.

    Args:
        input_path: Original video file
        output_path: Output path for converted video
        remux_only: If True, copy streams without re-encoding (fast container change)

    Returns:
        Path to converted video
    """
    try:
        if remux_only:
            logger.info(f"Remuxing {input_path} to MP4 container (no re-encoding)")
            cmd = [
                "ffmpeg",
                "-i", str(input_path),
                "-c", "copy",  # Copy all streams without re-encoding
                "-movflags", "+faststart",
                "-y",
                str(output_path),
            ]
        else:
            logger.info(f"Converting {input_path} to H.264 format")
            cmd = [
                "ffmpeg",
                "-i", str(input_path),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                "-y",
                str(output_path),
            ]

        subprocess.run(cmd, check=True, capture_output=True)

        logger.info(f"Successfully converted video to {output_path}")
        return output_path

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to convert video: {e}")
        logger.error(f"stderr: {e.stderr.decode() if e.stderr else 'N/A'}")
        raise
    except FileNotFoundError:
        logger.error("ffmpeg not found. Please install ffmpeg.")
        raise
