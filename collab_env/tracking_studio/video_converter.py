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
        return codec != "h264"

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to check video codec: {e}")
        # Assume conversion needed if check fails
        return True
    except FileNotFoundError:
        logger.error("ffprobe not found. Please install ffmpeg.")
        raise


def convert_to_h264(input_path: Path, output_path: Path) -> Path:
    """
    Convert video to H.264 format using ffmpeg.

    Args:
        input_path: Original video file
        output_path: Output path for converted video

    Returns:
        Path to converted video
    """
    try:
        logger.info(f"Converting {input_path} to H.264 format")

        cmd = [
            "ffmpeg",
            "-i",
            str(input_path),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",  # Web optimization
            "-y",  # Overwrite output
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
