import subprocess
import os
import sys
import csv
import json


def get_creation_time(filepath):
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                filepath,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        metadata = json.loads(result.stdout)
        format_tags = metadata.get("format", {}).get("tags", {})
        creation = format_tags.get("creation_time", "N/A")
        timecode = format_tags.get("timecode")

        # Fallback: check stream-level timecode if not in format tags
        if not timecode:
            streams = metadata.get("streams", [])
            for stream in streams:
                stream_tags = stream.get("tags", {})
                if "timecode" in stream_tags:
                    timecode = stream_tags["timecode"]
                    break

        if not timecode:
            timecode = "N/A"

        return creation, timecode

    except Exception as e:
        return f"Error: {e}", "N/A"


def extract_timestamps_from_folder(folder_path, out_path):
    """
    Scans a folder for video files and extracts their creation time and timecode.
    Results are saved to a CSV in a different folder.
    """
    if not os.path.isdir(folder_path):
        print(f"❌ Provided path is not a folder: {folder_path}")
        return

    video_extensions = {".mp4", ".mov", ".avi", ".mkv"}
    video_files = [
        f
        for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in video_extensions
    ]

    if not video_files:
        print("🚫 No video files found in the folder.")
        return

    print(f"\n📂 Scanning folder: {folder_path}")
    output = []
    for file in sorted(video_files):
        full_path = os.path.join(folder_path, file)
        print(f"📼 {file} ...", end=" ")
        creation_time, timecode = get_creation_time(full_path)
        print(f"Created: {creation_time}, Timecode: {timecode}")
        output.append((file, creation_time, timecode))

    # Save to CSV
    csv_path = os.path.join(out_path, "video_timestamps.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "CreationTime", "Timecode"])
        writer.writerows(output)

    print(f"\n✅ Timestamps saved to: {csv_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python video_timestamps.py /path/to/video_folder /path/to/output_folder"
        )
    else:
        extract_timestamps_from_folder(sys.argv[1], sys.argv[2])
