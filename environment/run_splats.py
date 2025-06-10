import os
import argparse
import subprocess
import tempfile
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple
import cv2

# File type configurations
FILE_TYPES = {
    "video": [".mp4"],
    "image": [".jpg", ".png"],
    "equirect": [".360"]
}

# Equirect configurations
EQUIRECT_CONFIG = {
    "images_per_equirect": 8,  # can be 8 or 14
    "crop_factors": {
        "top": 0,
        "bottom": 0,
        "left": 0,
        "right": 0
    }
}

class SplatProcessor:
    """Handles the processing of video and image files for neural splats."""
    
    @staticmethod
    def get_files_by_extensions(path: Path, extensions: List[str]) -> List[Path]:
        """
        Get files with given extensions, case insensitive.
        
        Args:
            path: Directory path to search in
            extensions: List of file extensions to look for
        
        Returns:
            List of Path objects for matching files
        """
        files = []
        for ext in extensions:
            files.extend(path.glob(f"*{ext}"))
            files.extend(path.glob(f"*{ext.upper()}"))
        return sorted(files)  # Sort for consistent ordering

    @staticmethod
    def get_output_path(data_dir: str, directory: str, output_dir: str = None) -> str:
        """
        Determine the output path based on input parameters.
        
        Args:
            data_dir: Base data directory
            directory: Current processing directory
            output_dir: Optional custom output directory
        
        Returns:
            String path for output
        """
        if output_dir is None:
            return os.path.join(data_dir, directory, "NS_SPLATS")
        return output_dir

    @staticmethod
    def collect_files(input_path: Path) -> List[Tuple[Path, str]]:
        """
        Collect all files to process with their types.
        
        Args:
            input_path: Path to search for files
        
        Returns:
            List of tuples containing (file_path, file_type)
        """
        all_files = []
        for file_type, extensions in FILE_TYPES.items():
            files = SplatProcessor.get_files_by_extensions(input_path, extensions)
            all_files.extend((f, file_type) for f in files)
        return all_files

    @staticmethod
    def concatenate_videos(video_files: List[Path], output_dir: Path) -> Path:
        """
        Concatenate multiple video files into a single video using ffmpeg.
        
        Args:
            video_files: List of video file paths
            output_dir: Directory to save the concatenated video
        
        Returns:
            Path to the concatenated video file
        """
        if len(video_files) == 1:
            return video_files[0]
        
        # Create a temporary file listing all videos to concatenate
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for video in video_files:
                f.write(f"file '{video.absolute()}'\n")
            concat_list = f.name

        # Output path for concatenated video
        output_video = output_dir / "concatenated_video.mp4"
        
        try:
            # Concatenate videos using ffmpeg --> remove audio
            cmd = (
                f"ffmpeg -y -f concat -safe 0 -i {concat_list} "
                f"-c:v copy -an {output_video}"
            )
            subprocess.run(cmd, shell=True, check=True)
        finally:
            # Clean up temporary file
            os.unlink(concat_list)
            
        return output_video

    @staticmethod
    def build_command(file: Path, file_type: str, output_path: str, frame_proportion: float = 0.33) -> str:
        """
        Build the appropriate command based on file type.
        
        Args:
            file: Path to the input file
            file_type: Type of file (video, image, or equirect)
            output_path: Path for output
        
        Returns:
            Command string to execute
        """
        base_cmd = "ns-process-data"
        
        if file_type == "video":

            video_capture = cv2.VideoCapture(file)
            n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            n_samples = int(n_frames*frame_proportion)
            
            return (
                f"{base_cmd} video "
                f"--num-frames-target {n_samples} "
                f"--data {file} "
                f"--output-dir {output_path}"
            )
        elif file_type == "image":
            return (
                f"{base_cmd} images "
                f"--data {file} "
                f"--output-dir {output_path}"
            )
        elif file_type == "equirect":
            crop_factors = EQUIRECT_CONFIG["crop_factors"]
            crop_str = f"{crop_factors['top']},{crop_factors['bottom']},{crop_factors['left']},{crop_factors['right']}"
            return (
                f"{base_cmd} images "
                f"--camera-type equirectangular "
                f"--images-per-equirect {EQUIRECT_CONFIG['images_per_equirect']} "
                f"--crop-factor {crop_str} "
                f"--data {file} "
                f"--output-dir {output_path}"
            )
        
        raise ValueError(f"Unknown file type: {file_type}")

# Script execution boilerplate
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video and image files for neural splats.")
    parser.add_argument("--data_dir", type=str, required=True, help="Input data directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional custom output directory")
    args = parser.parse_args()

    # Get all directories in data_dir
    directories = [d for d in os.listdir(args.data_dir) 
                  if os.path.isdir(os.path.join(args.data_dir, d))]
    
    # Process each directory
    for directory in tqdm(directories, desc="Processing directories"):
        input_path = Path(os.path.join(args.data_dir, directory, "SplatsSD"))
        output_path = SplatProcessor.get_output_path(args.data_dir, directory, args.output_dir)
        os.makedirs(output_path, exist_ok=True)
        
        # Collect files by type
        print (input_path)
        all_files = SplatProcessor.collect_files(input_path)

        # Group files by type
        files_by_type = {}
        for file, file_type in all_files:
            if file_type not in files_by_type:
                files_by_type[file_type] = []
            files_by_type[file_type].append(file)
        
        # Process videos first - concatenate if multiple videos exist
        if "video" in files_by_type and files_by_type["video"]:
            print(f"Found {len(files_by_type['video'])} videos in {directory}")
            video_file = SplatProcessor.concatenate_videos(
                files_by_type["video"], 
                Path(output_path)
            )
            cmd = SplatProcessor.build_command(video_file, "video", output_path)
            subprocess.run(cmd, shell=True, check=True)
        
        # # Process other file types
        # for file_type in ["image", "equirect"]:
        #     if file_type in files_by_type:
        #         for file in tqdm(files_by_type[file_type], 
        #                        desc=f"Processing {file_type}s in {directory}", 
        #                        leave=False):
        #             cmd = SplatProcessor.build_command(file, file_type, output_path)
        #             subprocess.run(cmd, shell=True, check=True)