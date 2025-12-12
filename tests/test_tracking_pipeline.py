import os
from pathlib import Path

import numpy as np
import pytest

# Import Utility Functions
from collab_env.data.file_utils import expand_path, get_project_root
from collab_env.data.gcs_utils import GCSClient

# Import Custom Scripts
from collab_env.tracking.alignment_gui import align_videos_CI
from collab_env.tracking.model.local_model_inference import infer_with_yolo
from collab_env.tracking.model.local_model_tracking import (
    generate_thermal_masks_from_bboxes,
    get_detections_from_video,
    output_tracked_bboxes_csv,
    run_tracking,
)
from collab_env.tracking.thermal_processing import (
    process_directory,
    validate_session_structure,
)
from collab_env.tracking.visualization import (  # overlay_tracks_on_video, # TODO: this doesn't work
    export_tracks_with_masks,
    plot_tracks_at_frame_bbox_from_video,
)


# Import Environment Variables

# %% [markdown]
# ### Set flags for data processing

# %%
skip_download = False
skip_thermal_extraction = False


# %% [markdown]
# ### Setup gcloud
@pytest.mark.skipif("SKIP_GCS_TESTS" in os.environ, reason="Requires GCS credentials")
def test_tracking_pipeline():
    # %%
    gcs_client = GCSClient()  # use default connection credentials

    # %% [markdown]
    # Check available buckets

    # %%
    # Verify connection
    print("Available buckets:", gcs_client.list_buckets())

    # %% [markdown]
    # Show files within buckets

    # %%
    BUCKET_NAME = "collab-data-test"
    gcs_client.glob(f"{BUCKET_NAME}/*")

    # %% [markdown]
    # Select a session to download and process

    # %%
    # Download Data from Cloud Bucket
    SESSION_FOLDER = "test-session"
    CLOUD_PREFIX = f"{BUCKET_NAME}/{SESSION_FOLDER}"
    gcs_client.glob(f"{CLOUD_PREFIX}/**")
    LOCAL_DOWNLOAD_DIR = expand_path(f"data/raw/{SESSION_FOLDER}", get_project_root())
    LOCAL_PROCESSED_DIR = expand_path(
        f"data/processed/{SESSION_FOLDER}", get_project_root()
    )

    # %% [markdown]
    # Download from gcloud

    # %%
    if not skip_download or not LOCAL_DOWNLOAD_DIR.exists():
        if not LOCAL_DOWNLOAD_DIR.exists():
            LOCAL_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        if not LOCAL_PROCESSED_DIR.exists():
            LOCAL_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        for blob in gcs_client.glob(f"{CLOUD_PREFIX}/**"):
            relative_path = Path(blob).relative_to(f"{CLOUD_PREFIX}")
            local_name = relative_path.name
            suffix = relative_path.suffix
            print(f"local_name: {local_name}, suffix: {suffix}")
            if len(str(suffix)) > 0:
                # print("File!")
                parent_folder = relative_path.parent
                if not Path(LOCAL_DOWNLOAD_DIR / parent_folder).exists():
                    print(f"Creating folder: {LOCAL_DOWNLOAD_DIR / parent_folder}")
                    Path(LOCAL_DOWNLOAD_DIR / parent_folder).mkdir(
                        parents=True, exist_ok=True
                    )
                # print(f"parent_folder: {parent_folder}")
                local_path = LOCAL_DOWNLOAD_DIR / parent_folder / local_name
                print(f"Downloading file: {blob} to {local_path}")
                gcs_client.gcs.get_file(blob, str(local_path))
            else:
                if not Path(LOCAL_PROCESSED_DIR / relative_path).exists():
                    print(f"Creating folder: {LOCAL_PROCESSED_DIR / relative_path}")
                    Path(LOCAL_PROCESSED_DIR / relative_path).mkdir(
                        parents=True, exist_ok=True
                    )
            # check if there is an extension, if not this is a folder and we need to create it

        # print("Downloaded files:", list(LOCAL_DOWNLOAD_DIR.iterdir()))

    # %% [markdown]
    # Ensure everything worked properly

    # %%
    # Validate session structure
    print("Validating session structure...")
    issues = validate_session_structure(LOCAL_DOWNLOAD_DIR)
    print(f"Issues found: {issues if len(issues) > 0 else 'None'}")

    # %% [markdown]
    # ### Extract thermal video info

    # %%
    if not skip_thermal_extraction:
        # thermal files processing
        print("Processing thermal files...")

        # call with preview=False to choose the vmin/vmax automatically, otherwise the user will be asked to choose the vmin/vmax
        # process_directory(folder_path=LOCAL_DOWNLOAD_DIR, out_path=LOCAL_DOWNLOAD_DIR, color='magma', preview=True, max_frames=None, fps=30)

        process_directory(
            folder_path=LOCAL_DOWNLOAD_DIR,
            out_path=LOCAL_PROCESSED_DIR,
            color="magma",
            preview=False,
            max_frames=1000,
            fps=30,
            vmin=3.0,
            vmax=17.0,
        )

    # %%
    # default parameters for alignment to run auto example
    max_frames = 300  # Process some frames for testing
    rotation_angle = 181.0
    camera_numbers = [1]

    # precalculated parameters for example alignment
    roi = (1859, 1251, 1319, 883)  # x, y, width, height
    H = np.array(
        [
            [0.93132, -0.12753, 17.745],
            [0.066574, 0.87069, 1.573],
            [2.6001e-05, -0.00014062, 1.0003],
        ]
    )
    rgb_offset = 27
    thermal_offset = 36

    for camera in camera_numbers:
        print(f"Processing camera {camera}...")

        # Dynamically find the RGB and thermal MP4 files
        rgb_dir = LOCAL_DOWNLOAD_DIR / f"rgb_{camera}"
        if skip_thermal_extraction:
            thermal_dir = LOCAL_DOWNLOAD_DIR / f"thermal_{camera}"
        else:
            thermal_dir = LOCAL_PROCESSED_DIR / f"thermal_{camera}"

        # Find the MP4 file in the RGB directory
        rgb_video_files = list(rgb_dir.glob("*.MP4")) + list(rgb_dir.glob("*.mp4"))
        print("files in rgb_dir:", rgb_video_files)
        if len(rgb_video_files) == 0:
            print(f"No MP4 file found in {rgb_dir}. Skipping camera {camera}.")
            continue
        elif len(rgb_video_files) > 1:
            print(f"Multiple MP4 files found in {rgb_dir}. Using the first one.")
        rgb_video_path = rgb_video_files[0]

        # Find the MP4 file in the thermal directory
        thermal_video_files = list(thermal_dir.glob("*.mp4")) + list(
            thermal_dir.glob("*.MP4")
        )
        print("files in thermal_dir:", thermal_video_files)
        if len(thermal_video_files) == 0:
            print(f"No MP4 file found in {thermal_dir}. Skipping camera {camera}.")
            continue
        elif len(thermal_video_files) > 1:
            print(f"Multiple MP4 files found in {thermal_dir}. Using the first one.")
        thermal_video_path = thermal_video_files[0]

        print(f"RGB video path: {rgb_video_path}")
        print(f"Thermal video path: {thermal_video_path}")

        output_dir_rgb = LOCAL_PROCESSED_DIR / "aligned_frames" / f"rgb_{camera}"
        output_dir_thm = LOCAL_PROCESSED_DIR / "aligned_frames" / f"thermal_{camera}"
        output_dir_rgb.mkdir(parents=True, exist_ok=True)
        output_dir_thm.mkdir(parents=True, exist_ok=True)

        # Align videos
        print(f"Aligning videos for camera {camera}...")

        align_videos_CI(
            rgb_video_path,
            thermal_video_path,
            output_dir_rgb,
            output_dir_thm,
            frame_size=(640, 480),
            max_frames=max_frames,
            warp_to="thermal",
            rotation_angle=rotation_angle,
            skip_homography=False,
            skip_translation=True,
            crop_rect=roi,
            H=H,
            rgb_offset=rgb_offset,
            thermal_offset=thermal_offset,
        )

    # %%
    # download yolo weights
    gcs_client.gcs.get_file(
        "roboflow_model/yolov11_weights.pt", LOCAL_DOWNLOAD_DIR / "yolo11_weights.pt"
    )

    # %%
    # Detection and tracking
    print("Running detection...")
    for camera in camera_numbers:
        print(f"Running detection and tracking on: thermal_{camera}")

        # Define paths for the thermal video and model inference
        thermal_video_path = (
            LOCAL_PROCESSED_DIR
            / "aligned_frames"
            / f"thermal_{camera}"
            / "warped_thermal_adjusted.mp4"
        )
        if not thermal_video_path.exists():
            print(f"Thermal video not found for camera {camera}. Skipping...")
            continue

        rgb_video_path = (
            LOCAL_PROCESSED_DIR
            / "aligned_frames"
            / f"rgb_{camera}"
            / "cropped_rgb_adjusted.mp4"
        )

        # Run local_model_inference script
        try:
            detect_csv = (
                LOCAL_PROCESSED_DIR
                / "aligned_frames"
                / f"thermal_{camera}"
                / f"detections_{camera}.csv"
            )
            checkpoint_path = (
                LOCAL_DOWNLOAD_DIR / "yolo11_weights.pt"
            )  # Update with your model path

            infer_with_yolo(
                video_path=thermal_video_path,
                model_path=checkpoint_path,
                output_csv_path=detect_csv,
                show_window=False,
                verbose=False,
                max_frames=100,
            )
            print(
                f"Object detection completed for camera {camera}. Results saved to {detect_csv}."
            )
        except Exception as e:
            print(f"Error during object detection for camera {camera}: {e}")
            continue

    # %%
    # running tracking
    for camera in camera_numbers:
        # Define paths for the thermal video and model inference
        thermal_video_path = (
            LOCAL_PROCESSED_DIR
            / "aligned_frames"
            / f"thermal_{camera}"
            / "warped_thermal_adjusted.mp4"
        )
        if not thermal_video_path.exists():
            print(f"Thermal video not found for camera {camera}. Skipping...")
            continue

        rgb_video_path = (
            LOCAL_PROCESSED_DIR
            / "aligned_frames"
            / f"rgb_{camera}"
            / "cropped_rgb_adjusted.mp4"
        )
        if not rgb_video_path.exists():
            print(f"RGB video not found for camera {camera}. Skipping...")
            continue

        # Check if the detection CSV exists
        detect_csv = (
            LOCAL_PROCESSED_DIR
            / "aligned_frames"
            / f"thermal_{camera}"
            / f"detections_{camera}.csv"
        )
        if not detect_csv.exists():
            print(f"Detection CSV not found for camera {camera}. Skipping tracking.")
            continue
        # Visualize detections on the thermal and RGB videos
        # visualization
        get_detections_from_video(
            csv_path=detect_csv,
            video_path=thermal_video_path,
            output_video_path=LOCAL_PROCESSED_DIR
            / "aligned_frames"
            / f"thermal_{camera}"
            / f"visualized_thermal_{camera}.mp4",
        )
        # visualization
        get_detections_from_video(
            csv_path=detect_csv,
            video_path=rgb_video_path,
            output_video_path=LOCAL_PROCESSED_DIR
            / "aligned_frames"
            / f"rgb_{camera}"
            / f"visualized_rgb_{camera}.mp4",
        )
        # Run tracking
        print(f"Running tracking on: camera {camera}")
        run_tracking(LOCAL_PROCESSED_DIR / "aligned_frames", "thermal", camera)
        run_tracking(LOCAL_PROCESSED_DIR / "aligned_frames", "rgb", camera)
        tracked_csv = (
            LOCAL_PROCESSED_DIR
            / "aligned_frames"
            / f"thermal_{camera}"
            / f"thermal_{camera}_tracks.csv"
        )
        if not tracked_csv.exists():
            print(
                f"Tracking CSV not found for camera {camera}. Skipping visualization."
            )
            continue
        # Output tracked bounding boxes to CSV
        output_tracked_bboxes_csv(
            track_csv=tracked_csv,
            detect_csv=detect_csv,
            output_csv=Path(
                LOCAL_PROCESSED_DIR
                / f"aligned_frames/rgb_{camera}/tracked_bboxes_{camera}.csv"
            ),
            iou_threshold=0.1,
        )  # Lower threshold if tracks are not centered

        print(f"Visualizing tracks for rgb camera {camera}...")
        # TODO: this doesn't work
        # overlay_tracks_on_video(
        #     csv_path=tracked_csv,
        #     frame_dir=LOCAL_PROCESSED_DIR
        #     / "aligned_frames"
        #     / f"rgb_{camera}"
        #     / "annotated_frames",
        #     output_video=LOCAL_PROCESSED_DIR
        #     / "aligned_frames"
        #     / f"rgb_{camera}"
        #     / f"overlayed_tracks_{camera}.mp4",
        # )

    # %% [markdown]
    # ### Optional: Visualization

    # %%
    frame_number = 100  # Example frame number to visualize
    camera = 1  # Example camera number to visualize

    bboxes_csv = (
        LOCAL_PROCESSED_DIR
        / "aligned_frames"
        / f"rgb_{camera}"
        / f"tracked_bboxes_{camera}.csv"
    )

    print(f"Plotting tracks at frame {frame_number} for camera {camera}...")
    plot_tracks_at_frame_bbox_from_video(
        tracked_bboxes_csv=bboxes_csv,
        video_path=rgb_video_path,
        output_image=LOCAL_PROCESSED_DIR
        / "aligned_frames"
        / f"rgb_{camera}"
        / f"tracked_bboxes_{camera}_{frame_number}.png",
        frame_number=frame_number,
        max_frame=1000,
    )

    # If possible, generate pixel masks within each bounding box
    print(f"Exporting tracks with masks for camera {camera}...")

    generate_thermal_masks_from_bboxes(
        bbox_csv=bboxes_csv,
        video_path=LOCAL_PROCESSED_DIR
        / "aligned_frames"
        / f"thermal_{camera}"
        / "warped_thermal_adjusted.mp4",
        output_mask_dir=LOCAL_PROCESSED_DIR
        / "aligned_frames"
        / f"thermal_{camera}"
        / "masks",
        temp_threshold=128,  # default threshold for thermal images, half of 255
        mask_value=255,
    )
    export_tracks_with_masks(
        tracked_bboxes_csv=bboxes_csv,
        mask_dir=LOCAL_PROCESSED_DIR / "aligned_frames" / f"rgb_{camera}" / "masks",
        output_csv=LOCAL_PROCESSED_DIR
        / "aligned_frames"
        / f"rgb_{camera}"
        / f"tracked_bboxes_{camera}_with_masks.csv",
    )
