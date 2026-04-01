# Tracking Scripts Usage Guide

See the main [README](../../README.rst) for environment setup instructions.

As part of the `full_pipeline.ipynb` notebook, there are 2 key steps: preprocessing (explained here), and inference. If you plan on using already processed data, skip the alignment stage. 

For preprocessing, you will use `thermal_processing.py` and the `alignment_gui.py` scripts, both called in the notebook, but also accessible as standalone scripts. *Note* that the GUI works best when run locally. 

This guide provides best practices and instructions for using the script effectively within the pipeline.

---

## **Overview**

### What is the data?

Our fieldwork data is stored in Google Cloud, with unprocessed data in an internal google drive. The metadata YML file for each session should contain the following fields:

- `notes`: (string, optional) Freeform notes about the session, site, or observations. These are recorded by the researchers on site.
- `data_sources`: (list, required) A list of data source entries, each describing a file used in the session. The thermal folders should be populated with a single .csq file, and the rgb folders with an .MP4

Each entry in `data_sources` should be a dictionary with the following fields:

  - `description`: (string, required) A short description of the data source (e.g., type of camera, sensor, etc.). Camera 1 (thermal or rgb) refers to the leftmost camera in the field. This may sometimes not align with the actual camera number (e.g., thermal_1 may be FLIR2).
  - `original_path`: (string, required) The original location of the file (e.g., on a local drive or cloud storage). Useful for provenance.
  - `path`: (string, required) The relative or final path to the file within the project or data repository. Based on the project structure, this should be a path to a thermal_1, thermal_2, rgb_1, or rgb_2 directory, assuming 2 camera set up.


The data structure is as follows: 
Unique session name: `YYYY_MM_DD-session_0001`

```text
YYYY_MM_DD-session_0001/             # Unique session folder
    ├── thermal_1/                   # Thermal camera 1 data
    │   ├── cameraInfoTime.csq
    │   ├── cameraInfoTime_vmin-vmax.mp4 # if using preprocessed
    │   
    ├── thermal_2/                   # Thermal camera 2 data
    │   ├── cameraInfoTime.csq
    │   ├── cameraInfoTime_vmin-vmax.mp4
    │ 
    ├── rgb_cam_1/                   # RGB camera 1 data
    │   └── cameraSerial.mp4
    ├── rgb_cam_2/                   # RGB camera 2 data
    │   └── cameraSerial.mp4
    └── Metadata.yaml                # Session metadata and notes, described above
```

- `thermal_1` and `thermal_2` contain raw `.csq` files, and will store the processed `.mp4` videos. One may be empty or missing if only one camera is useable.
- `rgb_cam_1` and `rgb_cam_2` store RGB video files from each camera. One may be empty or missing if only one camera is useable.
- `Metadata.yaml` includes session notes, project tags, and metadata for all sources, described above.


### Data processing

`thermal_processing.py` is a tool for converting .csq files into mp4s. 

1. **Conversion**: Set visualization parameters to create the thermal video from the raw data.
2. **Exporting**: Saving the .mp4 videos in the correct file structure.

The `alignment_gui.py` script is a tool for aligning RGB and thermal videos both spatially and temporally. 

The alignment process involves three main steps:
1. **Cropping and Rotating**: Adjust the RGB video to match the thermal video.
2. **Spatial Alignment**: Align the RGB and thermal videos using homography or translation.
3. **Temporal Alignment**: Synchronize the RGB and thermal videos by adjusting frame offsets.

---
## **Tips and Tricks**
### Best Practices for Thermal Processing
* Selecting vmin and vmax  
    **What are vmin and vmax?**  
    These values define the minimum and maximum pixel intensities for visualizing thermal data. Often, the optimal contrast for thermal videos is not the full range of captured temperatures. If autodetection does not produce good visual results, use the preview mode (`--preview` in the command line or `preview = True` in notebook) to manually adjust these parameters.  

    **How to Select:**  
    1. Use the --preview mode to visualize the thermal video.  
        Assess the temperature of the animals and their immediate background. Adjust vmin and vmax to maximize the contrast between them in the remapped coloring to ensure the thermal features of interest are clearly visible.  
        Avoid setting values too high or too low, as this may obscure important details.  
    2. Choosing a Color Map  
        Use color maps like magma or jet to enhance contrast and make thermal features more distinguishable.  
        Experiment with different color maps to find the one that best highlights the features of interest. All should work with the trained YOLO and RF-DETR models, but some finetuning may help.  
    3. Processing Frames  
        If you are working with large .csq files, use the --max_frames argument to limit the number of frames processed for faster results.  
        Ensure the --fps value matches the original frame rate of the thermal camera for accurate playback. In the `data` folder, all videos are at a frame rate of 30fps.  

### Best Practices for Alignment
The alignment mechanism is manual, requiring some user input. The field of view (FOV) of the RGB videos is far larger than that of the thermal camera. To match the FOVs, you may need to crop, rotate, and translate the videos.

* Cropping and Rotation
    **How to Select:**  
    1. Rotate the RGB video to match the line of the horizon in the thermal video.
    2. Pick a cropping rectangle that most similarly resembles that of the thermal video. Looking for relative distances from static keypoints may be helpful. This crop does not need to be perfect, and will likely still have some differences from the thermal FOV. This is due to differences in position and lens shape. 
* Spatial Homography
    **How to Select:**  
    1. Visually inspect the cropped video. Use homography for complex transformations and translation for simple shifts. Typically both are needed, but in cases where the cameras are very close together, translation may be the only necessary step. In this case, change `skip_homography` to true.
    2. Pick at least 4 corresponding points in the RGB and thermal videos for accurate alignment. Again, aligning based on static keypoints (mountain peak, tree, feeder, etc.) leads to the best results. **Try to pick points from all quadrants of the frame.**
    3. Repeat selection on the translated frames to apply the homography.
* Temporal Alignment
    Once the videos are spatially aligned you can use the overlayed videos to move frame by frame to find the highest alignment. This could also be automated via cross-correlation, but manual checking is likely necessary in all cases.


## **Usage**

Use the `full_pipeline.ipynb` notebook, or run the scripts from the command line with the required arguments:
```python
python thermal_processing.py \
    --folder_path path/to/thermal_data \
    --out_path path/to/output_directory \
    --color magma \
    --vmin 50 \
    --vmax 200 \
    --preview True \
    --max_frames 100 \
    --fps 30
```
```python
python alignment_gui.py \
    --rgb_video_path path/to/rgb_video.mp4 \
    --thermal_video_path path/to/thermal_video.mp4 \
    --output_dir_rgb path/to/output/rgb \
    --output_dir_thm path/to/output/thermal \
    --frame_size 640,480 \
    --max_frames 1000 \
    --warp_to rgb \
    --rotation_angle 0.0 \
    --skip_homography \
    --skip_translation
```

### Inference
#### Downloading Model Weights and running inference


This section explains how to download training images from the cloud and train various models (YOLO, RF-DETR) using either the Roboflow API or local scripts. This section is useful if you want to fine-tune a model more. 

#### Downloading Training Images or weights

To access training images please download the zip file from the roboflow_model bucket in the google cloud. 
The bucket should contain a .pt file containing the model weights for a YOLOv11 and rf-detr model and a zip file containing the images, labels, and annotations in YOLO v7 PyTorch format, as well as in the COCO format.


#### Downloading from Google Cloud
1. **Install the Google Cloud CLI**:
   If you haven't already, install the Google Cloud CLI by following the instructions [here](https://cloud.google.com/sdk/docs/install).

2. **Authenticate with Google Cloud**:
   ```bash
   gcloud auth login
   ```

3. **Download the Zip File** OR **Weights**
   Use the `gsutil` command to download the zip file containing images, labels, and annotations:
   ```bash
   gsutil cp gs://roboflow_model/Bird tracking.v11i.yolov7pytorch.zip ./
   ```

4. **Extract the Zip File (if getting images)**:
   After downloading, extract the contents:
   ```bash
   unzip "Bird tracking.v11i.yolov7pytorch.zip" -d model_training
   ```

   Ensure the extracted dataset is organized in the required format (e.g., YOLO or COCO) for training.

   If you want to use the pretrained weights, save the .pt file locally to be called in:
    ```python
    checkpoint_path = LOCAL_DOWNLOAD_DIR / "yolo11_weights.pt"  # Update with your model path

    infer_with_yolo(
        video_path=thermal_video_path,
        model_path=checkpoint_path,
        output_csv_path=detect_csv
    )
    ```


## Tracking
Once the animals are detected via the inference step, we recommend checking the bounding boxes before feeding them to the tracker. 

We use [Ultralytics for object tracking ](https://docs.ultralytics.com/modes/track/) with default parameters. 

### Example Tracking with ByteTracker

The pipeline notebook uses the ByteTracker option. To see the version with idtracker.ai, see below. 

In the pipeline, videos are first parsed using `get_detections_from_video`, which creates a directory of annotated frames and a video showing the overlayed bounding boxes. 
This is what is used as input to `run_tracking`, which uses the `track_objects` function implementing ByteTracker. 
Finally, `output_tracked_bboxes_csv` combines the tracking and detection files into a single one, which can be used to build pixel masks or infer animal size. 


Detection and Tracking is also compatible with [idtracker.ai](https://idtracker.ai/latest/user_guide/usage.html), with parameters selected in their GUI. 

### Example Tracker Configuration for idtracker.ai

Below is an example configuration .toml for running tracking on a processed thermal video:

```python
video_paths = [LOCAL_PROCESSED_DIR_{some_video},
LOCAL_PROCESSED_DIR_{some_video}
]
intensity_ths = [19, 255]                # Minimum and maximum pixel intensity thresholds
area_ths = [50.0, float('inf')]          # Minimum and maximum area thresholds for detected objects
tracking_intervals = ""                  # Specify intervals if needed, empty string for full video
number_of_animals = 10                   # Expected number of animals to track
use_bkg = True                           # Enable background subtraction
check_segmentation = False               # Disable segmentation check
track_wo_identities = False              # Track with identities
background_subtraction_stat = 'median'   # Use median for background subtraction
```

You can also run 
```python
idtrackerai --load example.toml --track
```
Adjust these parameters as needed for your dataset and tracking requirements.
