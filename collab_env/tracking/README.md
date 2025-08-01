# Tracking scripts usage guide

As part of the 'full_pipeline' notebook, there are 2 key steps: preprocessing (explained here), and inference (weight download and running explained in the model folder README). If you plan on using already processed data, skip the alignment stage. 

For preprocessing, you will use `thermal_processing.py` and the `alignment_gui.py` scripts, both called in the notebook, but also accessible as standalone scripts. *Note* that the GUI works best when run locally. 

This guide provides best practices and instructions for using the script effectively within the pipeline.

---

## **Overview**
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
#### Best Practices for Thermal Processing
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

#### Best Practices for Alignment
The alignment mechanism is manual, requiring some user input. The field of view (FOV) of the RGB videos is far larger than that of the thermal camera. To match the FOVs, you may need to crop, rotate, and translate the videos.

* Cropping and Rotation
    **How to Select:**  
    1. Rotate the RGB video to match the line of the horizon in the thermal video.
    2. Pick a cropping rectangle that most similarly resembles that of the thermal video. Looking for relative distances from static keypoints may be helpful. This crop does not need to be perfect, and will likely still have some differences from the thermal FOV. This is due to differences in position and lens shape. 
* Spatial Homography
    **How to Select:**  
    1. Visually inspect the cropped video. Use homography for complex transformations and translation for simple shifts. Typically both are needed, but in cases where the cameras are very close together, translation may be the only necessary step. In this case, change `skip_homography` to true.
    2. Pick at least 4 corresponding points in the RGB and thermal videos for accurate alignment. Again, aligning based on static keypoints (mountain peak, tree, feeder, etc.) leads to the best results. Try to pick points from all quadrants of the frame.
    3. Repeat selection on the translated frames to apply the homography.
* Temporal Alignment
    Once the videos are spatially aligned you can use the overlayed videos to move frame by frame to find the highest alignment. This could also be automated via cross-correlation, but manual checking is likely necessary in all cases.


## **Usage**

Use the `full_pipeline` notebook, or run the scripts from the command line with the required arguments:
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