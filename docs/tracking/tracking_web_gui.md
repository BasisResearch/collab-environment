# Tracking Studio Web GUI

Interactive web-based application for real-time video object detection and tracking using YOLO and ByteTrack.

## Overview

The Tracking Studio provides a user-friendly interface for:
- Loading videos from Google Cloud Storage or local uploads
- Selecting detection models (YOLO, Roboflow, or custom .pt files)
- Tuning ByteTrack parameters in real-time
- Visualizing tracking results with live preview
- Exporting tracking data to CSV

## Quick Start

### Prerequisites

1. **Python 3.10** with the project installed:

   ```bash
   pip install -e .
   ```

2. **FFmpeg** (for video format conversion):

   ```bash
   # macOS
   brew install ffmpeg
   # Ubuntu/Debian
   sudo apt install ffmpeg
   ```

3. **GCS credentials** (for browsing videos in Google Cloud Storage):
   - Place your service account JSON at `config-local/collab-data-463313-c340ad86b28e.json`
   - Or set the env var: `export GCS_CREDENTIALS=/path/to/credentials.json`
   - If neither is set, GCS browsing is disabled (video upload still works)

4. **Roboflow API key** (only needed for Roboflow models):

   ```bash
   export ROBOFLOW_API_KEY=your_api_key_here
   ```

   Get your key from [Roboflow settings](https://app.roboflow.com/settings/api)

### Running the Application

```bash
# From the repository root
python scripts/tracking/run_tracking_studio.py
```

The application will start on `http://localhost:8080`

## Workflow

### 1. Load Video

**From Google Cloud Storage:**
1. Select **Bucket** (e.g., `collab-data-463313`)
2. Select **Folder** (optional subfolder)
3. Select **Video** from the list
4. Click **Load Video**

**From Local Upload:**
1. Click **Upload** button
2. Select video file (.mp4, .mov, .avi)
3. Click **Load Video**

The first frame will display in the preview area.

### 2. Load Model

Choose one of three model sources:

#### YOLO Models
- Enter any YOLO model name (e.g., `yolo11n.pt`, `yolo26n.pt`)
- Models will auto-download if available from Ultralytics
- Click **Load Model**

#### Roboflow Models
- Enter **Project ID** in format: `workspace/project`
- Click **List Models** to fetch available versions
- Select a **Version** from dropdown
- Click **Load Model**

**Supported types:**
- Object detection models (standard)
- Instance segmentation models (extracts bounding boxes only)

#### Custom Models
- Upload your own `.pt` file
- Click **Load Model**

### 3. Configure Parameters

**Detection Confidence**
- Threshold for detection scores (0.1 - 0.9)
- Higher = fewer false positives, may miss detections
- Lower = more detections, may include noise

**ByteTrack Parameters** (see [ByteTrack Parameters](#bytetrack-parameters) below)

**Skip Frames**
- Process every Nth frame (1 = all frames, 30 = every 30th frame)
- Use higher values for faster preview on long videos
- Final tracking still captures data for skipped frames

**Display Update**
- Update preview every Nth frame (1-30, default: 10)
- Lower values = smoother preview (more network traffic)
- Higher values = less frequent updates (lower bandwidth)
- At 30fps video: 10 frames = ~3 updates/second, 5 frames = ~6 updates/second

### 4. Start Tracking

1. Click **Start Tracking**
2. Watch live preview with bounding boxes and track IDs
3. Use **Pause** to temporarily halt processing
4. Use **Stop** to terminate early
5. Drag the time slider to jump to specific frames

### 5. Export Results

When complete, click **Download CSV** to save tracking data.

## ByteTrack Parameters

ByteTrack uses a two-stage association algorithm to track objects across frames:

### `track_high_thresh` (default: 0.25)
Detection confidence threshold separating high-confidence and low-confidence detections.
- Detections **above** this → Stage 1 (primary IoU matching)
- Detections **between** `track_low_thresh` and this → Stage 2 (secondary matching)
- **Raise** to restrict primary matching to most confident detections
- **Lower** to feed more detections into Stage 1

### `track_low_thresh` (default: 0.1)
Absolute minimum detection confidence.
- Detections **below** this are discarded entirely
- **Lower** to recover marginal detections (more noise)
- **Raise** to filter weak false positives

### `new_track_thresh` (default: 0.25)
Minimum confidence required to initialize a new track.
- Unmatched detections from Stage 1 must exceed this to spawn new track IDs
- **Higher** prevents spurious tracks from false positives
- **Lower** allows tracks to start from weaker detections

### `track_buffer` (default: 30)
Number of frames a lost track is kept alive before deletion.
- Internally scaled by frame rate: `max_time_lost = int(fps / 30.0 * track_buffer)`
- **Higher** values let tracks survive longer occlusions (more ID switches risk)
- **Lower** values remove lost tracks faster

### `match_thresh` (default: 0.8)
IoU-distance gating threshold for Stage 1 association.
- Cost = 1 - IoU, so threshold of 0.8 accepts matches with IoU ≥ 0.2
- **Higher** values are more lenient (easier to match)
- **Lower** values require stronger spatial overlap
- Tune based on detector quality

### `fuse_score` (default: true)
Multiply IoU similarity by detection confidence before matching.
- Formula: `fused_cost = 1 - (iou_similarity * detection_score)`
- **Enable** to bias matching toward high-confidence detections
- **Disable** if detector's confidence scores are poorly calibrated

## Output Format

### Tracking CSV

Format: `tracking.csv`

| Column | Type | Description |
|--------|------|-------------|
| `track_id` | int | Unique object track identifier |
| `frame` | int | Frame number (0-indexed) |
| `x1` | int | Bounding box top-left X |
| `y1` | int | Bounding box top-left Y |
| `x2` | int | Bounding box bottom-right X |
| `y2` | int | Bounding box bottom-right Y |
| `confidence` | float | Detection confidence score |
| `class` | int | Object class ID |

Example:
```csv
track_id,frame,x1,y1,x2,y2,confidence,class
1,0,245,150,345,280,0.87,0
1,1,247,152,346,281,0.85,0
2,1,450,200,550,320,0.92,0
```

## Model Support

### YOLO Models (Ultralytics)
- ✅ YOLO11 series (`yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, etc.)
- ✅ YOLO26 series (`yolo26n.pt`, `yolo26s.pt`, etc.)
- ✅ Custom trained YOLO models (.pt files)
- Uses **native Ultralytics tracking** (supports all 6 ByteTrack parameters)

### Roboflow Models
- ✅ Object detection models
- ✅ Instance segmentation models (bounding boxes only, masks ignored)
- Uses **model download + local inference** (YOLO-compatible weights)
- Fallback to **supervision ByteTrack** if native tracking unavailable

### Custom Models
- ✅ Upload any YOLO-compatible `.pt` file
- Must be trainable with Ultralytics YOLO framework

## Architecture

### Components

**Frontend: NiceGUI**
- Reactive web interface
- Real-time frame updates via WebSocket
- Slider-based parameter tuning

**Backend: FastAPI (via NiceGUI)**
- Async video processing with `asyncio.to_thread()`
- Background thread handles heavy CV operations
- Event loop scheduling for UI updates

**Video Processing: [video_processor.py](../../collab_env/tracking_studio/video_processor.py)**
- OpenCV video capture
- Frame-by-frame detection + tracking
- Supervision library for annotation
- Temporary YAML config for ByteTrack parameters

**Model Management: [model_manager.py](../../collab_env/tracking_studio/model_manager.py)**
- YOLO model loading (Ultralytics)
- Roboflow model loading (inference SDK + fallback download)
- Model caching for faster reloads

## Video Format Support

### Supported Formats
- MP4 (H.264 codec) - **recommended**
- MOV (QuickTime)
- AVI (uncompressed)

### Automatic Conversion
Videos not in H.264 format are automatically converted on load:
- Source: mjpeg, raw, etc.
- Target: H.264 MP4 (1080p max, 30fps)
- Uses FFmpeg via `video_converter.py`

## Playback Controls

### Real-time Controls
- **Start Tracking**: Begin processing video
- **Pause**: Temporarily halt (can resume)
- **Stop**: Hard stop (terminates processing)

### Seeking
- Drag **time slider** during processing to jump to specific frame
- Shows raw frame preview during drag (no tracking)
- Releases to seek tracker forward/backward

### Preview Updates
- Updates every 10 frames for performance (~3x per second at 30fps)
- Shows annotated frames with bounding boxes and track IDs

## Performance Tips

### For Long Videos
1. Use **Skip Frames** to process every Nth frame for faster preview
2. Increase **track_buffer** to maintain tracks across skipped frames
3. Lower **detection confidence** if missing objects

### For Crowded Scenes
1. Raise **track_high_thresh** to focus on confident detections
2. Lower **match_thresh** to require tighter spatial overlap
3. Increase **new_track_thresh** to reduce spurious tracks

### For Fast Motion
1. Lower **match_thresh** to accept looser spatial matches
2. Increase **track_buffer** to keep tracks alive longer
3. Process all frames (Skip = 1) for smoother tracking

## Troubleshooting

### "No detections found"
- Lower **detection confidence** slider
- Check video quality and lighting
- Try different model (e.g., `yolo11m.pt` instead of `yolo11n.pt`)

### "Too many false positives"
- Raise **detection confidence**
- Increase **new_track_thresh**
- Raise **track_high_thresh**

### "Track IDs jumping/switching"
- Lower **track_high_thresh** to feed more detections into Stage 1
- Raise **match_thresh** for more lenient matching
- Increase **track_buffer** to keep lost tracks alive longer
- Enable **fuse_score** if disabled

### "Video conversion failed"
- Check FFmpeg installation: `ffmpeg -version`
- Ensure video file is not corrupted
- Try converting manually: `ffmpeg -i input.mp4 -c:v libx264 output.mp4`

### "Model loading failed"
- YOLO: Check model name spelling and internet connection
- Roboflow: Verify `ROBOFLOW_API_KEY` is set and has access
- Custom: Ensure `.pt` file is YOLO-compatible format

### Batch Processing

For offline batch processing without the GUI, use the [full tracking pipeline notebook](full_pipeline.ipynb) or direct API:

```python
from collab_env.tracking_studio.video_processor import VideoTracker
from collab_env.tracking_studio.model_manager import ModelManager

# Load model
manager = ModelManager()
model = manager.load_yolo_model("yolo11n.pt")

# Configure tracker
tracker_config = {
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.25,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "fuse_score": True,
}

tracker = VideoTracker(model=model, tracker_config=tracker_config, confidence=0.5)

# Process video
results = await tracker.process_video_realtime("input.mp4", "/tmp/output")
print(f"Saved to: {results['tracking_csv']}")
```

## References

- [ByteTrack Paper](https://arxiv.org/abs/2110.06864)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [Supervision Library](https://supervision.roboflow.com/)
- [Roboflow Inference](https://inference.roboflow.com/)
