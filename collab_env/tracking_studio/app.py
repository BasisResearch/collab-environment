"""
Main NiceGUI Tracking Studio Application

Single-page interactive app for video tracking with:
- GCS bucket browsing and video upload
- Model selection (YOLO and Roboflow)
- Real-time tracking visualization
- CSV output download
"""

from nicegui import ui, app
import asyncio
from pathlib import Path
import uuid
import os
import io
from loguru import logger

from .gcs_browser import GCSVideoBrowser
from .model_manager import ModelManager
from .video_processor import VideoTracker
from .video_converter import convert_to_h264, needs_conversion


# Load ByteTrack parameter definitions
def load_bytetrack_params():
    """Load ByteTrack parameter schema from JSON"""
    import json
    params_file = Path(__file__).parent / "bytetrack_params.json"
    with open(params_file, 'r') as f:
        return json.load(f)

bytetrack_params_schema = load_bytetrack_params()


# Initialize services
def get_credentials_path():
    """Get GCS credentials path from environment or default. Returns None for ADC."""
    env_path = os.getenv("GCS_CREDENTIALS")
    if env_path:
        return env_path
    default = "/workspace/config/collab-data-463313-c340ad86b28e.json"
    if os.path.exists(default):
        return default
    return None  # GCSClient will use Application Default Credentials


try:
    gcs_browser = GCSVideoBrowser(credentials_path=get_credentials_path())
except Exception as e:
    logger.error(f"Failed to initialize GCS browser: {e}")
    gcs_browser = None

model_manager = ModelManager()


@ui.page("/")
async def index():
    """Main tracking studio page"""
    session_id = str(uuid.uuid4())[:8]

    # State variables (stored in page context)
    import threading
    state = {
        "selected_bucket": None,
        "selected_video_path": None,
        "selected_model": None,
        "processing": False,
        "results": None,
        "uploaded_video": None,
        "uploaded_model": None,  # Uploaded model .pt file
        "stop_event": None,  # Hard stop
        "pause_event": None,  # Pause/resume
        "skip_frames_event": None,  # Skip forward signal
        "video_path": None,  # Path to current video being processed
        "video_loaded": False,  # Video is loaded and ready for playback
        "model_loaded": False,  # Model is loaded and ready for tracking
        "loaded_model": None,  # Reference to loaded model object
    }

    # UI Layout
    with ui.column().classes("w-full p-4 gap-3"):
        # Header
        with ui.row().classes("w-full items-center mb-2"):
            ui.label("🎯 Video Tracking Studio").classes("text-2xl font-bold")
            ui.space()
            ui.label("Real-time object detection and tracking").classes("text-sm text-gray-600")

        # Row 1: Video source selection (Bucket | Folder | Video | Upload)
        with ui.card().classes("w-full shadow-md p-2"):
            with ui.row().classes("w-full gap-2 items-end"):
                # GCS selection
                if gcs_browser:
                    bucket_select = ui.select(
                        label="Bucket", options=[],
                    ).style("width: 250px")
                    try:
                        buckets = gcs_browser.list_buckets()
                        bucket_select.options = buckets
                        if buckets:
                            bucket_select.value = buckets[0]
                    except Exception as e:
                        logger.error(f"Failed to list buckets: {e}")

                    folder_select = ui.select(
                        label="Folder", options=[""], value="", clearable=True
                    ).style("width: 350px")

                    video_select = ui.select(label="Video", options=[]).classes("flex-grow")

                    async def update_folders(e):
                        """Update folder list when bucket changes"""
                        try:
                            bucket = bucket_select.value
                            if bucket:
                                folders = gcs_browser.list_folders(bucket, "")
                                folder_select.options = [""] + folders
                                folder_select.value = ""
                                folder_select.update()
                                await update_video_list(None)
                        except Exception as error:
                            logger.error(f"Failed to list folders: {error}")

                    async def update_video_list(e):
                        """Update video list when bucket or folder changes"""
                        try:
                            bucket = bucket_select.value
                            folder = folder_select.value or ""
                            if bucket:
                                videos = gcs_browser.list_videos(bucket, folder)
                                video_select.options = [v["rel_path"] for v in videos]
                                video_select.update()
                        except Exception as error:
                            logger.error(f"Failed to list videos: {error}")

                    def enable_load_video_btn(e=None):
                        """Enable Load Video button when video is selected"""
                        if video_select.value or state.get("uploaded_video"):
                            load_video_btn.enable()

                    bucket_select.on("update:model-value", update_folders)
                    folder_select.on("update:model-value", update_video_list)
                    video_select.on("update:model-value", enable_load_video_btn)

                    if bucket_select.value:
                        ui.timer(0.1, lambda: update_folders(None), once=True)

                # Upload widget
                async def handle_upload(e):
                    """Handle user video upload"""
                    try:
                        upload_path = Path(f"/tmp/uploads/{session_id}")
                        upload_path.mkdir(parents=True, exist_ok=True)
                        uploaded_file = upload_path / e.name
                        uploaded_file.write_bytes(e.content.read())
                        state["uploaded_video"] = uploaded_file
                        ui.notify(f"Uploaded: {e.name}")
                        load_video_btn.enable()  # Enable Load Video button
                    except Exception as error:
                        logger.error(f"Upload failed: {error}")
                        ui.notify(f"Upload failed: {error}", type="negative")

                upload = ui.upload(
                    on_upload=handle_upload,
                    auto_upload=True,
                ).props("accept=video/mp4,video/quicktime,video/x-msvideo dense flat").props("label=Upload").style("width: 120px; height: 40px")

        # Row 2: Model/Params (left) | Controls + Preview (right)
        with ui.row().classes("w-full gap-3"):
            # LEFT: Model + Parameters (stacked)
            with ui.column().classes("gap-3").style("flex: 0 0 280px; min-width: 280px"):
                # Model card
                with ui.card().classes("w-full shadow-md p-3"):
                    ui.label("🤖 Model").classes("text-sm font-semibold mb-2")

                    # Radio with three options
                    with ui.row().classes("gap-2"):
                        model_source = ui.radio(["YOLO", "Roboflow", "Custom"], value="YOLO").classes("text-xs")

                    # YOLO model selection (default visible)
                    yolo_container = ui.column().classes("w-full mt-2")
                    with yolo_container:
                        yolo_model_input = ui.input(
                            label="Model Name",
                            placeholder="e.g., yolo11n.pt",
                            value="yolo11n.pt"
                        ).classes("w-full").tooltip("Enter any YOLO model name (will auto-download if available)")

                    # Roboflow model selection (hidden by default)
                    rf_container = ui.column().classes("w-full mt-2 gap-2")
                    rf_container.visible = False
                    with rf_container:
                        rf_project_input = ui.input(
                            label="Project ID",
                            placeholder="workspace/project",
                            value="dima-sdrkv/ratsmerged20260211"
                        ).classes("w-full")

                        # Define the function BEFORE referencing it in the button
                        async def list_rf_models():
                            """Query Roboflow for available model versions"""
                            project_id = rf_project_input.value
                            if not project_id:
                                ui.notify("Please enter project ID", type="warning")
                                return

                            try:
                                rf_list_btn.disable()
                                # Call directly (synchronous HTTP request, no need for threading)
                                versions = model_manager.list_roboflow_project_models(project_id)
                                if versions:
                                    rf_version_select.options = versions
                                    rf_version_select.value = versions[0]
                                    rf_version_select.enable()
                                    ui.notify(f"Found {len(versions)} versions", type="positive")
                                else:
                                    ui.notify("No versions found", type="warning")
                            except Exception as error:
                                logger.error(f"Failed to list models: {error}")
                                ui.notify(f"Error: {error}", type="negative")
                            finally:
                                rf_list_btn.enable()

                        # Now create the button and version select (after function definition)
                        with ui.row().classes("w-full gap-2"):
                            rf_list_btn = ui.button("List Models", on_click=list_rf_models).props("size=sm color=primary")

                        rf_version_select = ui.select(
                            label="Version",
                            options=[],
                        ).classes("w-full")
                        rf_version_select.disable()

                    # Custom model upload (hidden by default)
                    custom_container = ui.column().classes("w-full mt-2")
                    custom_container.visible = False
                    with custom_container:
                        # Upload widget for model weights
                        async def handle_model_upload(e):
                            """Handle model .pt file upload"""
                            try:
                                model_upload_path = Path(f"/tmp/models/{session_id}")
                                model_upload_path.mkdir(parents=True, exist_ok=True)
                                uploaded_model_file = model_upload_path / e.name
                                uploaded_model_file.write_bytes(e.content.read())
                                state["uploaded_model"] = uploaded_model_file
                                ui.notify(f"Model uploaded: {e.name}", type="positive")
                                logger.info(f"Model uploaded to: {uploaded_model_file}")
                                load_model_btn.enable()  # Enable Load Model button
                            except Exception as error:
                                logger.error(f"Model upload failed: {error}")
                                ui.notify(f"Model upload failed: {error}", type="negative")

                        ui.upload(
                            on_upload=handle_model_upload,
                            auto_upload=True,
                        ).props("accept=.pt dense flat").props("label=Upload Model (.pt)").classes("w-full")

                    # Toggle visibility based on model source
                    def toggle_model_ui(e=None):
                        value = model_source.value
                        if value == "YOLO":
                            yolo_container.visible = True
                            rf_container.visible = False
                            custom_container.visible = False
                            # Enable load button if YOLO model name is entered
                            if yolo_model_input.value:
                                load_model_btn.enable()
                        elif value == "Roboflow":
                            yolo_container.visible = False
                            rf_container.visible = True
                            custom_container.visible = False
                            # Enable load button if Roboflow model is selected
                            if rf_version_select.value:
                                load_model_btn.enable()
                        else:  # Custom
                            yolo_container.visible = False
                            rf_container.visible = False
                            custom_container.visible = True
                            # Enable load button if custom model uploaded
                            if state.get("uploaded_model"):
                                load_model_btn.enable()

                    def enable_load_model_btn(e=None):
                        """Enable Load Model button when model is selected"""
                        if model_source.value == "YOLO" and yolo_model_input.value:
                            load_model_btn.enable()
                        elif model_source.value == "Roboflow" and rf_version_select.value:
                            load_model_btn.enable()
                        elif model_source.value == "Custom" and state.get("uploaded_model"):
                            load_model_btn.enable()

                    model_source.on("update:model-value", toggle_model_ui)
                    yolo_model_input.on("update:model-value", enable_load_model_btn)
                    rf_version_select.on("update:model-value", enable_load_model_btn)

                # Parameters card
                with ui.card().classes("w-full shadow-md p-3"):
                    ui.label("⚙️ Parameters").classes("text-sm font-semibold mb-2")

                    # Detection confidence (not in ByteTrack params)
                    with ui.column().classes("w-full gap-1"):
                        conf_label = ui.label("Confidence: 0.50").classes("text-xs")
                        conf_slider = ui.slider(min=0.1, max=0.9, step=0.05, value=0.5).classes("w-full").tooltip("Detection confidence threshold")
                        conf_slider.on("update:model-value", lambda e: conf_label.set_text(f"Confidence: {e.args:.2f}"))

                    # Dynamic ByteTrack parameters from JSON
                    param_widgets = {}  # Store references to UI elements
                    with ui.column().classes("w-full gap-1 mt-2"):
                        for param_name, param_config in bytetrack_params_schema.items():
                            if param_config["type"] == "float":
                                # Float slider
                                default_val = param_config["default"]
                                min_val, max_val = param_config["range"]

                                # Create label with tooltip
                                param_label = ui.label(f"{param_name.replace('_', ' ').title()}: {default_val:.2f}").classes("text-xs")
                                param_label.tooltip(param_config["description"])

                                # Create slider
                                step = 0.05 if max_val <= 1.0 else 0.1
                                param_slider = ui.slider(
                                    min=min_val,
                                    max=max_val,
                                    step=step,
                                    value=default_val
                                ).classes("w-full")

                                # Update label on change
                                param_slider.on(
                                    "update:model-value",
                                    lambda e, lbl=param_label, name=param_name: lbl.set_text(
                                        f"{name.replace('_', ' ').title()}: {e.args:.2f}"
                                    )
                                )
                                param_widgets[param_name] = param_slider

                            elif param_config["type"] == "int":
                                # Int slider
                                default_val = param_config["default"]
                                min_val = param_config["range"][0]
                                max_val = param_config["range"][1] if param_config["range"][1] else 300

                                param_label = ui.label(f"{param_name.replace('_', ' ').title()}: {default_val}").classes("text-xs")
                                param_label.tooltip(param_config["description"])

                                param_slider = ui.slider(
                                    min=min_val,
                                    max=max_val,
                                    step=1,
                                    value=default_val
                                ).classes("w-full")

                                param_slider.on(
                                    "update:model-value",
                                    lambda e, lbl=param_label, name=param_name: lbl.set_text(
                                        f"{name.replace('_', ' ').title()}: {int(e.args)}"
                                    )
                                )
                                param_widgets[param_name] = param_slider

                            elif param_config["type"] == "bool":
                                # Checkbox
                                param_checkbox = ui.checkbox(
                                    param_name.replace('_', ' ').title(),
                                    value=param_config["default"]
                                ).classes("text-xs")
                                param_checkbox.tooltip(param_config["description"])
                                param_widgets[param_name] = param_checkbox

                            elif param_config["type"] == "string":
                                # Dropdown for options
                                param_select = ui.select(
                                    label=param_name.replace('_', ' ').title(),
                                    options=param_config["options"],
                                    value=param_config["default"]
                                ).classes("w-full text-xs")
                                param_select.tooltip(param_config["description"])
                                param_widgets[param_name] = param_select

                    # Skip frames (for fast-forward, not a ByteTrack param)
                    with ui.row().classes("w-full items-center gap-2 mt-2"):
                        skip_frames_label = ui.label("Skip Frames: 1 frame").classes("text-xs")
                        skip_frames_slider = ui.slider(min=1, max=30, step=1, value=1).style("width: 100px")
                        skip_frames_slider.tooltip("Process every Nth frame (1 = all frames)")
                        skip_frames_slider.on("update:model-value", lambda e: skip_frames_label.set_text(f"Skip Frames: {int(e.args)} {'frame' if int(e.args) == 1 else 'frames'}"))

                    # GUI refresh rate (display updates, not a ByteTrack param)
                    with ui.row().classes("w-full items-center gap-2 mt-1"):
                        display_update_label = ui.label("Display Update: 10 frames").classes("text-xs")
                        display_update_slider = ui.slider(min=1, max=30, step=1, value=10).style("width: 100px")
                        display_update_slider.tooltip("Update display every Nth frame (lower = smoother, more network traffic)")
                        display_update_slider.on("update:model-value", lambda e: display_update_label.set_text(f"Display Update: {int(e.args)} {'frame' if int(e.args) == 1 else 'frames'}"))

            # RIGHT: Controls + Preview (stacked vertically)
            with ui.column().classes("flex-grow gap-3"):
                # Controls card
                with ui.card().classes("w-full shadow-md p-3"):
                    # Row 1: Load buttons
                    with ui.row().classes("w-full items-center gap-2 mb-2"):
                        load_video_btn = ui.button("Load Video").props("color=primary icon=video_file")
                        load_video_btn.disable()  # Enabled when video selected

                        load_model_btn = ui.button("Load Model").props("color=primary icon=model_training")
                        load_model_btn.disable()  # Enabled when model selected

                        ui.separator().props("vertical")

                        with ui.column().classes("flex-grow gap-1"):
                            status_label = ui.label("Select video and model").classes("text-xs")

                    # Row 2: Playback controls
                    with ui.row().classes("w-full items-center gap-2"):
                        start_btn = ui.button("Start Tracking").props("color=positive icon=play_arrow")
                        start_btn.disable()  # Enabled when both video and model loaded

                        pause_btn = ui.button("Pause").props("color=warning icon=pause")
                        pause_btn.disable()

                        stop_btn = ui.button("Stop").props("color=negative icon=stop")
                        stop_btn.disable()

                        ui.separator().props("vertical")

                        with ui.column().classes("flex-grow gap-1"):
                            progress_label = ui.label("Ready").classes("text-xs")
                            progress = ui.linear_progress(value=0).props("size=15px color=primary")

                    # Time slider for seeking
                    with ui.column().classes("w-full gap-1 mt-2"):
                        time_label = ui.label("Frame: 0 / 0").classes("text-xs text-gray-600")
                        time_slider = ui.slider(min=0, max=100, value=0).props("lazy").classes("w-full")
                        time_slider.disable()

                        async def preview_frame_on_drag(e):
                            """Show video frame preview during drag (no detection/tracking)"""
                            if not state.get("video_path"):
                                return

                            target_frame = int(e.args)
                            try:
                                # Open video for preview (separate from processing thread)
                                import cv2
                                import base64
                                cap = cv2.VideoCapture(str(state["video_path"]))
                                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                                ret, frame = cap.read()
                                cap.release()

                                if ret:
                                    # Show raw frame without annotations
                                    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                                    video_display.set_source(f"data:image/jpeg;base64,{img_base64}")
                                    total_frames = state.get("total_frames", target_frame)
                                    time_label.text = f"Frame: {target_frame} / {total_frames}"
                            except Exception as err:
                                logger.warning(f"Preview failed: {err}")

                        def seek_to_frame(e):
                            """Seek to specific frame when slider is released"""
                            if state.get("processing") and state.get("skip_frames_event"):
                                target_frame = int(e.args)
                                current_frame = state.get("current_frame", 0)
                                if target_frame != current_frame:
                                    # Clear any pending seeks
                                    state["skip_frames_event"]["skip_amount"] = target_frame - current_frame
                                    ui.notify(f"Seeking to frame {target_frame}...", type="info")

                        # Live preview during drag
                        time_slider.on("update:model-value", preview_frame_on_drag)
                        # Actual seek on release
                        time_slider.on("change", seek_to_frame)

                # Preview card
                with ui.card().classes("w-full shadow-md p-3"):
                    ui.label("Live Preview").classes("text-sm font-semibold mb-2")
                    video_display = ui.interactive_image().classes("w-full border-2 border-gray-200 rounded bg-gray-50").style("max-height: 500px; object-fit: contain;")

        # Results (initially hidden, separate row)
        results_container = ui.card().classes("w-full shadow-md p-3 hidden")
        with results_container:
            with ui.row().classes("w-full items-center gap-3"):
                ui.label("✅ Results").classes("text-sm font-semibold")
                stats_label = ui.label().classes("text-sm flex-grow")
                download_track_btn = ui.button("Download CSV").props("color=primary icon=download size=sm")

    # Event handlers
    async def load_video():
        """Load and prepare video for viewing/tracking"""
        from nicegui import context

        try:
            status_label.text = "Loading video..."
            load_video_btn.disable()

            # Capture client context before threading
            client = context.client

            # Get video (either download from GCS or use uploaded)
            if state.get("uploaded_video"):
                # Use uploaded video
                local_video = state["uploaded_video"]
                status_label.text = "Using uploaded video..."
            elif gcs_browser and bucket_select.value and video_select.value:
                # Download from GCS
                status_label.text = "Downloading video..."
                bucket = bucket_select.value
                folder = folder_select.value or ""
                video_name = video_select.value
                gcs_path = f"{bucket}/{video_name}"

                local_video_dir = Path(f"/tmp/videos/{session_id}")
                local_video_dir.mkdir(parents=True, exist_ok=True)
                local_video = local_video_dir / Path(video_name).name

                await asyncio.to_thread(gcs_browser.download_video, gcs_path, str(local_video))
            else:
                raise ValueError("No video selected. Please select or upload a video.")

            # Convert to H.264 if needed
            if await asyncio.to_thread(needs_conversion, local_video):
                status_label.text = "Converting to H.264..."
                converted_video = local_video.parent / f"{local_video.stem}_h264.mp4"
                await asyncio.to_thread(convert_to_h264, local_video, converted_video)
                local_video = converted_video
                with client:
                    ui.notify("Video converted to H.264")

            # Restore context for UI updates after threading
            with client:
                # Store video path in state
                state["video_path"] = local_video
                state["video_loaded"] = True

                # Set up video display and time slider
                import cv2
                import base64
                cap = cv2.VideoCapture(str(local_video))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                state["total_frames"] = total_frames

                # Display first frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if ret:
                    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    video_display.set_source(f"data:image/jpeg;base64,{img_base64}")

                cap.release()

                # Enable time slider for playback
                time_slider.enable()
                time_slider.set_value(0)
                time_slider.props(f"max={total_frames}")
                time_label.text = f"Frame: 0 / {total_frames}"

                status_label.text = "Video loaded ✓"
                ui.notify("Video loaded successfully", type="positive")

                # Enable Start button if model is also loaded
                if state["model_loaded"]:
                    start_btn.enable()

        except Exception as e:
            logger.error(f"Failed to load video: {e}", exc_info=True)
            with client:
                status_label.text = f"Error loading video"
                ui.notify(f"Error: {str(e)}", type="negative")
        finally:
            with client:
                load_video_btn.enable()

    async def load_model():
        """Load selected model"""
        from nicegui import context

        try:
            status_label.text = "Loading model..."
            load_model_btn.disable()

            # Capture client context before threading
            client = context.client

            if model_source.value == "YOLO":
                model = await asyncio.to_thread(
                    model_manager.load_yolo_model, yolo_model_input.value
                )
            elif model_source.value == "Roboflow":
                # Roboflow mode: download from API
                project_id = rf_project_input.value
                version = rf_version_select.value

                if not project_id or not version:
                    raise ValueError("Please select a Roboflow model (project ID and version)")

                # Validate project ID format (should be workspace/project)
                project_parts = project_id.split('/')
                if len(project_parts) != 2:
                    raise ValueError(
                        f"Invalid project ID: '{project_id}'\n"
                        f"Expected format: workspace/project (e.g., 'dima-sdrkv/ratsmerged20260211')"
                    )

                # Construct full model ID: workspace/project/version
                model_id = f"{project_id}/{version}"
                logger.info(f"Loading Roboflow model: {model_id}")

                model = await asyncio.to_thread(
                    model_manager.load_roboflow_model, model_id
                )
            else:  # Custom
                # Custom mode: use uploaded model file
                if state.get("uploaded_model"):
                    model_path = str(state["uploaded_model"])
                    logger.info(f"Loading uploaded model: {model_path}")
                    model = await asyncio.to_thread(
                        model_manager.load_roboflow_model, model_path
                    )
                else:
                    raise ValueError("Please upload a model .pt file")

            # Restore context for UI updates after threading
            with client:
                # Store model in state
                state["loaded_model"] = model
                state["model_loaded"] = True

                # Detect tracker type for display
                from ultralytics import YOLO
                tracker_type = "YOLO Native" if isinstance(model, YOLO) else "Supervision"
                state["tracker_type"] = tracker_type

                status_label.text = f"Model loaded ✓ ({tracker_type} tracking)"
                ui.notify("Model loaded successfully", type="positive")

                # Enable Start button if video is also loaded
                if state["video_loaded"]:
                    start_btn.enable()

        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            # Restore context for error UI updates
            with client:
                status_label.text = f"Error loading model"
                ui.notify(f"Error: {str(e)}", type="negative")
        finally:
            with client:
                load_model_btn.enable()

    def pause_tracking():
        """Pause/resume tracking"""
        if state["pause_event"]:
            if state["pause_event"].is_set():
                # Currently paused, resume
                state["pause_event"].clear()
                pause_btn.props("icon=pause")
                pause_btn.text = "Pause"
                progress_label.text = "Resuming..."
                ui.notify("Resumed", type="info")
            else:
                # Currently running, pause
                state["pause_event"].set()
                pause_btn.props("icon=play_arrow")
                pause_btn.text = "Resume"
                progress_label.text = "Paused"
                ui.notify("Paused", type="warning")


    def stop_tracking():
        """Hard stop - terminates processing"""
        if state["stop_event"]:
            state["stop_event"].set()
            progress_label.text = "Stopping..."
            ui.notify("Stopping tracking...", type="negative")
    async def start_tracking():
        """Start tracking on already-loaded video with already-loaded model"""
        if not state.get("video_loaded") or not state.get("model_loaded"):
            ui.notify("Please load video and model first", type="warning")
            return

        state["processing"] = True
        state["stop_event"] = threading.Event()  # Hard stop
        state["pause_event"] = threading.Event()  # Pause (starts clear = not paused)
        state["skip_frames_event"] = {"skip_amount": 0}  # Skip forward
        state["current_frame"] = 0
        start_btn.disable()
        pause_btn.enable()
        stop_btn.enable()
        results_container.classes(add="hidden")

        try:
            # Show tracker type in progress label
            tracker_type = state.get("tracker_type", "Unknown")
            progress_label.text = f"Starting tracking ({tracker_type})..."
            progress.value = 0

            # Use already-loaded video and model from state
            local_video = state["video_path"]
            model = state["loaded_model"]

            # Frame callback for real-time UI updates
            # Capture display update interval from slider
            display_interval = int(display_update_slider.value)

            async def frame_callback(annotated_frame, frame_idx, total_frames):
                """Update UI with current frame (throttled for performance)"""
                state["current_frame"] = frame_idx

                # Update every N frames based on slider setting
                if frame_idx % display_interval != 0 and frame_idx != total_frames - 1:
                    return

                # Convert frame to bytes for display
                import cv2
                import base64
                _, buffer = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                video_display.set_source(f"data:image/jpeg;base64,{img_base64}")

                # Update progress
                progress.value = 0.1 + 0.8 * (frame_idx / total_frames)
                progress_label.text = f"Tracking: Frame {frame_idx + 1}/{total_frames}"

                # Update time slider
                time_slider.set_value(frame_idx)
                time_label.text = f"Frame: {frame_idx} / {total_frames}"

            # Build tracker config from dynamic parameter widgets
            tracker_config = {
                "skip_frames": int(skip_frames_slider.value),  # Fast-forward
            }
            # Add ByteTrack parameters from param_widgets
            for param_name, widget in param_widgets.items():
                if hasattr(widget, 'value'):
                    tracker_config[param_name] = widget.value

            # Initialize tracker with dynamic parameters
            tracker = VideoTracker(
                model=model,
                tracker_config=tracker_config,
                confidence=conf_slider.value,
                frame_callback=frame_callback,
                stop_event=state["stop_event"],
                pause_event=state["pause_event"],
                skip_frames_event=state["skip_frames_event"],
            )

            output_dir = f"/tmp/outputs/{session_id}"
            results = await tracker.process_video_realtime(str(local_video), output_dir)

            # Show results
            progress.value = 1.0
            progress_label.text = "Complete!"

            state["results"] = results

            stats_label.text = (
                f"Processed {results['stats']['total_frames']} frames | "
                f"{results['stats']['total_detections']} detections | "
                f"{results['stats']['unique_tracks']} unique tracks"
            )

            # Setup download button (only tracking CSV)
            download_track_btn.on_click(lambda: ui.download(results["tracking_csv"]))

            results_container.classes(remove="hidden")

        except Exception as e:
            logger.error(f"Tracking failed: {e}", exc_info=True)
            try:
                progress_label.text = f"Error: {str(e)}"
                ui.notify(f"Error: {str(e)}", type="negative")
            except Exception as notify_error:
                logger.error(f"Failed to show error notification: {notify_error}")
                try:
                    progress_label.text = f"Error: {str(e)}"
                except:
                    pass

        finally:
            state["processing"] = False
            state["stop_event"] = None
            state["pause_event"] = None
            state["skip_frames_event"] = None
            start_btn.enable()
            pause_btn.disable()
            stop_btn.disable()

    # Wire up buttons to event handlers (after functions are defined)
    load_video_btn.on_click(load_video)
    load_model_btn.on_click(load_model)
    start_btn.on_click(start_tracking)
    pause_btn.on_click(lambda: pause_tracking())
    stop_btn.on_click(lambda: stop_tracking())


# Run the NiceGUI app directly (no function wrapper for reload compatibility)
ui.run(
    host="0.0.0.0",
    port=int(os.getenv("PORT", 8080)),
    reload=os.getenv("NICEGUI_RELOAD", "true").lower() == "true",
    title="Tracking Studio",
)
