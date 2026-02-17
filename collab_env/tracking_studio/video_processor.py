"""
Video Processor Component

Core tracking pipeline with ByteTrack.
"""

import asyncio
import threading
import cv2
import supervision as sv
from ultralytics import YOLO
import pandas as pd
from pathlib import Path
from typing import Callable, Dict, List, Union, Any
import numpy as np
import tempfile
import yaml
from loguru import logger


class VideoTracker:
    """Video tracking processor with detection and tracking"""

    def __init__(
        self,
        model: Union[YOLO, Any],  # YOLO or Roboflow model
        tracker_config: Dict,  # ByteTrack parameters
        confidence: float = 0.5,
        detection_only: bool = False,
        display_interval: int = 10,
        frame_callback: Callable[[np.ndarray, int, int], None] = None,
        stop_event: threading.Event = None,
        pause_event: threading.Event = None,
        skip_frames_event: Dict = None,
    ):
        """
        Initialize video tracker.

        Args:
            model: Detection model (YOLO or Roboflow)
            tracker_config: Tracker configuration dict
            confidence: Detection confidence threshold
            detection_only: If True, run detection without tracking (no track IDs)
            display_interval: Update display every Nth frame (1 = every frame)
            frame_callback: Async callback for frame updates (frame, frame_idx, total_frames)
            stop_event: Threading event to signal hard stop
            pause_event: Threading event to signal pause/resume
            skip_frames_event: Dict with skip_amount for forward seeking
        """
        self.model = model
        self.confidence = confidence
        self.detection_only = detection_only
        self.display_interval = max(1, display_interval)
        self.frame_callback = frame_callback
        self.stop_event = stop_event or threading.Event()
        self.pause_event = pause_event or threading.Event()
        self.skip_frames_event = skip_frames_event or {"skip_amount": 0}
        self._pending_update = None  # Track in-flight UI update

        # Store tracker config for use with model.track()
        self.tracker_config = tracker_config

        # Check if model supports native tracking
        self.use_native_tracking = isinstance(model, YOLO)

        if detection_only:
            logger.info("Detection-only mode (no tracking)")
            self.tracker = None
            self.tracker_yaml_path = None
        elif not self.use_native_tracking:
            # For Roboflow inference models (fallback), initialize supervision tracker
            logger.info("Using supervision ByteTrack (Roboflow inference model fallback)")
            self.tracker = sv.ByteTrack(
                track_activation_threshold=tracker_config.get("track_high_thresh", 0.25),
                lost_track_buffer=tracker_config.get("track_buffer", 30),
                minimum_matching_threshold=tracker_config.get("match_thresh", 0.8),
                minimum_consecutive_frames=1,
                frame_rate=30,
            )
            self.tracker_yaml_path = None
        else:
            logger.info("Using Ultralytics native ByteTrack (supports all parameters)")
            self.tracker = None
            # Create temporary ByteTrack YAML config from parameters
            self.tracker_yaml_path = self._create_bytetrack_config(tracker_config)

        # Fast-forward: Skip frames for faster preview
        self.skip_frames = tracker_config.get("skip_frames", 1)  # 1 = process every frame

        logger.info(
            f"VideoTracker initialized (confidence: {self.confidence}, native_tracking: {self.use_native_tracking})"
        )

    def _create_bytetrack_config(self, config: Dict) -> str:
        """
        Create a temporary ByteTrack YAML config file from parameters.

        Args:
            config: Tracker configuration dict

        Returns:
            Path to temporary YAML config file
        """
        # Map our parameter names to Ultralytics ByteTrack YAML format
        bytetrack_yaml = {
            "tracker_type": "bytetrack",
            "track_high_thresh": config.get("track_high_thresh", 0.25),
            "track_low_thresh": config.get("track_low_thresh", 0.1),
            "new_track_thresh": config.get("new_track_thresh", 0.25),
            "track_buffer": config.get("track_buffer", 30),
            "match_thresh": config.get("match_thresh", 0.8),
            "fuse_score": config.get("fuse_score", True),
        }

        # Create temporary YAML file
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.yaml',
            delete=False,
            prefix='bytetrack_'
        )

        with temp_file as f:
            yaml.dump(bytetrack_yaml, f, default_flow_style=False)

        logger.info(f"Created ByteTrack config: {temp_file.name}")
        logger.debug(f"Config values: {bytetrack_yaml}")

        return temp_file.name

    def _process_video_sync(
        self, video_path: str, output_dir: str, event_loop
    ) -> Dict[str, Any]:
        """
        Synchronous video processing function (runs in background thread).

        Args:
            video_path: Path to input video
            output_dir: Directory for output CSV
            event_loop: Main asyncio event loop for scheduling UI updates

        Returns:
            Dict with tracking_csv path and stats
        """
        logger.info(f"Processing video in background thread: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            f"Video info: {total_frames} frames, {fps} fps, {width}x{height}"
        )

        detections_list = []
        tracking_list = []

        frame_idx = 0
        while frame_idx < total_frames:
            # Check if stop was requested (hard stop)
            if self.stop_event.is_set():
                logger.info(f"Stop requested at frame {frame_idx}, stopping processing")
                break

            # Check if pause was requested
            while self.pause_event.is_set():
                import time
                time.sleep(0.1)  # Wait while paused
                if self.stop_event.is_set():
                    break

            # Check if seek was requested (forward or backward)
            if self.skip_frames_event["skip_amount"] != 0:
                skip_to = frame_idx + self.skip_frames_event["skip_amount"]
                # Clamp to valid range
                skip_to = max(0, min(skip_to, total_frames - 1))
                logger.info(f"Seeking from frame {frame_idx} to {skip_to}")
                cap.set(cv2.CAP_PROP_POS_FRAMES, skip_to)
                self.skip_frames_event["skip_amount"] = 0  # Reset
                frame_idx = skip_to
                continue

            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame {frame_idx}, stopping")
                break

            # Fast-forward: Skip frames if requested
            if self.skip_frames > 1 and frame_idx % self.skip_frames != 0:
                frame_idx += 1
                continue

            # 1. Run detection (and optionally tracking)
            try:
                if self.detection_only:
                    # Detection only - no tracking
                    if self.use_native_tracking:
                        results = self.model(
                            source=frame,
                            conf=self.confidence,
                            verbose=False,
                        )[0]
                        detections = sv.Detections.from_ultralytics(results)
                    else:
                        results = self.model.infer(frame, confidence=self.confidence)[0]
                        detections = sv.Detections.from_inference(results)
                    tracked_detections = detections

                elif self.use_native_tracking:
                    # Use Ultralytics native tracking (supports all ByteTrack parameters)
                    results = self.model.track(
                        source=frame,
                        conf=self.confidence,
                        persist=True,  # Maintain track IDs across frames
                        tracker=self.tracker_yaml_path,  # Custom ByteTrack config
                        verbose=False,
                    )[0]

                    # Convert to supervision Detections (with track IDs)
                    tracked_detections = sv.Detections.from_ultralytics(results)

                    # Also get detections without tracking for stats
                    detections = tracked_detections
                else:
                    # Roboflow inference model (fallback to supervision ByteTrack)
                    logger.debug(f"Running Roboflow inference on frame {frame_idx}...")
                    results = self.model.infer(frame, confidence=self.confidence)[0]
                    detections = sv.Detections.from_inference(results)
                    logger.debug(f"Frame {frame_idx}: {len(detections)} detections")

                    # Update tracker (adds track IDs via supervision ByteTrack)
                    tracked_detections = self.tracker.update_with_detections(detections)

            except Exception as e:
                logger.error(f"Detection/tracking failed on frame {frame_idx}: {e}", exc_info=True)
                detections = sv.Detections.empty()
                tracked_detections = sv.Detections.empty()

            # 2. Save detections
            for i, (bbox, conf, class_id) in enumerate(
                zip(detections.xyxy, detections.confidence, detections.class_id)
            ):
                detections_list.append(
                    {
                        "frame": frame_idx,
                        "x1": bbox[0],
                        "y1": bbox[1],
                        "x2": bbox[2],
                        "y2": bbox[3],
                        "confidence": conf,
                        "class": class_id,
                    }
                )

            # 3. Save tracking data (with track IDs if tracking is enabled)
            if not self.detection_only and tracked_detections.tracker_id is not None and len(tracked_detections) > 0:
                for bbox, track_id, conf, class_id in zip(
                    tracked_detections.xyxy,
                    tracked_detections.tracker_id,
                    tracked_detections.confidence,
                    tracked_detections.class_id,
                ):
                    tracking_list.append(
                        {
                            "track_id": int(track_id),
                            "frame": frame_idx,
                            "x1": int(bbox[0]),
                            "y1": int(bbox[1]),
                            "x2": int(bbox[2]),
                            "y2": int(bbox[3]),
                            "confidence": float(conf),
                            "class": int(class_id),
                        }
                    )

            # 4. Send frame + detections to UI for display
            is_last = (frame_idx >= total_frames - 1)
            should_display = (frame_idx % self.display_interval == 0) or is_last

            if should_display and self.frame_callback and event_loop:
                # Skip if previous UI update is still in-flight (prevents queue buildup)
                if self._pending_update is None or self._pending_update.done():
                    self._pending_update = asyncio.run_coroutine_threadsafe(
                        self.frame_callback(
                            frame, tracked_detections, frame_idx, total_frames
                        ),
                        event_loop
                    )

            # Increment frame counter for next iteration
            frame_idx += 1

        cap.release()

        # Cleanup temporary tracker config file if created
        if self.tracker_yaml_path:
            try:
                import os
                os.unlink(self.tracker_yaml_path)
                logger.debug(f"Cleaned up temporary tracker config: {self.tracker_yaml_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup tracker config: {e}")

        unique_tracks = len(set(t['track_id'] for t in tracking_list)) if tracking_list else 0
        logger.info(
            f"Processing complete: {total_frames} frames, "
            f"{len(detections_list)} detections, "
            f"{unique_tracks} unique tracks"
        )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.detection_only:
            # Save detections CSV (no track IDs)
            output_csv = output_path / "detections.csv"
            if len(detections_list) > 0:
                det_df = pd.DataFrame(detections_list)
                det_df = det_df[["frame", "x1", "y1", "x2", "y2", "confidence", "class"]]
                det_df.to_csv(output_csv, index=False)
            else:
                pd.DataFrame(
                    columns=["frame", "x1", "y1", "x2", "y2", "confidence", "class"]
                ).to_csv(output_csv, index=False)
            logger.info(f"Saved detections CSV to {output_csv}")
        else:
            # Save tracking CSV (with track IDs)
            output_csv = output_path / "tracking.csv"
            if len(tracking_list) > 0:
                tracking_df = pd.DataFrame(tracking_list)
                tracking_df = tracking_df[
                    ["track_id", "frame", "x1", "y1", "x2", "y2", "confidence", "class"]
                ]
                tracking_df.to_csv(output_csv, index=False)
            else:
                pd.DataFrame(
                    columns=["track_id", "frame", "x1", "y1", "x2", "y2", "confidence", "class"]
                ).to_csv(output_csv, index=False)
            logger.info(f"Saved tracking CSV to {output_csv}")

        return {
            "tracking_csv": str(output_csv),
            "stats": {
                "total_frames": total_frames,
                "total_detections": len(detections_list),
                "unique_tracks": unique_tracks,
                "fps": fps,
            },
        }

    async def process_video_realtime(
        self, video_path: str, output_dir: str
    ) -> Dict[str, Any]:
        """
        Process video frame-by-frame with real-time UI updates.

        This runs the heavy processing in a background thread to prevent
        blocking the asyncio event loop and WebSocket connections.

        Args:
            video_path: Path to input video
            output_dir: Directory for output CSV

        Returns:
            Dict with tracking_csv path and stats
        """
        # Get current event loop for scheduling UI updates from background thread
        loop = asyncio.get_running_loop()

        # Run processing in background thread
        logger.info("Starting video processing in background thread...")
        result = await asyncio.to_thread(
            self._process_video_sync, video_path, output_dir, loop
        )

        logger.info("Video processing complete")
        return result
