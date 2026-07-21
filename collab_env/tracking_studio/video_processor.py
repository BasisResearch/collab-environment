"""
Video Processor Component

Core detection + tracking pipeline.

Detection and tracking are decoupled: every frame runs plain
``model.predict``; in tracking mode the boxes feed a per-run Ultralytics
``BYTETracker`` we own — the exact operation ``model.track`` performs
internally, minus its two landmines: (1) it replaces results with
tracker-confirmed boxes only, so "detection only" on a model that ever ran
``.track()`` was silently tracker-filtered; (2) it registers callbacks on the
model object permanently, and every ``model.predictor = None`` reset stacked
another copy — run N of a session executed the tracker N times per frame.
Never call ``model.track`` here. This mirrors
``collab-data:collab_data/rat_sensor_pipeline/tracking.py`` so both tools
give identical results for the same video, params, and start frame.
"""

import asyncio
import threading
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import pandas as pd
from pathlib import Path
from concurrent.futures import Future
from typing import Callable, Coroutine, Dict, Optional, Union, Any
from loguru import logger


def _strip_tracker_callbacks(model) -> None:
    """Remove tracker callbacks a past ``model.track()`` registered.

    ``model.track`` permanently appends ``ultralytics.trackers.track``
    callbacks to ``model.callbacks``; they then run (and filter results!) on
    every plain ``predict`` call too. Strip them and drop predictor state so
    detection is a pure function of (weights, frame, conf).
    """
    for event in ("on_predict_start", "on_predict_postprocess_end"):
        cbs = getattr(model, "callbacks", {}).get(event)
        if cbs:
            model.callbacks[event] = [
                cb
                for cb in cbs
                if getattr(getattr(cb, "func", cb), "__module__", "")
                != "ultralytics.trackers.track"
            ]
    if hasattr(model, "predictor"):
        model.predictor = None


class VideoTracker:
    """Video tracking processor with detection and tracking"""

    def __init__(
        self,
        model: Union[YOLO, Any],  # YOLO or Roboflow model
        tracker_config: Dict,  # ByteTrack parameters
        confidence: float = 0.5,
        detection_only: bool = False,
        upscale: float = 1.0,
        display_interval: int = 10,
        frame_callback: Optional[Callable[..., Coroutine[Any, Any, None]]] = None,
        stop_event: Optional[threading.Event] = None,
        pause_event: Optional[threading.Event] = None,
        skip_frames_event: Optional[Dict] = None,
        save_csv: bool = True,
        device: str = "auto",
    ):
        """
        Initialize video tracker.

        Args:
            model: Detection model (YOLO or Roboflow)
            tracker_config: Tracker configuration dict
            confidence: Detection confidence threshold
            detection_only: If True, run detection without tracking (no track IDs)
            upscale: Multiplier on native video resolution for YOLO inference
                (1.0 = off). Implemented via predict's imgsz, so output boxes
                stay in original-frame pixel space. Ignored for Roboflow models.
            display_interval: Update display every Nth frame (1 = every frame)
            frame_callback: Async callback for frame updates (frame, frame_idx, total_frames)
            stop_event: Threading event to signal hard stop
            pause_event: Threading event to signal pause/resume
            skip_frames_event: Dict with skip_amount for forward seeking
            save_csv: If True, write detection/tracking results to CSV at end
            device: torch device for YOLO predict; "auto" resolves cuda > mps >
                cpu, matching the rat-sensor pipeline (ultralytics' own default
                is cpu on Macs, which would make results differ across tools)
        """
        self.model = model
        self.device = self._resolve_device(device)
        self.confidence = confidence
        self.detection_only = detection_only
        self.upscale = max(1.0, float(upscale))
        self.display_interval = max(1, display_interval)
        self.frame_callback = frame_callback
        self.stop_event = stop_event or threading.Event()
        self.pause_event = pause_event or threading.Event()
        self.skip_frames_event = skip_frames_event or {"skip_amount": 0}
        self.save_csv = save_csv
        self._pending_update: Optional[Future[None]] = None

        # YOLO models run plain predict + our own BYTETracker; anything else
        # falls back to Roboflow inference + supervision ByteTrack.
        self.is_yolo_model = isinstance(model, YOLO)

        if self.upscale > 1.0 and not self.is_yolo_model:
            logger.warning(
                f"Upscale {self.upscale}x ignored: the Roboflow inference server "
                "letterboxes input to the model's fixed size, so pre-upscaling "
                "has no effect. Use a local YOLO model for upscaled inference."
            )

        if self.is_yolo_model:
            # Undo any contamination from a past model.track() on this model
            # object (earlier studio sessions/tools) — see module docstring.
            _strip_tracker_callbacks(model)

        if detection_only:
            logger.info("Detection-only mode (no tracking)")
            self.tracker = None
        elif not self.is_yolo_model:
            # For Roboflow inference models (fallback), initialize supervision tracker
            logger.info(
                "Using supervision ByteTrack (Roboflow inference model fallback)"
            )
            self.tracker = sv.ByteTrack(
                track_activation_threshold=tracker_config.get(
                    "track_high_thresh", 0.25
                ),
                lost_track_buffer=tracker_config.get("track_buffer", 30),
                minimum_matching_threshold=tracker_config.get("match_thresh", 0.8),
                minimum_consecutive_frames=1,
                frame_rate=30,
            )
        else:
            logger.info("Using Ultralytics ByteTrack (decoupled from detection)")
            self.tracker = self._make_bytetracker(tracker_config)

        # Fast-forward: Skip frames for faster preview
        self.skip_frames = tracker_config.get(
            "skip_frames", 1
        )  # 1 = process every frame

        logger.info(
            f"VideoTracker initialized (confidence: {self.confidence}, upscale: {self.upscale}, yolo_model: {self.is_yolo_model})"
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve "auto" to cuda/mps/cpu — same rule as the rat-sensor
        pipeline's ``resolve_device`` so both tools run the model identically."""
        if device != "auto":
            return device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            return "mps"
        return "cpu"

    @staticmethod
    def _make_bytetracker(config: Dict):
        """Build a fresh Ultralytics BYTETracker (fresh state, ids from 1)."""
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace

        args = IterableSimpleNamespace(
            tracker_type="bytetrack",
            track_high_thresh=config.get("track_high_thresh", 0.25),
            track_low_thresh=config.get("track_low_thresh", 0.1),
            new_track_thresh=config.get("new_track_thresh", 0.25),
            track_buffer=config.get("track_buffer", 30),
            match_thresh=config.get("match_thresh", 0.8),
            fuse_score=config.get("fuse_score", True),
        )
        logger.debug(f"ByteTrack config: {vars(args)}")
        return BYTETracker(args=args)

    def _process_video_sync(
        self, video_path: str, output_dir: str, event_loop, start_frame: int = 0
    ) -> Dict[str, Any]:
        """
        Synchronous video processing function (runs in background thread).

        Args:
            video_path: Path to input video
            output_dir: Directory for output CSV
            event_loop: Main asyncio event loop for scheduling UI updates
            start_frame: Frame index to start processing from (0-based)

        Returns:
            Dict with output_csv path (or None if save_csv=False) and stats
        """
        logger.info(f"Processing video in background thread: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Video info: {total_frames} frames, {fps} fps, {width}x{height}")

        # Upscale = raise YOLO inference resolution via imgsz. predict()
        # letterboxes the frame up internally and scale_boxes() maps output
        # coords back to the ORIGINAL frame space, so no coordinate
        # conversion is needed anywhere downstream. imgsz must be a multiple
        # of the model stride (32); round up ourselves to avoid per-frame
        # warnings. When upscale == 1.0 the kwargs stay empty and behavior
        # is identical to before (ultralytics default imgsz).
        predict_kwargs: Dict[str, Any] = {}
        if self.upscale > 1.0 and self.is_yolo_model:
            imgsz = int(np.ceil(max(width, height) * self.upscale / 32) * 32)
            predict_kwargs["imgsz"] = imgsz
            logger.info(
                f"Upscale {self.upscale}x: YOLO inference at imgsz={imgsz} "
                f"(native max dim {max(width, height)})"
            )

        tracking_list = []  # one entry per tracked detection (_bboxes.csv format)
        raw_list = []  # one entry per RAW detection (track_id -1 = suppressed)
        total_detection_count = 0

        frame_idx = start_frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            logger.info(f"Starting from frame {start_frame}")
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

            # 1. Run detection (always plain predict), then optionally feed the
            #    boxes to our own tracker — never model.track (module docstring).
            try:
                det_to_track: Dict[int, int] = {}
                if self.is_yolo_model:
                    results = self.model.predict(
                        source=frame,
                        conf=self.confidence,
                        device=self.device,
                        verbose=False,
                        **predict_kwargs,
                    )[0]
                    detections = sv.Detections.from_ultralytics(results)
                    if self.detection_only:
                        tracked_detections = detections
                    else:
                        # Exactly what ultralytics' on_predict_postprocess_end
                        # does, without replacing the detector results.
                        # STrack.result: x1, y1, x2, y2, track_id, score, cls, det_idx
                        assert self.tracker is not None and results.boxes is not None
                        tracker: Any = self.tracker  # ultralytics BYTETracker
                        boxes_np = results.boxes.cpu().numpy()
                        tracks = tracker.update(boxes_np, frame)
                        if len(tracks) > 0:
                            arr = np.asarray(tracks, dtype=np.float32)
                            # model.track clipped its output boxes to image
                            # bounds (Results.update -> ops.clip_boxes);
                            # Kalman extrapolation can exceed them. Match it.
                            h, w = frame.shape[:2]
                            arr[:, [0, 2]] = arr[:, [0, 2]].clip(0, w)
                            arr[:, [1, 3]] = arr[:, [1, 3]].clip(0, h)
                            # STrack det_idx (t[-1]) is LOCAL to the high/low
                            # confidence branch BYTETracker split detections
                            # into before appending indices — NOT a global
                            # detection index. Rebuild the branch masks and
                            # resolve the branch from the row's score (copied
                            # verbatim from the matched detection). Mirrors
                            # collab-data rat_sensor_pipeline/tracking.py.
                            conf_np = np.asarray(boxes_np.conf, dtype=float)
                            hi_t = tracker.args.track_high_thresh
                            lo_t = tracker.args.track_low_thresh
                            hi_mask = conf_np >= hi_t
                            branch_idx = {
                                True: np.flatnonzero(hi_mask),
                                False: np.flatnonzero((conf_np > lo_t) & ~hi_mask),
                            }
                            det_to_track = {}
                            for t in tracks:
                                branch = branch_idx[bool(t[5] >= hi_t)]
                                k = int(t[-1])
                                if 0 <= k < len(branch):
                                    det_to_track[int(branch[k])] = int(t[4])
                            tracked_detections = sv.Detections(
                                xyxy=arr[:, :4],
                                confidence=arr[:, 5],
                                class_id=arr[:, 6].astype(int),
                                tracker_id=arr[:, 4].astype(int),
                            )
                        else:
                            tracked_detections = sv.Detections.empty()
                elif self.detection_only:
                    results = self.model.infer(frame, confidence=self.confidence)[0]
                    detections = sv.Detections.from_inference(results)
                    tracked_detections = detections
                else:
                    # Roboflow inference model (fallback to supervision ByteTrack)
                    logger.debug(f"Running Roboflow inference on frame {frame_idx}...")
                    results = self.model.infer(frame, confidence=self.confidence)[0]
                    detections = sv.Detections.from_inference(results)
                    logger.debug(f"Frame {frame_idx}: {len(detections)} detections")

                    # Update tracker (adds track IDs via supervision ByteTrack)
                    assert self.tracker is not None
                    tracked_detections = self.tracker.update_with_detections(detections)
                    # update_with_detections stamps a full-length tracker_id
                    # array (-1 = unmatched) on the INPUT detections before
                    # slicing its return value; use it to label raw rows. If a
                    # future supervision stops mutating the input, tracker_id
                    # stays None and raw rows degrade to all -1.
                    if detections.tracker_id is not None and len(
                        detections.tracker_id
                    ) == len(detections):
                        det_to_track = {
                            k: int(tid)
                            for k, tid in enumerate(detections.tracker_id)
                            if int(tid) != -1
                        }

            except Exception as e:
                logger.error(
                    f"Detection/tracking failed on frame {frame_idx}: {e}",
                    exc_info=True,
                )
                detections = sv.Detections.empty()
                tracked_detections = sv.Detections.empty()
                det_to_track = {}

            # 2. Save ALL raw detections (track_id -1 where the tracker
            #    suppressed the detection; all -1 in detection-only mode).
            if (
                detections.confidence is not None
                and detections.class_id is not None
                and len(detections) > 0
            ):
                for k, (bbox, conf, class_id) in enumerate(
                    zip(detections.xyxy, detections.confidence, detections.class_id)
                ):
                    raw_list.append(
                        {
                            "track_id": det_to_track.get(k, -1),
                            "frame": frame_idx,
                            "x1": float(bbox[0]),
                            "y1": float(bbox[1]),
                            "x2": float(bbox[2]),
                            "y2": float(bbox[3]),
                            "class": int(class_id),
                            "confidence": float(conf),
                        }
                    )

            # 3. Save tracking data (with track IDs if tracking is enabled)
            if (
                not self.detection_only
                and tracked_detections.tracker_id is not None
                and len(tracked_detections) > 0
            ):
                confidences: Any = (
                    tracked_detections.confidence
                    if tracked_detections.confidence is not None
                    else []
                )
                class_ids: Any = (
                    tracked_detections.class_id
                    if tracked_detections.class_id is not None
                    else []
                )
                for bbox, track_id, conf, class_id in zip(
                    tracked_detections.xyxy,
                    tracked_detections.tracker_id,
                    confidences,
                    class_ids,
                ):
                    tracking_list.append(
                        {
                            "track_id": int(track_id),
                            "frame": frame_idx,
                            "x1": float(bbox[0]),
                            "y1": float(bbox[1]),
                            "x2": float(bbox[2]),
                            "y2": float(bbox[3]),
                            "class": int(class_id),
                            "confidence": float(conf),
                        }
                    )
                total_detection_count += len(tracked_detections)
            elif self.detection_only:
                total_detection_count += len(detections)

            # 4. Send frame + detections to UI for display
            is_last = frame_idx >= total_frames - 1
            should_display = (frame_idx % self.display_interval == 0) or is_last

            if should_display and self.frame_callback is not None and event_loop:
                # Skip if previous UI update is still in-flight (prevents queue buildup)
                if self._pending_update is None or self._pending_update.done():
                    self._pending_update = asyncio.run_coroutine_threadsafe(
                        self.frame_callback(
                            frame, tracked_detections, frame_idx, total_frames
                        ),
                        event_loop,
                    )

            # Increment frame counter for next iteration
            frame_idx += 1

        cap.release()

        unique_tracks = (
            len(set(t["track_id"] for t in tracking_list)) if tracking_list else 0
        )
        logger.info(
            f"Processing complete: {total_frames} frames, "
            f"{total_detection_count} detections ({len(raw_list)} raw), "
            f"{unique_tracks} unique tracks"
        )

        # Same column order as the rat-sensor pipeline's BBOX_CSV_COLUMNS, so
        # studio and pipeline outputs are directly comparable.
        bbox_cols = ["track_id", "frame", "x1", "y1", "x2", "y2", "class", "confidence"]

        def _write_bboxes(rows, path):
            if len(rows) > 0:
                pd.DataFrame(rows)[bbox_cols].to_csv(path, index=False)
            else:
                pd.DataFrame(columns=bbox_cols).to_csv(path, index=False)

        output_csv: Optional[Path] = None
        output_raw_csv: Optional[Path] = None
        if self.save_csv:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Raw detections always (track_id -1 = suppressed / no tracking).
            output_raw_csv = output_path / "raw_bboxes.csv"
            _write_bboxes(raw_list, output_raw_csv)
            logger.info(f"Saved raw_bboxes.csv to {output_raw_csv}")

            if self.detection_only:
                output_csv = output_raw_csv
            else:
                # _bboxes.csv format: one row per tracked detection
                output_csv = output_path / "tracked_bboxes.csv"
                _write_bboxes(tracking_list, output_csv)
                logger.info(f"Saved tracked_bboxes.csv to {output_csv}")

        return {
            "output_csv": str(output_csv) if output_csv else None,
            "output_raw_csv": str(output_raw_csv) if output_raw_csv else None,
            "stats": {
                "total_frames": total_frames,
                "total_detections": total_detection_count,
                "total_raw_detections": len(raw_list),
                "unique_tracks": unique_tracks,
                "fps": fps,
            },
        }

    async def process_video_realtime(
        self, video_path: str, output_dir: str, start_frame: int = 0
    ) -> Dict[str, Any]:
        """
        Process video frame-by-frame with real-time UI updates.

        This runs the heavy processing in a background thread to prevent
        blocking the asyncio event loop and WebSocket connections.

        Args:
            video_path: Path to input video
            output_dir: Directory for output CSV
            start_frame: Frame index to start processing from (0-based)

        Returns:
            Dict with output_csv path (or None if save_csv=False) and stats
        """
        # Get current event loop for scheduling UI updates from background thread
        loop = asyncio.get_running_loop()

        # Run processing in background thread
        logger.info(
            f"Starting video processing in background thread (frame {start_frame})..."
        )
        result = await asyncio.to_thread(
            self._process_video_sync, video_path, output_dir, loop, start_frame
        )

        logger.info("Video processing complete")
        return result
