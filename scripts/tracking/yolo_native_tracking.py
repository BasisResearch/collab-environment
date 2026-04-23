"""Real-time tracking using YOLO's native track() method"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to YOLO model")
    parser.add_argument("path_to_video", type=str, help="Path to video file")
    parser.add_argument("--confidence", type=float, default=0.2, help="Confidence threshold")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml",
                       help="Tracker config: bytetrack.yaml, botsort.yaml, or path to custom .yaml")
    parser.add_argument("--iou", type=float, default=0.5, help="IOU threshold for NMS")
    parser.add_argument("--no-persist", action="store_true", help="Don't persist tracks between frames (default: persist=True)")

    args = parser.parse_args()

    # Load model
    model = YOLO(args.model_path)

    # Open video
    cap = cv2.VideoCapture(args.path_to_video)

    if not cap.isOpened():
        print(f"Error: Could not open video {args.path_to_video}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height} @ {fps:.2f} fps, {total_frames} frames")
    print(f"Tracker: {args.tracker}")
    print(f"Confidence: {args.confidence}, IOU: {args.iou}")
    print("\nPress 'q' to quit, 'p' to pause/unpause, SPACE to step frame when paused")

    frame_idx = 0
    paused = False
    current_frame = None

    # Calculate padding for YOLO
    target_height = ((height + 31) // 32) * 32
    target_width = ((width + 31) // 32) * 32

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\nEnd of video")
                break

            # Run YOLO tracking (track() method does detection + tracking in one step!)
            results = model.track(
                source=frame,
                conf=args.confidence,
                iou=args.iou,
                tracker=args.tracker,
                persist=not args.no_persist,  # Persist tracks between frames (True by default)
                verbose=False,
                imgsz=(target_width, target_height),
                device='mps' if hasattr(model, 'device') else 'cpu'  # Use MPS on Mac if available
            )

            # Get annotated frame with tracking visualization
            # YOLO's plot() method draws boxes, masks, and track IDs automatically
            annotated_frame = results[0].plot()

            # Add custom info overlay
            if results[0].boxes is not None and results[0].boxes.id is not None:
                n_detections = len(results[0].boxes.id)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                unique_tracks = len(np.unique(track_ids))
                info_text = f"Frame: {frame_idx}/{total_frames} | Detections: {n_detections} | Unique IDs: {unique_tracks}"
            else:
                info_text = f"Frame: {frame_idx}/{total_frames} | Detections: 0"

            cv2.putText(annotated_frame, info_text, (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            current_frame = annotated_frame
            frame_idx += 1

        # Display frame
        if current_frame is not None:
            cv2.imshow('YOLO Native Tracking', current_frame)

        # Handle key presses
        key = cv2.waitKey(1 if not paused else 0) & 0xFF

        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('p'):
            paused = not paused
            print(f"\n{'Paused' if paused else 'Resumed'}")
        elif key == ord(' ') and paused:
            # Step one frame forward when paused
            paused = False
            continue

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
