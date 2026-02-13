"""Real-time ByteTracker inference on video with live visualization"""

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to YOLO model")
    parser.add_argument("path_to_video", type=str, help="Path to video file")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--track_activation", type=float, default=0.2, help="Track activation threshold")
    parser.add_argument("--lost_buffer", type=int, default=90, help="Lost track buffer frames")
    parser.add_argument("--match_threshold", type=float, default=0.8, help="Minimum matching threshold")
    parser.add_argument("--min_frames", type=int, default=5, help="Minimum consecutive frames")

    args = parser.parse_args()

    # Load model
    model = YOLO(args.model_path)

    # Initialize tracker
    tracker = sv.ByteTrack(
        track_activation_threshold=args.track_activation,
        lost_track_buffer=args.lost_buffer,
        minimum_matching_threshold=args.match_threshold,
        minimum_consecutive_frames=args.min_frames
    )

    # Initialize annotators
    box_annotator = sv.BoxAnnotator(thickness=1)
    mask_annotator = sv.MaskAnnotator(opacity=0.4)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.3,
        text_thickness=1,
        text_padding=3,
        text_position=sv.Position.TOP_LEFT,
        color=sv.Color.BLACK,
        text_color=sv.Color.WHITE,
        border_radius=2,
        smart_position=True
    )

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
    print(f"Confidence: {args.confidence}")
    print(f"ByteTracker params: activation={args.track_activation}, buffer={args.lost_buffer}, match={args.match_threshold}, min_frames={args.min_frames}")
    print("\nPress 'q' to quit, 'p' to pause/unpause, SPACE to step frame when paused")

    frame_idx = 0
    paused = False

    # Calculate padding for YOLO
    target_height = ((height + 31) // 32) * 32
    target_width = ((width + 31) // 32) * 32

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\nEnd of video")
                break

            # Run YOLO detection
            results = model.predict(
                source=frame,
                conf=args.confidence,
                verbose=False,
                imgsz=(target_width, target_height)
            )

            # Process detections
            if results and results[0].boxes:
                boxes = results[0].boxes
                masks = results[0].masks

                # Resize masks to original frame size
                if masks is not None:
                    mask_array = masks.data.cpu().numpy()
                    resized_masks = np.zeros((mask_array.shape[0], height, width))
                    for i in range(mask_array.shape[0]):
                        resized_masks[i] = cv2.resize(
                            mask_array[i],
                            (width, height),
                            interpolation=cv2.INTER_LINEAR
                        )
                    resized_masks = resized_masks > 0.5

                    # Create detections with masks
                    detections = sv.Detections(
                        xyxy=boxes.xyxy.cpu().numpy(),
                        mask=resized_masks,
                        confidence=boxes.conf.cpu().numpy(),
                        class_id=boxes.cls.cpu().numpy().astype(np.int32),
                    )
                else:
                    # No masks, just boxes
                    detections = sv.Detections(
                        xyxy=boxes.xyxy.cpu().numpy(),
                        confidence=boxes.conf.cpu().numpy(),
                        class_id=boxes.cls.cpu().numpy().astype(np.int32),
                    )

                # Update tracker
                detections = tracker.update_with_detections(detections)

                # Create labels
                labels = [
                    f"#{int(tid)} ({conf:.2f})"
                    for tid, conf in zip(detections.tracker_id, detections.confidence)
                ]

                # Annotate frame
                annotated_frame = frame.copy()
                if detections.mask is not None:
                    annotated_frame = mask_annotator.annotate(annotated_frame, detections=detections)
                else:
                    annotated_frame = box_annotator.annotate(annotated_frame, detections=detections)
                annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)

                # Add info overlay
                info_text = f"Frame: {frame_idx}/{total_frames} | Detections: {len(detections)}"
                cv2.putText(annotated_frame, info_text, (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                annotated_frame = frame.copy()
                info_text = f"Frame: {frame_idx}/{total_frames} | Detections: 0"
                cv2.putText(annotated_frame, info_text, (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            frame_idx += 1

        # Display frame
        cv2.imshow('ByteTracker Inference', annotated_frame)

        # Handle key presses
        key = cv2.waitKey(1 if not paused else 0) & 0xFF

        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('p'):
            paused = not paused
            print(f"\n{'Paused' if paused else 'Resumed'}")
        elif key == ord(' ') and paused:
            # Step one frame forward
            ret, frame = cap.read()
            if ret:
                frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
