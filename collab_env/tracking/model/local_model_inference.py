import os
import csv
import cv2
from PIL import Image
from rfdetr.detr import RFDETRBase  
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
import torch
import json
import pandas as pd
import numpy as np

def infer_with_yolo(video_path, model_path, output_csv_path, output_video_path=None, show_window=True):
    model = YOLO(model_path)
    print("Model loaded successfully for inference!")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Total frames in video: {total_frames}")

    # Video writer for visualization
    if output_video_path:
        writer = cv2.VideoWriter(str(output_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    else:
        writer = None

    # Write header to CSV if it doesn't exist
    if not os.path.exists(output_csv_path):
        with open(output_csv_path, "w", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(["count_objects", "output_image", "predictions"])

    for frame_idx in tqdm(range(total_frames), desc="Processing video frames"):
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Unable to read frame {frame_idx}")
            break

        input_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        with torch.no_grad():
            results = model(input_tensor)
            result = results[0]

        count_objects = len(result.boxes)
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy.numel() > 0 else np.empty((0, 4))
        confs = result.boxes.conf.cpu().numpy() if hasattr(result.boxes, "conf") else []
        class_ids = result.boxes.cls.cpu().numpy() if hasattr(result.boxes, "cls") else []
        # If you have class names, you can map them here
        class_names = [str(int(cid)) for cid in class_ids]

        pred_list = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            width = float(abs(x2 - x1))
            height = float(abs(y2 - y1))
            x = float((x1 + x2) / 2)
            y = float((y1 + y2) / 2)
            pred = {
                "width": width,
                "height": height,
                "x": x,
                "y": y,
                "confidence": float(confs[i]) if i < len(confs) else None,
                "class_id": int(class_ids[i]) if i < len(class_ids) else None,
                "class": class_names[i] if i < len(class_names) else None,
                "detection_id": None,
                "parent_id": None
            }
            pred_list.append(pred)

        image_info = {"width": int(width), "height": int(height)}
        output_dict = {
            "image": image_info,
            "predictions": pred_list
        }
        predictions_json = json.dumps(output_dict)

        # Visualization: draw boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        if show_window:
            cv2.imshow("Detections", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if writer:
            writer.write(frame)

        # Save results to CSV
        with open(output_csv_path, "a", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([count_objects, "<deducted_image>", predictions_json])

    cap.release()
    if writer:
        writer.release()
    if show_window:
        cv2.destroyAllWindows()
    print(f"Inference completed. Results saved to {output_csv_path}")
    if writer:
        print(f"Visualization video saved to {output_video_path}")


def process_video_with_rfdetr(video_path, output_csv_path, output_video_path, checkpoint_path, confidence=0.5):
    """
    Process a video using RF-DETR with custom weights and save the results to a CSV file and an annotated video.
    Args:
        video_path (str): Path to the input video.
        output_csv_path (str): Path to save the detection results as a CSV file.
        output_video_path (str): Path to save the annotated video.
        checkpoint_path (str): Path to the custom weights file (.pt).
        confidence (float): Confidence threshold for detections.
    """
    # Load the RF-DETR model with custom weights
    print(f"🔄 Loading RF-DETR model with weights from: {checkpoint_path}")
    model = RFDETRBase(pretrain_weights=checkpoint_path)
    print("✅ Model loaded successfully.")

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer for annotated video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") # type: ignore
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write header to CSV
    with open(output_csv_path, "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["frame_index", "class_name", "confidence", "x1", "y1", "x2", "y2"])

    # Process each frame
    for frame_idx in tqdm(range(total_frames), desc="Processing video frames"):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Perform inference
        detections = model.predict(image, confidence=confidence)

        # Convert predictions to Supervision Detections
        sv_detections = sv.Detections.from_inference(detections)

        # Extract labels
        labels = [detection.class_name for detection in detections.predictions]

        # Annotate the frame
        annotated_frame = sv.BoxAnnotator().annotate(frame, sv_detections)
        annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, sv_detections, labels)

        # Write annotated frame to video
        writer.write(annotated_frame)

        # Save detections to CSV
        with open(output_csv_path, "a", newline="") as f:
            writer_csv = csv.writer(f)
            for detection in detections.predictions:
                x1, y1, x2, y2 = detection.bbox
                writer_csv.writerow([frame_idx, detection.class_name, detection.confidence, x1, y1, x2, y2])

    cap.release()
    writer.release()
    print(f"Inference completed. Results saved to {output_csv_path} and {output_video_path}")
