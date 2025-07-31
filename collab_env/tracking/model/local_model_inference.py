import os
import csv
import cv2
from PIL import Image
from rfdetr.detr import RFDETRBase  
import supervision as sv
from tqdm import tqdm

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
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
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