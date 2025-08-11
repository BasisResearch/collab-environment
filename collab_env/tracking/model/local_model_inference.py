import os
import csv
import cv2
from PIL import Image
from rfdetr.detr import RFDETRBase  
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
import torch

def infer_with_yolo(video_path, model_path, output_csv_path="output_results.csv"):
    """
    Perform inference on a video using the YOLO model and save results to a CSV file.
    
    Args:
        video_path (str): Path to the input video file.
        model (YOLO): Loaded YOLO model for inference.
    """

    # Load the YOLO model
    model = YOLO(model_path)  # pretrained YOLO11n model # Automatically loads the model
    print("Model loaded successfully for inference!")

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")

    # Write header to CSV if it doesn't exist
    if not os.path.exists(output_csv_path):
        with open(output_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_index", "count_objects", "predictions"])

    # Process video frames
    for frame_idx in tqdm(range(total_frames), desc="Processing video frames"):
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Unable to read frame {frame_idx}")
            break

        # Preprocess the frame for the model
        input_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # Perform inference
        with torch.no_grad():
            predictions = model(input_tensor)

        # Extract results (customize based on your model's output format)
        count_objects = len(predictions)  # Example: number of detected objects
        predictions_list = [pred.tolist() for pred in predictions.xyxy[0]]  # Convert predictions to list format

    # Save results to CSV
    with open(output_csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([frame_idx, count_objects, predictions_list])

    cap.release()
    cv2.destroyAllWindows()
    print(f"Inference completed. Results saved to {output_csv_path}")



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