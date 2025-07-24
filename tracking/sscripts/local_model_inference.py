import torch
import cv2
import os
import csv
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run object detection using a locally stored model.")
    parser.add_argument("--vid_name", type=str, required=True, help="Name of the video file.")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to the video directory.")
    parser.add_argument("--model_weights", type=str, default="pipeline_code_out_of_the_box/model_weights.pt", help="Path to the model weights file.")
    args = parser.parse_args()

    # Update paths based on arguments
    vid_name = args.vid_name
    root_dir = args.root_dir
    video_path = f"{root_dir}/{vid_name}"
    output_csv_path = f"{root_dir}/{vid_name}_inference_results.csv"

    # Load the model
    model = torch.load(args.model_weights)
    model.eval()

    # Write header to CSV if it doesn't exist
    if not os.path.exists(output_csv_path):
        with open(output_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_index", "count_objects", "predictions"])

    # Open video file
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Process video frames
    for frame_idx in tqdm(range(total_frames), desc="Processing video frames"):
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame for the model
        input_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # Perform inference
        with torch.no_grad():
            predictions = model(input_tensor)

        # Extract results (customize based on your model's output format)
        count_objects = len(predictions)  # Example: number of detected objects

        # Save results to CSV
        with open(output_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([frame_idx, count_objects, predictions])

        # Optionally display the frame with detections
        for pred in predictions:
            x1, y1, x2, y2 = map(int, pred["bbox"])  # Example: bounding box coordinates
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Detections", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Inference completed. Results saved to {output_csv_path}")

if __name__ == "__main__":
    main()
