Downloading Training Images and Training Models
=============================================

This section explains how to download training images from the cloud and train various models (YOLO, RF-DETR) using either the Roboflow API or local scripts.

Downloading Training Images
---------------------------
To access training images please download the zip file from the roboflow_model bucket in the google cloud. 
The bucket should contain a .pt file containing the model weights for a YOLOv11 and rf-detr model and a zip file containing the images, labels, and annotations in YOLO v7 PyTorch format, as well as in the COCO format. The default setting for the local_model_inference script is YOLO. 


### Downloading a Zip File from Google Cloud
1. **Install the Google Cloud CLI**:
   If you haven't already, install the Google Cloud CLI by following the instructions [here](https://cloud.google.com/sdk/docs/install).

2. **Authenticate with Google Cloud**:
   ```bash
   gcloud auth login
   ```

3. **Download the Zip File**:
   Use the `gsutil` command to download the zip file containing images, labels, and annotations:
   ```bash
   gsutil cp gs://roboflow_model/Bird tracking.v11i.yolov7pytorch.zip ./
   ```

4. **Extract the Zip File**:
   After downloading, extract the contents:
   ```bash
   unzip "Bird tracking.v11i.yolov7pytorch.zip" -d model_training
   ```

   Ensure the extracted dataset is organized in the required format (e.g., YOLO or COCO) for training.

Training Models
---------------

### Training a YOLOv8 Model

1. **Prepare the Dataset**:
   Ensure the dataset is downloaded and organized in the YOLO format (with `images/` and `labels/` directories).

2. **Train the Model**:
   Use the YOLOv8 training script:
   ```bash
   yolo task=detect mode=train data=path/to/dataset.yaml model=yolov8n.pt epochs=50 imgsz=640
   ```

3. **Deploy the Model Using Roboflow**:
   ```python
   workspace.deploy_model(
       model_type="yolov8",
       model_path="./runs/train/weights",
       project_ids=["project-1", "project-2", "project-3"],
       model_name="my-custom-model"
   )
   ```

### Training an RF-DETR Model

1. **Prepare the Dataset**:
   Download the dataset in a format compatible with RF-DETR (e.g., COCO format).

2. **Train the Model**:
   Use the RF-DETR training script:
   ```bash
   python train.py --dataset path/to/dataset --epochs 50 --batch-size 16
   ```

3. **Deploy the Model Using Roboflow**:
   ```python
   workspace.deploy_model(
       model_type="rf-detr",
       model_path="./runs/train/weights",
       project_ids=["project-1", "project-2", "project-3"],
       model_name="my-custom-rf-detr-model"
   )
   ```

### Training Models Locally

If you prefer to train models locally without using Roboflow, follow these steps:

#### Training a YOLOv8 Model Locally

1. **Prepare the Dataset**:
   Ensure the dataset is downloaded and organized in the YOLO format (with `images/` and `labels/` directories).

2. **Install YOLOv8 Dependencies**:
   Install the required dependencies for YOLOv8:
   ```bash
   pip install ultralytics
   ```

3. **Train the Model**:
   Use the YOLOv8 training script:
   ```bash
   yolo task=detect mode=train data=path/to/dataset.yaml model=yolov8n.pt epochs=50 imgsz=640
   ```

4. **Evaluate the Model**:
   After training, evaluate the model's performance using the validation dataset:
   ```bash
   yolo task=detect mode=val data=path/to/dataset.yaml model=path/to/best.pt
   ```

#### Training an RF-DETR Model Locally

1. **Prepare the Dataset**:
   Download the dataset in a format compatible with RF-DETR (e.g., COCO format).

2. **Install RF-DETR Dependencies**:
   Ensure you have the required dependencies installed for RF-DETR. Refer to the RF-DETR repository for installation instructions.

3. **Train the Model**:
   Use the RF-DETR training script:
   ```bash
   python train.py --dataset path/to/dataset --epochs 50 --batch-size 16
   ```

4. **Evaluate the Model**:
   After training, evaluate the model's performance using the validation dataset:
   ```bash
   python evaluate.py --dataset path/to/validation_dataset --weights path/to/best_weights.pt
   ```

Notes
-----
- Replace `YOUR_API_KEY`, `YOUR_WORKSPACE`, `YOUR_PROJECT_NAME`, and `YOUR_VERSION_NUMBER` with your actual Roboflow credentials and project details if you want to use their serverless API or web app to add more to the dataset or use the weights as a checkpoint. 
- Ensure you have the necessary dependencies installed for YOLO and RF-DETR training.
