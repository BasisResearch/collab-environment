import torch
import torchvision
import numpy as np
import cv2
import requests
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Union
from PIL import Image
from transformers import pipeline, AutoModelForMaskGeneration, AutoProcessor
from .utils import load_hf_weights

import random
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import gc

# Recognize Anything Model & Tag2Text
from ram.models import ram
from ram import inference_ram


########################################################
########### Model Loading Utils ########################
########################################################

def load_grounding_dino_detector(detector_id: str = "IDEA-Research/grounding-dino-tiny"):
    """
    Load a Grounding DINO detector model.
    
    Args:
        detector_id: Model ID for the detector.
        
    Returns:
        Loaded detector pipeline.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return pipeline(model=detector_id, task="zero-shot-object-detection", device=device)

def load_sam_segmenter(segmenter_id: str = "syscv-community/sam-hq-vit-base"):
    """
    Load a SAM segmenter model and processor.
    
    Args:
        segmenter_id: Model ID for the segmenter.
        
    Returns:
        Tuple of (segmenter, processor).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmenter = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    processor = AutoProcessor.from_pretrained(segmenter_id)
    return segmenter, processor

def load_ram_model(model_name: str, image_size: int = 384, device: str = "cpu"):
    import ram.models as ram_models

    RAM_MODELS = {
        'ram': {
            'repo_id': "xinyu1205/recognize_anything_model",
            'filename': "ram_swin_large_14m.pth"
        },
        'ram_plus': {
            'repo_id': "xinyu1205/recognize-anything-plus-model",
            'filename': "ram_plus_swin_large_14m.pth"
        }
    }

    assert (model_name in RAM_MODELS), f"Model {model_name} not found in RAM_MODELS"

    weights_path = load_hf_weights(
        **RAM_MODELS[model_name]
    )

    model = getattr(ram_models, model_name)(pretrained=weights_path, image_size=image_size, vit='swin_l')
    model = model.eval().to(device)
    return model

########################################################
########### Recognize Anything Utils ###################
########################################################

def get_ram_transform(image_size: int = 384):
    """
    Get the transformation pipeline for RAM model input.
    
    Args:
        image_size: Size to resize images to.
        
    Returns:
        Transform pipeline for RAM input.
    """
    import torchvision.transforms as T
    
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(), 
        normalize
    ])
    
    return transform

def infer_ram_labels(
    image: Image.Image,
    ram_model: Any,
    image_size: int = 384,
    device: str = "cpu"
) -> List[str]:
    """
    Use Recognize Anything Model to automatically detect labels in an image.
    
    Args:
        image: Input image to analyze.
        ram_model: Pre-loaded RAM model.
        image_size: Size to resize image to.
        device: Device to run inference on.
        
    Returns:
        List of detected labels.
    """
    from ram import inference_ram
    
    # Prepare image for RAM
    transform = get_ram_transform(image_size)

    if isinstance(image, str):
        image = load_image(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image_resized = image.resize((image_size, image_size))
    image_input = transform(image_resized).unsqueeze(0).to(device)
    
    try:
        # Run inference
        labels = inference_ram(image_input, ram_model)
        labels = labels[0].replace(' | ', ',').split(',')
        
        # Post-process labels
        labels = postproc_labels(labels)
    finally:
        # Clean up GPU memory
        del image_input
        del image_resized
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    return labels

########################################################
########### Grounded DINO Detection Utils ##############
########################################################

@dataclass
class BoundingBox:
    """Represents a bounding box with xmin, ymin, xmax, ymax coordinates."""
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        """Return coordinates as [xmin, ymin, xmax, ymax] list."""
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    @property
    def center(self) -> Tuple[float, float]:
        """Return the center point of the bounding box as (x, y)."""
        center_x = (self.xmin + self.xmax) / 2
        center_y = (self.ymin + self.ymax) / 2
        return (center_x, center_y)

@dataclass
class DetectionResult:
    """Represents a detection result with score, label, bounding box, and optional mask."""
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        """Create DetectionResult from dictionary format."""
        return cls(
            score=detection_dict['score'],
            label=detection_dict['label'],
            box=BoundingBox(
                xmin=detection_dict['box']['xmin'],
                ymin=detection_dict['box']['ymin'],
                xmax=detection_dict['box']['xmax'],
                ymax=detection_dict['box']['ymax']
            )
        )

def postprocess_boxes(
    detections: List[DetectionResult],
    image: Image.Image,
    iou_threshold: float = 0.5
) -> List[DetectionResult]:
    """
    Postprocess bounding boxes using pixel scaling and Non-Maximum Suppression (NMS).

    Args:
        detections: Raw detection results.
        image: Original input image.
        iou_threshold: IoU threshold for NMS.

    Returns:
        Filtered and pixel-converted results.
    """
    W, H = image.size

    boxes = []
    scores = []
    phrases = []

    for det in detections:
        xmin, ymin, xmax, ymax = det.box.xyxy
        cx = (xmin + xmax) / 2 / W
        cy = (ymin + ymax) / 2 / H
        w = (xmax - xmin) / W
        h = (ymax - ymin) / H
        boxes.append([cx, cy, w, h])
        scores.append(det.score)
        phrases.append(det.label)

    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)

    # Convert to pixel coordinates
    boxes_px = boxes.clone()
    boxes_px[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * W
    boxes_px[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * H
    boxes_px[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * W
    boxes_px[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * H

    # Apply NMS
    keep = torchvision.ops.nms(boxes_px, scores, iou_threshold)

    final_results = []
    for i in keep.tolist():
        xmin, ymin, xmax, ymax = boxes_px[i].tolist()
        final_results.append(
            DetectionResult(
                score=scores[i].item(),
                label=phrases[i],
                box=BoundingBox(
                    xmin=int(xmin),
                    ymin=int(ymin),
                    xmax=int(xmax),
                    ymax=int(ymax)
                )
            )
        )

    # Clean up tensors
    del boxes, scores, boxes_px, keep
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return final_results

def detect(
    image: Image.Image,
    labels: List[str],
    detector: Any,
    threshold: float = 0.3,
    postprocess: bool = True,
    iou_threshold: float = 0.5
) -> List[DetectionResult]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    
    Args:
        image: Input image to detect objects in.
        labels: List of labels to detect.
        detector: Pre-loaded Grounding DINO detector pipeline.
        threshold: Detection confidence threshold.
        postprocess: Whether to apply postprocessing.
        iou_threshold: IoU threshold for NMS.
        
    Returns:
        List of detection results.
    """
    if isinstance(image, str):
        image = load_image(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Ensure labels end with period for better detection
    labels = [label if label.endswith(".") else label + "." for label in labels]

    results = detector(image, candidate_labels=labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]

    if not postprocess:
        return results

    results = postprocess_boxes(results, image, iou_threshold)
    return results

########################################################
########### SAM Segmentation Utils #####################
########################################################

def get_boxes(results: List[DetectionResult]) -> List[List[List[float]]]:
    """Extract bounding boxes from detection results for SAM input."""
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)
    return [boxes]

def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    """
    Refine segmentation masks with optional polygon refinement.
    
    Args:
        masks: Raw masks from SAM.
        polygon_refinement: Whether to apply polygon refinement.
        
    Returns:
        List of refined masks.
    """
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks

def segment(
    image: Image.Image,
    detection_results: List[DetectionResult],
    segmenter: AutoModelForMaskGeneration,
    processor: AutoProcessor,
    polygon_refinement: bool = False
) -> List[DetectionResult]:
    """
    Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
    
    Args:
        image: Input image.
        detection_results: Detection results with bounding boxes.
        segmenter: Pre-loaded SAM segmenter model.
        processor: Pre-loaded SAM processor.
        polygon_refinement: Whether to apply polygon refinement.
        
    Returns:
        Detection results with masks added.
    """
    device = next(segmenter.parameters()).device
    boxes = get_boxes(detection_results)
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

    try:
        outputs = segmenter(**inputs)
        masks = processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes
        )[0]

        masks = refine_masks(masks, polygon_refinement)

        for detection_result, mask in zip(detection_results, masks):
            detection_result.mask = mask

    finally:
        # Clean up GPU memory
        del inputs
        if 'outputs' in locals():
            del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return detection_results

########################################################
########### Mask Processing Utils ######################
########################################################

def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    """
    Convert a binary mask to a polygon representation.
    
    Args:
        mask: Binary mask.
        
    Returns:
        List of polygon vertices.
    """
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon

def polygon_to_mask(polygon: List[List[int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
        polygon: List of [x, y] coordinates representing the vertices of the polygon.
        image_shape: Shape of the image (height, width) for the mask.

    Returns:
        Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask

########################################################
########### Recognize Anything Utils ###################
########################################################

########################################################
########### Utility Functions ##########################
########################################################

def load_image(image_str: str) -> Image.Image:
    """
    Load an image from a file path or URL.
    
    Args:
        image_str: Path to image file or URL.
        
    Returns:
        PIL Image object.
    """
    if image_str.startswith("http"):
        image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_str).convert("RGB")

    return image

def postproc_labels(labels: List[str]) -> List[str]:
    """
    Post-process labels by splitting into individual words and removing duplicates.
    
    Args:
        labels: List of labels to process.
        
    Returns:
        Processed list of unique words.
    """
    processed = []
    for label in labels:
        # Split into separate words
        words = label.split(' ')
        processed.extend(words)

    processed = list(set(processed))
    return processed

def cleanup_gpu_memory():
    """
    Utility function to force cleanup of GPU memory.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

########################################################
########### Main Interface Function ####################
########################################################

def grounded_segmentation(
    image: Union[Image.Image, str],
    labels: List[str],
    detector: Any,
    segmenter: AutoModelForMaskGeneration,
    processor: AutoProcessor,
    threshold: float = 0.3,
    polygon_refinement: bool = False
) -> Tuple[np.ndarray, List[DetectionResult]]:
    """
    Perform grounded segmentation using Grounding DINO for detection and SAM for segmentation.
    
    Args:
        image: Input image or path to image file.
        labels: List of labels to detect and segment.
        detector: Pre-loaded Grounding DINO detector pipeline.
        segmenter: Pre-loaded SAM segmenter model.
        processor: Pre-loaded SAM processor.
        threshold: Detection confidence threshold.
        polygon_refinement: Whether to apply polygon refinement to masks.
        
    Returns:
        Tuple of (image_array, detection_results_with_masks).
    """
    if isinstance(image, str):
        image = load_image(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    try:
        detections = detect(image, labels, detector, threshold)
        detections = segment(image, detections, segmenter, processor, polygon_refinement)
        
        return image, detections
    finally:
        # Force cleanup after processing
        cleanup_gpu_memory()

def auto_detect_and_segment(
    image: Union[Image.Image, str],
    ram_model: Any,
    detector: Any,
    segmenter: AutoModelForMaskGeneration,
    processor: AutoProcessor,
    threshold: float = 0.3,
    polygon_refinement: bool = False,
    image_size: int = 384,
    device: str = "cpu"
) -> Tuple[np.ndarray, List[DetectionResult]]:
    """
    Automatically detect labels using RAM and then perform grounded segmentation.
    
    Args:
        image: Input image or path to image file.
        ram_model: Pre-loaded RAM model.
        detector: Pre-loaded Grounding DINO detector pipeline.
        segmenter: Pre-loaded SAM segmenter model.
        processor: Pre-loaded SAM processor.
        threshold: Detection confidence threshold.
        polygon_refinement: Whether to apply polygon refinement to masks.
        image_size: Size to resize image for RAM.
        device: Device to run inference on.
        
    Returns:
        Tuple of (image_array, detection_results_with_masks).
    """
    if isinstance(image, str):
        image = load_image(image)
    
    try:
        # Use RAM to automatically detect labels
        labels = infer_ram_labels(image, ram_model, image_size, device)
        
        print(f"Found {len(labels)} labels: {labels}")
        
        # Perform grounded segmentation with detected labels
        return grounded_segmentation(
            image, labels, detector, segmenter, processor, 
            threshold, polygon_refinement
        )
    finally:
        # Force cleanup after processing
        cleanup_gpu_memory()

########################################################
########### Plotting Utils ##############################
########################################################

def annotate(image: Union[Image.Image, np.ndarray], detection_results: List[DetectionResult]) -> np.ndarray:
    # Convert PIL Image to OpenCV format
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    # Iterate over detections and add bounding boxes and masks
    for detection in detection_results:
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        # Sample a random color for each detection
        color = np.random.randint(0, 256, size=3)

        # Draw bounding box
        cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2)
        cv2.putText(image_cv2, f'{label}: {score:.2f}', (box.xmin, box.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

        # If mask is available, apply it
        if mask is not None:
            # Convert mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

def plot_detections(
    image: Union[Image.Image, np.ndarray],
    detections: List[DetectionResult],
    save_name: Optional[str] = None
) -> None:
    annotated_image = annotate(image, detections)
    plt.imshow(annotated_image)
    plt.axis('off')
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()
     

def random_named_css_colors(num_colors: int) -> List[str]:
    """
    Returns a list of randomly selected named CSS colors.

    Args:
    - num_colors (int): Number of random colors to generate.

    Returns:
    - list: List of randomly selected named CSS colors.
    """
    # List of named CSS colors
    named_css_colors = [
        'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond',
        'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
        'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey',
        'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
        'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
        'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite',
        'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory',
        'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow',
        'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray',
        'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine',
        'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise',
        'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive',
        'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip',
        'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown',
        'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey',
        'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white',
        'whitesmoke', 'yellow', 'yellowgreen'
    ]

    # Sample random named CSS colors
    return random.sample(named_css_colors, min(num_colors, len(named_css_colors)))

def plot_detections_plotly(
    image: np.ndarray,
    detections: List[DetectionResult],
    class_colors: Optional[Dict[str, str]] = None
) -> None:
    # If class_colors is not provided, generate random colors for each class
    if class_colors is None:
        num_detections = len(detections)
        colors = random_named_css_colors(num_detections)
        class_colors = {}
        for i in range(num_detections):
            class_colors[i] = colors[i]


    fig = px.imshow(image)

    # Add bounding boxes
    shapes = []
    annotations = []
    for idx, detection in enumerate(detections):
        label = detection.label
        box = detection.box
        score = detection.score
        mask = detection.mask

        polygon = mask_to_polygon(mask)

        fig.add_trace(go.Scatter(
            x=[point[0] for point in polygon] + [polygon[0][0]],
            y=[point[1] for point in polygon] + [polygon[0][1]],
            mode='lines',
            line=dict(color=class_colors[idx], width=2),
            fill='toself',
            name=f"{label}: {score:.2f}"
        ))

        xmin, ymin, xmax, ymax = box.xyxy
        shape = [
            dict(
                type="rect",
                xref="x", yref="y",
                x0=xmin, y0=ymin,
                x1=xmax, y1=ymax,
                line=dict(color=class_colors[idx])
            )
        ]
        annotation = [
            dict(
                x=(xmin+xmax) // 2, y=(ymin+ymax) // 2,
                xref="x", yref="y",
                text=f"{label}: {score:.2f}",
            )
        ]

        shapes.append(shape)
        annotations.append(annotation)

    # Update layout
    button_shapes = [dict(label="None",method="relayout",args=["shapes", []])]
    button_shapes = button_shapes + [
        dict(label=f"Detection {idx+1}",method="relayout",args=["shapes", shape]) for idx, shape in enumerate(shapes)
    ]
    button_shapes = button_shapes + [dict(label="All", method="relayout", args=["shapes", sum(shapes, [])])]

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        # margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        updatemenus=[
            dict(
                type="buttons",
                direction="up",
                buttons=button_shapes
            )
        ],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Show plot
    fig.show()