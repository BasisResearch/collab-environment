import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List

def plot_yolo_results(image_path, detections, class_names=None, conf_threshold=0.5):
    """
    Plot original image with YOLO bounding boxes
    
    Args:
        image_path: Path to the original image
        detections: YOLO detections in one of these formats:
                   - List of [x1, y1, x2, y2, confidence, class_id] (xyxy format)
                   - List of [x_center, y_center, width, height, confidence, class_id] (xywh format)
                   - Ultralytics Results object
        class_names: List of class names (optional)
        conf_threshold: Confidence threshold for displaying boxes
    """
    
    # Load and display image
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img)
    
    # Handle different detection formats
    if hasattr(detections, 'boxes'):  # Ultralytics Results object
        boxes = detections.boxes
        if boxes is not None:
            # Convert to numpy arrays
            xyxy = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            conf = boxes.conf.cpu().numpy()  # confidence scores
            cls = boxes.cls.cpu().numpy().astype(int)  # class indices
            
            for i in range(len(xyxy)):
                if conf[i] >= conf_threshold:
                    x1, y1, x2, y2 = xyxy[i]
                    confidence = conf[i]
                    class_id = cls[i]
                    
                    # Draw bounding box
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                           linewidth=2, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
                    
                    # Add label
                    label = f'Class {class_id}: {confidence:.2f}'
                    if class_names and class_id < len(class_names):
                        label = f'{class_names[class_id]}: {confidence:.2f}'
                    
                    ax.text(x1, y1-10, label, fontsize=10, color='red', 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    else:  # List format
        for detection in detections:
            if len(detection) >= 6:  # [x, y, w/x2, h/y2, conf, class]
                x, y, w_or_x2, h_or_y2, confidence, class_id = detection[:6]
                
                if confidence >= conf_threshold:
                    # Determine if it's xywh or xyxy format
                    if w_or_x2 <= 1.0 and h_or_y2 <= 1.0:  # Normalized xywh format
                        # Convert from normalized xywh to pixel xyxy
                        x_center = x * img_width
                        y_center = y * img_height
                        width = w_or_x2 * img_width
                        height = h_or_y2 * img_height
                        x1 = x_center - width/2
                        y1 = y_center - height/2
                        x2 = x_center + width/2
                        y2 = y_center + height/2
                    elif x < w_or_x2 and y < h_or_y2:  # xyxy format
                        x1, y1, x2, y2 = x, y, w_or_x2, h_or_y2
                    else:  # Assume pixel xywh format
                        x1 = x - w_or_x2/2
                        y1 = y - h_or_y2/2
                        x2 = x + w_or_x2/2
                        y2 = y + h_or_y2/2
                    
                    # Draw bounding box
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                           linewidth=2, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
                    
                    # Add label
                    label = f'Class {int(class_id)}: {confidence:.2f}'
                    if class_names and int(class_id) < len(class_names):
                        label = f'{class_names[int(class_id)]}: {confidence:.2f}'
                    
                    ax.text(x1, y1-10, label, fontsize=10, color='red',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)  # Flip y-axis to match image coordinates
    ax.set_title('YOLO Detection Results')
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_features_over_image(image, features, feature_indices=None, alpha=0.6, cmap='viridis'):
    """
    Plot feature tensor over original image with proper size matching.
    
    Args:
        image: PIL Image, numpy array (H,W,C) or (H,W), or path to image file
        features: Tensor of shape (dim, X, Y) - your feature tensor 
        feature_indices: List of feature indices to visualize (if None, shows first 4)
        alpha: Transparency of feature overlay (0-1)
        cmap: Colormap for features
    """
    
    # Convert features to numpy if it's a tensor
    if torch.is_tensor(features):
        features_np = features.detach().cpu().numpy()
    else:
        features_np = features
    
    # Handle original image
    if isinstance(image, str):
        img = Image.open(image)
        img_np = np.asarray(img)
    elif isinstance(image, Image.Image):
        img_np = np.asarray(image)
    else:
        img_np = image
        
    # Get dimensions
    dim, feat_h, feat_w = features_np.shape
    
    if len(img_np.shape) == 3:
        img_h, img_w, _ = img_np.shape
    else:
        img_h, img_w = img_np.shape
        img_np = np.stack([img_np] * 3, axis=-1)  # Convert grayscale to RGB
    
    # Determine which features to show
    if feature_indices is None:
        feature_indices = list(range(min(4, dim)))
    
    # Create subplots
    n_features = len(feature_indices)
    fig, axes = plt.subplots(1, n_features + 1, figsize=(4 * (n_features + 1), 4))
    
    if n_features == 0:
        axes = [axes]
    
    # Show original image
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot each selected feature
    for i, feat_idx in enumerate(feature_indices):
        # Get the feature map
        feature_map = features_np[feat_idx]
        
        # Resize feature map to match image dimensions
        feature_resized = resize_image(feature_map, (img_h, img_w))
        
        # Normalize feature map for visualization
        feature_norm = (feature_resized - feature_resized.min()) / (feature_resized.max() - feature_resized.min())
        
        # Plot
        axes[i + 1].imshow(img_np)
        im = axes[i + 1].imshow(feature_norm, alpha=alpha, cmap=cmap)
        axes[i + 1].set_title(f'Feature {feat_idx}')
        axes[i + 1].axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i + 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def resize_image(features: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize mask to target size using interpolation.
    
    Args:
        mask: 2D numpy array (H, W)
        target_size: Tuple (H, W) - target dimensions
    """
    # Convert to tensor for interpolation
    features_tensor = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    
    # Resize using bilinear interpolation
    resized = F.interpolate(features_tensor, size=target_size, mode='bilinear', align_corners=False)
    
    # Convert back to boolean mask
    return resized.squeeze().numpy()

def show_annotations(anns):
    """
    Visualizes segmentation annotations by overlaying colored masks on an image.

    Args:
        anns (List[Dict]): List of annotation dictionaries, each containing:
            - 'area': Area of the segmentation mask
            - 'segmentation': Binary mask of shape (H, W) indicating the segmented region

    The function:
    1. Sorts annotations by area in descending order
    2. Creates a transparent RGBA image
    3. For each annotation, generates a random color with 0.35 opacity
    4. Overlays the colored masks on the image
    5. Displays the result using matplotlib

    Returns:
        None. Displays the visualization in the current matplotlib axis.
    """
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)