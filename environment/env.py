import os
import argparse
import subprocess
import tempfile
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

class Environment:
    def __init__(self, file_path: str):
        """Initialize the environment with a directory and SAM model.
        
        Args:
            file_path: Path to the file to create an environment for (image, video, etc.)
            **kwargs: Additional keyword arguments to store in the environment
        """
        self.file_path = Path(file_path)
        # self.kwargs = kwargs
    
    # def create_gaussian_splat(self, image_paths: List[str], output_dir: str) -> Path:
    #     """Create gaussian splats from a set of images using nerfstudio.
        
    #     Args:
    #         image_paths: List of paths to input images
    #         output_dir: Directory to save the gaussian splat output
            
    #     Returns:
    #         Path to the generated gaussian splat model
    #     """
    #     # Validate inputs
    #     for img_path in image_paths:
    #         if not Path(img_path).exists():
    #             raise ValueError(f"Image path {img_path} does not exist")
        
    #     output_path = Path(output_dir)
    #     output_path.mkdir(parents=True, exist_ok=True)
        
    #     # Prepare nerfstudio command
    #     cmd = [
    #         "ns-train", "gaussian-splatting",
    #         "--data", str(output_path),
    #         "--pipeline.model.density-field.init-num-points", "8192",
    #         "--pipeline.datamanager.train-num-rays-per-batch", "4096",
    #         "--pipeline.model.density-field.init-num-points-per-ray", "8",
    #     ]
        
    #     # Add image paths
    #     for img_path in image_paths:
    #         cmd.extend(["--data.images", str(img_path)])
        
    #     # Run nerfstudio training
    #     subprocess.run(cmd, check=True)
        
    #     return output_path / "gaussian_splat_model"
    
    # def segment_image(self, 
    #                  image_path: str,
    #                  points: Optional[List[Tuple[float, float]]] = None,
    #                  boxes: Optional[List[List[float]]] = None) -> np.ndarray:
    #     """Segment an image using the Segment Anything Model.
        
    #     Args:
    #         image_path: Path to the input image
    #         points: Optional list of point prompts (x, y coordinates)
    #         boxes: Optional list of box prompts [x1, y1, x2, y2]
            
    #     Returns:
    #         Binary mask of the segmentation
    #     """
    #     if self.sam_predictor is None:
    #         raise RuntimeError("SAM model not initialized. Please provide checkpoint in constructor.")
        
    #     # Read and preprocess image
    #     image = cv2.imread(image_path)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    #     # Set image in predictor
    #     self.sam_predictor.set_image(image)
        
    #     # Prepare input prompts
    #     input_point = np.array(points) if points else None
    #     input_box = np.array(boxes) if boxes else None
    #     input_label = np.ones(len(points)) if points else None
        
    #     # Generate mask
    #     masks, scores, logits = self.sam_predictor.predict(
    #         point_coords=input_point,
    #         point_labels=input_label,
    #         box=input_box,
    #         multimask_output=True
    #     )
        
    #     # Return the mask with highest score
    #     best_mask_idx = np.argmax(scores)
    #     return masks[best_mask_idx]