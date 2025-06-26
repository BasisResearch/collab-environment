import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from typing import List
from tqdm import trange
import numpy as np
import cv2
from PIL import Image
import maskclip_onnx

########################################################
########## CLIP Feature Extraction Utils ###############
########################################################

class MaskCLIPFeaturizer(nn.Module):
    def __init__(self, clip_model_name: str = 'ViT-L/14@336px'):
        super().__init__()
        self.model, self.preprocess = maskclip_onnx.clip.load(clip_model_name)
        self.model.eval()
        self.patch_size = self.model.visual.patch_size

    def forward(self, img):
        b, _, input_size_h, input_size_w = img.shape
        patch_h = input_size_h // self.patch_size
        patch_w = input_size_w // self.patch_size
        features = self.model.get_patch_encodings(img).to(torch.float32)
        return features.reshape(b, patch_h, patch_w, -1).permute(0, 3, 1, 2)

def load_clip_model(model_name: str = 'ViT-L/14@336px'):
    """
    Loads a CLIP model from the maskclip_onnx library.

    Args:
        clip_model_name (str): The name of the CLIP model to load.

    Returns:
        clip_model: The loaded CLIP model.
    """
    print (f"Loading CLIP model with backbone: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MaskCLIPFeaturizer(model_name).to(device).eval()
    return model

def extract_clip_features(clip_model, image: np.ndarray, resolution: int = 224):
    """Extract CLIP features for a single image with preprocessing."""

    # Get model device
    is_cuda = next(clip_model.parameters()).is_cuda 
    device = "cuda" if is_cuda else "cpu"

    # Transform image to tensor and normalize
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = image[:, :, ::-1]  # BGR to RGB

    # Resize image to longest edge
    if max(image.shape[:2]) > resolution:
        if image.shape[0] > image.shape[1]:
            image = cv2.resize(image, (int(resolution * image.shape[1] / image.shape[0]), resolution))
        else:
            image = cv2.resize(image, (resolution, int(resolution * image.shape[0] / image.shape[1])))
    
    input_image = transform(Image.fromarray(image)).to(device)

    # Pass through CLIP
    image_embedding = clip_model(input_image[None])[0]
    feat_dim = image_embedding.shape[0]

    del input_image

    return image_embedding, feat_dim

def aggregate_samclip_features(clip_features, masks, obj_resolution: int, final_resolution: int):
    """
    Aggregate CLIP features based on SAM segmentation masks.
    
    Args:
        clip_features (torch.Tensor): CLIP features for the whole image (C,H,W)
        masks (torch.Tensor): Segmentation masks from SAM (N,H,W)
        obj_resolution (int): Resolution for intermediate feature map
        final_resolution (int): Resolution for final output
        
    Returns:
        torch.Tensor: Aggregated feature map (C,H,W)
    """
    
    # Get input dimensions
    _, h, w = clip_features.shape
    
    # Calculate object level dimensions while preserving aspect ratio
    obj_w = obj_resolution
    obj_h = h * obj_w // w
    obj_dim = (obj_h, obj_w)

    # Calculate final dimensions while preserving aspect ratio 
    final_w = final_resolution
    final_h = h * final_w // w
    final_dim = (final_h, final_w)

    # Interpolate CLIP features to object size
    clip_features = F.interpolate(clip_features.unsqueeze(0), 
                                size=obj_dim,
                                mode='bilinear',
                                align_corners=False)[0]

    # Interpolate masks to object size 
    masks = F.interpolate(masks.unsqueeze(1),
                         size=obj_dim,
                         mode='nearest').bool()[:,0]

    masks = masks.to(clip_features.device)

    # Use einsum for feature aggregation
    # Shape: (n_masks, h, w) * (c, h, w) -> (c, h, w)
    weighted_features = torch.einsum('nhw,chw->chw', masks.float(), clip_features)
    
    # Count number of masks per pixel
    mask_counts = masks.sum(0).float()
    
    # Normalize features by mask counts, avoiding division by zero
    aggregated_feat_map = weighted_features / (mask_counts + 1e-6).unsqueeze(0)
    
    # Resize to final dimensions
    aggregated_feat_map = F.interpolate(aggregated_feat_map.unsqueeze(0),
                                      size=final_dim,
                                      mode='bilinear',
                                      align_corners=False)[0]
    
    return aggregated_feat_map

########################################################
########## DINO Feature Extraction Utils ###############
########################################################

def extract_dino_features(model_name: str, image_paths: List[str], resolution: int):

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading DINOv2 model...")

    # Load DINOv2 model
    dinov2 = torch.hub.load('facebookresearch/dinov2', model_name)
    dinov2 = dinov2.to(device)

    all_features = []

    # Extract DINOv2 features for each image
    for i in trange(len(image_paths)):

        # Open image and resize --> apply image transformation
        image = Image.open(image_paths[i])
        image = resize_image(image, resolution)
        image = transform(image)[:3].unsqueeze(0)

        # Setup for DINO --> interpolating overall image to be evenly divisible by patch size
        image, target_H, target_W = interpolate_to_patch_size(image, dinov2.patch_size)
        image = image.cuda()

        # Forward pass through DINOv2
        with torch.no_grad():
            features = dinov2.forward_features(image)["x_norm_patchtokens"][0]
        
        # Reshape from patches to image and permute features
        features = features.cpu()
        features_hwc = features.reshape((target_H // dinov2.patch_size, target_W // dinov2.patch_size, -1))
        features_chw = features_hwc.permute((2, 0, 1))

        all_features.append(features_chw)

    del dinov2
    pytorch_gc()
    
    return all_features
########################################################
########## Feature Reconstruction Utils ################
########################################################

def reconstruct_features(feature_splatting_model, features):
    """
    Recover full dimensional features from distilled features
    of feature splatting model.
    """

    features = features.permute(1, 0).unsqueeze(0).unsqueeze(-1)  # [1, 13, N, 1]

    # Pass through 1d conv to upsample 
    hidden_features = feature_splatting_model.feature_mlp.hidden_conv(features)

    # Pass through branch in relation to the current feature
    recovered_features = {}
    for branch_name, branch_layer in feature_splatting_model.feature_mlp.feature_branch_dict.items():
        recovered_features[branch_name] = branch_layer(hidden_features)

    return recovered_features