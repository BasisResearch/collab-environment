"""
Enhanced Grounded Segmentation App with Scribble and Inpainting support.

Sourced from https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/main/gradio_app.py
Modified to use huggingface backend for GroundingDINO and SAM and RAM for captioning.
Added scribble and inpainting functionality.
"""

import os
import argparse
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import gc
import random
from typing import List, Optional, Union
from scipy import ndimage

# Import the grounded segmentation utils
from environment.utils.grounded_segmentation_utils import (
    load_grounding_dino_detector,
    load_sam_segmenter,
    load_ram_model,
    grounded_segmentation,
    auto_detect_and_segment,
    annotate,
    cleanup_gpu_memory,
    refine_masks
)

# Additional imports for inpainting
try:
    from diffusers import StableDiffusionInpaintPipeline
    INPAINTING_AVAILABLE = True
except ImportError:
    print("Warning: diffusers not available. Inpainting functionality will be disabled.")
    INPAINTING_AVAILABLE = False

class GroundedSegmentationApp:
    """Main application class for Grounded Segmentation with enhanced features."""
    
    def __init__(self):
        self.detector = None
        self.segmenter = None
        self.processor = None
        self.ram_model = None
        self.inpaint_pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_models(self):
        """Load all required models."""
        if self.detector is None:
            print("Loading Grounding DINO detector...")
            try:
                self.detector = load_grounding_dino_detector("IDEA-Research/grounding-dino-tiny")
            except Exception as e:
                print(f"Failed to load Grounding DINO: {e}")
                # Fallback to base model
                self.detector = load_grounding_dino_detector("IDEA-Research/grounding-dino-base")
        
        if self.segmenter is None or self.processor is None:
            print("Loading SAM segmenter...")
            try:
                self.segmenter, self.processor = load_sam_segmenter("syscv-community/sam-hq-vit-base")
            except Exception as e:
                print(f"Failed to load SAM: {e}")
                # Try alternative SAM model
                try:
                    self.segmenter, self.processor = load_sam_segmenter("syscv-community/sam-hq-vit-base")
                except Exception as e2:
                    print(f"Failed to load alternative SAM: {e2}")
                    raise e2
    
    def load_ram_model_if_needed(self, model_name: str = "ram"):
        """Load RAM model for automatic label detection."""
        if self.ram_model is None:
            print(f"Loading RAM model: {model_name}...")
            try:
                self.ram_model = load_ram_model(model_name, device=self.device)
            except Exception as e:
                print(f"Failed to load RAM model: {e}")
                # Try alternative RAM model
                try:
                    alt_model = "ram_plus" if model_name == "ram" else "ram"
                    self.ram_model = load_ram_model(alt_model, device=self.device)
                except Exception as e2:
                    print(f"Failed to load alternative RAM model: {e2}")
                    raise e2

    def load_inpainting_pipeline(self):
        """Load inpainting pipeline if needed."""
        if not INPAINTING_AVAILABLE:
            raise ValueError("Inpainting not available. Please install diffusers: pip install diffusers")
        
        if self.inpaint_pipeline is None:
            print("Loading inpainting pipeline...")
            self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting", 
                torch_dtype=torch.float16
            )
            self.inpaint_pipeline = self.inpaint_pipeline.to(self.device)

    def draw_mask(self, mask, draw, random_color=False):
        """Draw mask on the image"""
        print(f"Debug - draw_mask input shape: {mask.shape}, dtype: {mask.dtype}")
        
        if random_color:
            color = tuple(np.random.randint(0, 255, size=3).tolist() + [128])
        else:
            color = (255, 0, 0, 128)
        
        # Ensure mask is 2D
        if len(mask.shape) > 2:
            # If 3D, take the first channel or squeeze extra dimensions
            if mask.shape[2] == 1:
                mask = mask.squeeze(2)
            else:
                mask = mask[:, :, 0]  # Take first channel
        
        # Ensure mask is boolean
        if mask.dtype != bool:
            mask = mask > 0.5
        
        print(f"Debug - processed mask shape: {mask.shape}, unique values: {np.unique(mask)}")
        
        # Find coordinates where mask is True
        coords = np.where(mask)
        
        if len(coords[0]) > 0:
            print(f"Debug - found {len(coords[0])} mask pixels")
            # Draw each point (coords[0] = y, coords[1] = x)
            for y, x in zip(coords[0], coords[1]):
                draw.point((int(x), int(y)), fill=color)
        else:
            print("Debug - no mask pixels found")

    def draw_box(self, box, draw, label):
        """Draw bounding box with label on image."""
        # Random color for box
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        
        draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline=color, width=2)
        
        if label:
            font = ImageFont.load_default()
            if hasattr(font, "getbbox"):
                bbox = draw.textbbox((box[0], box[1]), str(label), font)
            else:
                w, h = draw.textsize(str(label), font)
                bbox = (box[0], box[1], w + box[0], box[1] + h)
            draw.rectangle(bbox, fill=color)
            draw.text((box[0], box[1]), str(label), fill="white")

    def process_scribble_segmentation(self, input_image, scribble_mode="split", polygon_refinement=False):
        """Process scribble-based segmentation."""
        try:
            self.load_models()
            
            # Handle different input formats
            if isinstance(input_image, dict):
                image = input_image.get("image")
                scribble = input_image.get("mask")
                
                # Debug information
                print(f"Debug - input_image keys: {list(input_image.keys())}")
                print(f"Debug - image type: {type(image)}")
                print(f"Debug - scribble type: {type(scribble)}")
                
            else:
                # Fallback: treat input as image only
                image = input_image
                scribble = None
                print(f"Debug - input_image is not dict, type: {type(input_image)}")
            
            if image is None:
                return input_image, "No image provided."
            
            if scribble is None:
                return image, "No scribble/mask provided. Please draw on the image to indicate areas to segment."
            
            # Convert to numpy arrays
            image_np = np.array(image.convert("RGB"))
            
            # Handle different scribble formats
            if isinstance(scribble, Image.Image):
                scribble_np = np.array(scribble.convert("RGB"))
            elif isinstance(scribble, np.ndarray):
                scribble_np = scribble
            else:
                return image, f"Unsupported scribble format: {type(scribble)}"
            
            print(f"Debug - scribble_np shape: {scribble_np.shape}")
            print(f"Debug - scribble_np min/max: {scribble_np.min()}/{scribble_np.max()}")
            
            # Extract scribble points - try multiple methods
            if len(scribble_np.shape) == 3:
                # RGB image - try different approaches
                
                # Method 1: Look for non-zero pixels (any color)
                scribble_mask = np.any(scribble_np > 0, axis=2)
                
                # Method 2: Look for non-white pixels (drawings are usually not white)
                non_white_mask = ~np.all(scribble_np == 255, axis=2)
                
                # Method 3: Look for pixels that are significantly different from white
                white_diff = np.sum(np.abs(scribble_np.astype(int) - 255), axis=2)
                significant_diff_mask = white_diff > 30
                
                # Method 4: Look for non-black pixels
                non_black_mask = np.any(scribble_np > 10, axis=2)
                
                # Try each method
                for method_name, mask in [
                    ("non-zero", scribble_mask),
                    ("non-white", non_white_mask), 
                    ("significant-diff", significant_diff_mask),
                    ("non-black", non_black_mask)
                ]:
                    num_pixels = np.sum(mask)
                    print(f"Debug - {method_name} method found {num_pixels} pixels")
                    
                    if num_pixels > 10:  # Need at least some pixels
                        scribble_gray = mask.astype(np.uint8) * 255
                        break
                else:
                    # If no method works, try converting to grayscale
                    scribble_gray = np.mean(scribble_np, axis=2)
                    print(f"Debug - fallback grayscale, unique values: {np.unique(scribble_gray)}")
            else:
                # Already grayscale
                scribble_gray = scribble_np
                print(f"Debug - already grayscale, unique values: {np.unique(scribble_gray)}")
            
            # Find connected components with multiple thresholds
            thresholds_to_try = [1, 10, 50, 100, 128, 200]
            
            for threshold in thresholds_to_try:
                binary_mask = scribble_gray > threshold
                labeled_array, num_features = ndimage.label(binary_mask)
                
                print(f"Debug - threshold {threshold}: {num_features} features, {np.sum(binary_mask)} pixels")
                
                if num_features > 0:
                    break
            
            if num_features == 0:
                # Last resort: find any non-zero pixels and use them as points
                nonzero_points = np.where(scribble_gray > 0)
                if len(nonzero_points[0]) > 0:
                    # Sample some points if there are too many
                    if len(nonzero_points[0]) > 10:
                        indices = np.random.choice(len(nonzero_points[0]), 10, replace=False)
                        centers = np.column_stack([nonzero_points[0][indices], nonzero_points[1][indices]])
                    else:
                        centers = np.column_stack([nonzero_points[0], nonzero_points[1]])
                    
                    print(f"Debug - using {len(centers)} sampled points as fallback")
                else:
                    return image, f"No valid scribbles found after trying multiple detection methods. Scribble range: {scribble_gray.min()}-{scribble_gray.max()}"
            else:
                # Calculate centroids of scribbles
                centers = ndimage.center_of_mass(binary_mask, labeled_array, range(1, num_features + 1))
                centers = np.array(centers)
                
                # Handle single point case
                if centers.ndim == 1:
                    centers = centers.reshape(1, -1)
                
                print(f"Debug - found {len(centers)} center points: {centers}")
            # Convert centers to input points format for HuggingFace processor
            # Centers are in [y, x] format, convert to [x, y] for the processor
            input_points = [[[int(center[1]), int(center[0])] for center in centers]]

            # Prepare inputs using the processor
            inputs = self.processor(images=image, input_points=input_points, return_tensors="pt")

            # Move inputs to device
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(self.device)

            # Generate masks
            with torch.no_grad():
                outputs = self.segmenter(**inputs)

            masks = self.processor.post_process_masks(
                masks=outputs.pred_masks.cpu(),
                original_sizes=inputs["original_sizes"].cpu(), 
                reshaped_input_sizes=inputs["reshaped_input_sizes"].cpu()
            )[0]

            print (f"Debug - masks shape: {masks.shape}")
            print (f"Debug - masks type: {type(masks)}")
            print (f"Debug - masks {masks}")

            masks = refine_masks(masks, polygon_refinement)

            # Debug masks structure
            print(f"Debug - masks type: {type(masks)}")
            print(f"Debug - masks length: {len(masks)}")
            if len(masks) > 0:
                print(f"Debug - masks[0] shape: {masks[0].shape}")

            # Create result image
            mask_image = Image.new('RGBA', image.size, color=(0, 0, 0, 0))
            mask_draw = ImageDraw.Draw(mask_image)

            # Process masks correctly
            num_generated_masks = 0
            if len(masks) > 0:
                mask_batch = masks[0]  # Shape is typically [batch_size, num_masks, height, width]
                
                if len(mask_batch.shape) == 4:  # [batch_size, num_masks, height, width]
                    batch_size, num_masks, height, width = mask_batch.shape
                    for i in range(num_masks):
                        mask = mask_batch[0, i]
                        if isinstance(mask, torch.Tensor):
                            mask = mask.cpu().numpy()  # Get single mask [height, width]
                        print(f"Debug - Processing mask {i}, shape: {mask.shape}")
                        self.draw_mask(mask, mask_draw, random_color=True)
                    num_generated_masks = num_masks
                    
                elif len(mask_batch.shape) == 3:  # [num_masks, height, width]
                    num_masks, height, width = mask_batch.shape
                    for i in range(num_masks):
                        mask = mask_batch[i]
                        if isinstance(mask, torch.Tensor):
                            mask = mask.cpu().numpy()
                        print(f"Debug - Processing mask {i}, shape: {mask.shape}")
                        self.draw_mask(mask, mask_draw, random_color=True)
                    num_generated_masks = num_masks
                    
                else:
                    print(f"Warning - Unexpected mask shape: {mask_batch.shape}")
                    # Try to process as single mask
                    mask = mask_batch
                    if isinstance(mask, torch.Tensor):
                        mask = mask.squeeze().cpu().numpy()
                    
                    if len(mask.shape) == 2:
                        self.draw_mask(mask, mask_draw, random_color=True)
                        num_generated_masks = 1

            result_image = image.convert('RGBA')
            result_image.alpha_composite(mask_image)

            info_text = f"Processed {len(centers)} scribble points and generated {num_generated_masks} masks."
            info_text += f"\nDetection method: threshold {threshold}, {num_features} connected components"

            return result_image, info_text
        except Exception as e:
            error_msg = f"Error during scribble processing: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            
            # Return the original image or a fallback
            if isinstance(input_image, dict):
                fallback_image = input_image.get("image", input_image)
            else:
                fallback_image = input_image
            return fallback_image, error_msg
        finally:
            cleanup_gpu_memory()

    def process_inpainting(self, input_image, text_prompt, inpaint_prompt, detection_threshold, inpaint_mode="merge"):
        """Process inpainting with detected objects."""
        try:
            if not INPAINTING_AVAILABLE:
                return input_image, "Inpainting not available. Please install diffusers."
            
            if not inpaint_prompt.strip():
                return input_image, "Please provide an inpainting prompt."
            
            if not text_prompt.strip():
                return input_image, "Please provide a text prompt to detect objects for inpainting."
            
            self.load_models()
            self.load_inpainting_pipeline()
            
            # Parse labels from text prompt
            labels = [label.strip() for label in text_prompt.split(',')]
            
            # Perform grounded segmentation to get masks
            image, detections = grounded_segmentation(
                image=input_image,
                labels=labels,
                detector=self.detector,
                segmenter=self.segmenter,
                processor=self.processor,
                threshold=detection_threshold,
                polygon_refinement=False
            )
            
            if not detections:
                return input_image, "No objects detected for inpainting. Try lowering the detection threshold."
            
            # Convert detections to masks
            masks = []
            for detection in detections:
                if hasattr(detection, 'mask') and detection.mask is not None:
                    masks.append(detection.mask)
            
            if not masks:
                return input_image, "No masks generated from detections."
            
            # Combine masks based on inpaint_mode
            if inpaint_mode == 'merge':
                # Merge all masks
                combined_mask = np.zeros_like(masks[0], dtype=bool)
                for mask in masks:
                    combined_mask = combined_mask | mask
                final_mask = combined_mask
            else:
                # Use first mask only
                final_mask = masks[0]
            
            # Convert mask to PIL Image
            mask_pil = Image.fromarray((final_mask * 255).astype(np.uint8))
            
            # Resize for inpainting pipeline (512x512)
            original_size = input_image.size
            image_resized = input_image.resize((512, 512))
            mask_resized = mask_pil.resize((512, 512))
            
            # Perform inpainting
            inpainted_image = self.inpaint_pipeline(
                prompt=inpaint_prompt,
                image=image_resized,
                mask_image=mask_resized
            ).images[0]
            
            # Resize back to original size
            inpainted_image = inpainted_image.resize(original_size)
            
            info_text = f"Inpainting completed using {len(detections)} detected objects.\n"
            info_text += f"Inpaint prompt: '{inpaint_prompt}'\n"
            info_text += f"Objects detected: {', '.join([det.label for det in detections])}"
            
            return inpainted_image, info_text
            
        except Exception as e:
            error_msg = f"Error during inpainting: {str(e)}"
            print(error_msg)
            return input_image, error_msg
        finally:
            cleanup_gpu_memory()

    def process_image(
        self,
        input_image,
        task_type: str,
        text_prompt: str,
        inpaint_prompt: str,
        detection_threshold: float,
        polygon_refinement: bool,
        ram_model_name: str,
        scribble_mode: str = "split",
        inpaint_mode: str = "merge"
    ) -> tuple:
        """
        Main processing function for different task types.
        
        Args:
            input_image: Input PIL Image or dict with image and mask
            task_type: Type of task to perform
            text_prompt: Text prompt for detection (if manual)
            inpaint_prompt: Prompt for inpainting
            detection_threshold: Confidence threshold for detections
            polygon_refinement: Whether to apply polygon refinement
            ram_model_name: Name of RAM model to use for auto detection
            scribble_mode: Mode for scribble processing
            inpaint_mode: Mode for inpainting mask combination
            
        Returns:
            Tuple of (result_image, info_text)
        """
        
        # Handle scribble task type
        if task_type == "scribble":
            return self.process_scribble_segmentation(input_image, scribble_mode, polygon_refinement)
        
        # Handle inpainting task type
        if task_type == "inpainting":
            # For inpainting, input_image should be a PIL Image
            if isinstance(input_image, dict):
                input_image = input_image["image"]
            return self.process_inpainting(
                input_image, text_prompt, inpaint_prompt, 
                detection_threshold, inpaint_mode
            )
        
        # For other task types, extract image if needed
        if isinstance(input_image, dict):
            input_image = input_image["image"]
        
        # Handle existing task types (automatic, manual)
        try:
            # Load base models
            self.load_models()
            
            if task_type == "automatic":
                # Load RAM model for automatic detection
                self.load_ram_model_if_needed(ram_model_name)
                
                # Perform automatic detection and segmentation
                image, detections = auto_detect_and_segment(
                    image=input_image,
                    ram_model=self.ram_model,
                    detector=self.detector,
                    segmenter=self.segmenter,
                    processor=self.processor,
                    threshold=detection_threshold,
                    polygon_refinement=polygon_refinement,
                    device=self.device
                )
                
                labels_found = list(set([det.label for det in detections]))
                info_text = f"Automatically detected {len(labels_found)} unique labels: {', '.join(labels_found)}\n"
                info_text += f"Found {len(detections)} total detections."
                
            elif task_type == "manual":
                if not text_prompt.strip():
                    return input_image, "Please provide text prompt for manual detection."
                
                # Parse labels from text prompt
                labels = [label.strip() for label in text_prompt.split(',')]
                
                # Perform manual grounded segmentation
                image, detections = grounded_segmentation(
                    image=input_image,
                    labels=labels,
                    detector=self.detector,
                    segmenter=self.segmenter,
                    processor=self.processor,
                    threshold=detection_threshold,
                    polygon_refinement=polygon_refinement
                )
                
                info_text = f"Searched for {len(labels)} labels: {', '.join(labels)}\n"
                info_text += f"Found {len(detections)} detections."
                
            else:
                return input_image, "Invalid task type selected."
            
            # Annotate the image with detections
            if detections:
                annotated_image = annotate(image, detections)
                
                # Add detailed detection info
                info_text += "\n\nDetailed Results:\n"
                for i, det in enumerate(detections):
                    info_text += f"{i+1}. {det.label}: {det.score:.3f} confidence\n"
                    info_text += f"   Box: ({det.box.xmin}, {det.box.ymin}) to ({det.box.xmax}, {det.box.ymax})\n"
                
                return Image.fromarray(annotated_image), info_text
            else:
                return input_image, info_text + "\n\nNo detections found. Try lowering the detection threshold."
                
        except Exception as e:
            error_msg = f"Error during processing: {str(e)}"
            print(error_msg)
            return input_image, error_msg
        finally:
            # Clean up GPU memory after processing
            cleanup_gpu_memory()

def create_gradio_interface():
    """Create and configure the Gradio interface."""
    
    app = GroundedSegmentationApp()
    
    with gr.Blocks(title="Enhanced Grounded Segmentation", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎯 Enhanced Grounded Segmentation
        
        Detect and segment objects in images using Grounding DINO + SAM + RAM.
        
        **Five modes available:**
        - **Automatic**: Uses RAM model to automatically detect objects
        - **Manual**: Specify what objects to detect using text prompts
        - **Scribble**: Interactive segmentation by drawing on the image
        - **Inpainting**: Replace detected objects with AI-generated content
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### 📤 Input")
                
                # Dynamic input based on task type
                input_image = gr.Image(
                    label="Upload Image",
                    type="pil",
                    height=300
                )
                
                # Scribble input - fallback for older Gradio versions
                try:
                    # Try modern ImageEditor first
                    scribble_input = gr.ImageEditor(
                        label="Draw on Image (for Scribble mode)",
                        type="pil",
                        height=300,
                        visible=False,
                        brush=gr.Brush(colors=["#ff0000"], color_mode="fixed")
                    )
                except (AttributeError, TypeError):
                    # Fallback to basic Image component
                    scribble_input = gr.Image(
                        label="Upload Image with Annotations (for Scribble mode)",
                        type="pil",
                        height=300,
                        visible=False
                    )
                
                task_type = gr.Radio(
                    choices=["automatic", "manual", "scribble", "inpainting"],
                    value="automatic",
                    label="Task Mode",
                    info="Choose the type of segmentation task"
                )
                
                text_prompt = gr.Textbox(
                    label="Text Prompt (for Manual/Inpainting modes)",
                    placeholder="Enter objects to detect, separated by commas (e.g., 'person, car, dog')",
                    lines=2,
                    visible=False
                )
                
                inpaint_prompt = gr.Textbox(
                    label="Inpainting Prompt (for Inpainting mode)",
                    placeholder="Describe what should replace the detected objects (e.g., 'a beautiful garden')",
                    lines=2,
                    visible=False
                )
                
                # Advanced settings
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    detection_threshold = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.3,
                        step=0.05,
                        label="Detection Threshold",
                        info="Higher values = more confident detections only"
                    )
                    
                    polygon_refinement = gr.Checkbox(
                        label="Polygon Refinement",
                        value=False,
                        info="Apply polygon refinement to masks (slower but more accurate)"
                    )
                    
                    ram_model_name = gr.Radio(
                        choices=["ram", "ram_plus"],
                        value="ram",
                        label="RAM Model (for Automatic mode)",
                        info="RAM Plus is more accurate but slower"
                    )
                    
                    scribble_mode = gr.Radio(
                        choices=["merge", "split"],
                        value="split",
                        label="Scribble Mode",
                        info="How to handle multiple scribbles"
                    )
                    
                    inpaint_mode = gr.Radio(
                        choices=["merge", "first"],
                        value="merge",
                        label="Inpaint Mode",
                        info="Use merged masks or first mask for inpainting"
                    )
                
                run_button = gr.Button("🚀 Run Segmentation", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                # Output section
                gr.Markdown("### 📤 Results")
                output_image = gr.Image(
                    label="Result",
                    type="pil",
                    height=400
                )
                
                info_output = gr.Textbox(
                    label="Processing Info",
                    lines=10,
                    max_lines=15,
                    show_copy_button=True
                )
        
        # Examples section
        gr.Markdown("### 💡 Examples")
        gr.Examples(
            examples=[
                # [image, task_type, text_prompt, inpaint_prompt, threshold, refinement, ram_model, scribble_mode, inpaint_mode]
                ["assets/demo1.jpg", "automatic", "", "", 0.3, False, "ram", "split", "merge"],
                ["assets/demo2.jpg", "manual", "person, car, building", "", 0.3, False, "ram", "split", "merge"],
                ["assets/demo3.jpg", "inpainting", "car", "a beautiful flower garden", 0.25, False, "ram", "split", "merge"],
            ],
            inputs=[input_image, task_type, text_prompt, inpaint_prompt, detection_threshold, polygon_refinement, ram_model_name, scribble_mode, inpaint_mode],
            outputs=[output_image, info_output],
            fn=app.process_image,
            cache_examples=False,
            label="Try these examples"
        )
        
        # Event handlers
        def update_interface_visibility(task):
            """Update interface elements based on selected task type."""
            show_text = task in ["manual", "inpainting"]
            show_inpaint = task == "inpainting"
            show_scribble = task == "scribble"
            show_regular = task != "scribble"
            
            return (
                gr.update(visible=show_text),      # text_prompt
                gr.update(visible=show_inpaint),   # inpaint_prompt
                gr.update(visible=show_scribble),  # scribble_input
                gr.update(visible=show_regular)    # input_image
            )
        
        task_type.change(
            update_interface_visibility,
            inputs=[task_type],
            outputs=[text_prompt, inpaint_prompt, scribble_input, input_image]
        )
        
        def run_processing(regular_image, scribble_image, task, text_prompt, inpaint_prompt, 
                          detection_threshold, polygon_refinement, ram_model_name, 
                          scribble_mode, inpaint_mode):
            """Route to appropriate processing based on task type."""
            
            # Choose the right input based on task type
            if task == "scribble":
                if scribble_image is None:
                    return regular_image, "Please draw on the image for scribble mode."
                
                # Handle different scribble image formats
                if isinstance(scribble_image, dict):
                    # Modern ImageEditor format
                    if "image" in scribble_image and "mask" in scribble_image:
                        input_img = {"image": scribble_image["image"], "mask": scribble_image["mask"]}
                    elif "background" in scribble_image:
                        # Alternative format with background and layers
                        background = scribble_image.get("background")
                        layers = scribble_image.get("layers", [])
                        if layers:
                            # Create a simple mask from the first layer
                            mask = layers[0] if layers else None
                            input_img = {"image": background, "mask": mask}
                        else:
                            input_img = {"image": background, "mask": None}
                    else:
                        # Unknown dict format, try to use as is
                        input_img = scribble_image
                else:
                    # Simple PIL Image format - create a dummy structure
                    input_img = {"image": scribble_image, "mask": None}
            else:
                if regular_image is None:
                    return None, "Please upload an image."
                input_img = regular_image
            
            return app.process_image(
                input_img, task, text_prompt, inpaint_prompt,
                detection_threshold, polygon_refinement, ram_model_name,
                scribble_mode, inpaint_mode
            )
        
        run_button.click(
            fn=run_processing,
            inputs=[
                input_image, scribble_input, task_type, text_prompt, inpaint_prompt,
                detection_threshold, polygon_refinement, ram_model_name, 
                scribble_mode, inpaint_mode
            ],
            outputs=[output_image, info_output],
            show_progress=True
        )
        
        # Footer
        gr.Markdown("""
        ---
        **Models Used:**
        - 🎯 **Grounding DINO**: Zero-shot object detection
        - 🎭 **SAM**: Segment Anything Model for precise segmentation  
        - 🧠 **RAM**: Recognize Anything Model for automatic label generation
        - 🎨 **Stable Diffusion**: Inpainting for object replacement
        
        **Task Modes:**
        - **Automatic**: AI automatically finds and segments objects
        - **Manual**: You specify what objects to find and segment
        - **Scribble**: Draw on the image to segment specific areas
        - **Inpainting**: Replace detected objects with AI-generated content
        
        **Tips:**
        - Lower detection threshold to find more objects (but may include false positives)
        - Enable polygon refinement for better mask accuracy (slower processing)
        - For inpainting, be descriptive in your prompts for better results
        - For scribble mode, draw roughly on the objects you want to segment
        """)
    
    return demo

def main():
    """Main function to run the application."""
    parser = argparse.ArgumentParser("Enhanced Grounded Segmentation App")
    parser.add_argument("--share", action="store_true", help="Share the app publicly")
    parser.add_argument("--port", type=int, default=80, help="Port to run the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Create the Gradio interface
    demo = create_gradio_interface()
    
    # Launch the app
    demo.queue(max_size=10).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True
    )

if __name__ == "__main__":
    main()