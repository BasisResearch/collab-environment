"""
Working video bbox viewer for Panel dashboard.
Uses the proven HTML/JS approach from the Flask app but adapted for Panel.
"""

import panel as pn
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
import base64

# Enable Panel extensions
pn.extension()


class WorkingVideoBboxViewer:
    """
    Video bbox viewer that actually works in Panel.
    Based on the proven Flask implementation.
    """
    
    def __init__(self, video_path: str, bbox_df: pd.DataFrame, fps: float = 30.0):
        """Initialize the viewer."""
        self.video_path = Path(video_path)
        self.bbox_df = bbox_df.copy()
        self.fps = fps
        
        # Validate video file
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Detect data format
        self.data_format = self._detect_data_format()
        logger.info(f"Detected data format: {self.data_format}")
        
        # Prepare bbox data
        self.bbox_data = self._prepare_bbox_data()
        
        # Create the component
        self.component = self._create_component()
    
    def _detect_data_format(self) -> str:
        """Detect whether data contains bounding boxes or centroids."""
        if 'x1' in self.bbox_df.columns:
            return "bbox"
        elif 'x' in self.bbox_df.columns:
            return "centroid"
        else:
            raise ValueError("DataFrame must contain either bbox (x1,y1,x2,y2) or centroid (x,y) columns")
    
    def _prepare_bbox_data(self) -> Dict[int, list]:
        """Convert DataFrame to frame-indexed dictionary."""
        data_by_frame = {}
        
        for _, row in self.bbox_df.iterrows():
            frame = int(row['frame'])
            track_id = row['track_id']
            
            if frame not in data_by_frame:
                data_by_frame[frame] = []
            
            if self.data_format == "bbox":
                annotation = {
                    'track_id': track_id,
                    'x1': float(row['x1']),
                    'y1': float(row['y1']),
                    'x2': float(row['x2']),
                    'y2': float(row['y2']),
                    'type': 'bbox'
                }
            else:  # centroid
                annotation = {
                    'track_id': track_id,
                    'x': float(row['x']),
                    'y': float(row['y']),
                    'type': 'centroid'
                }
            
            data_by_frame[frame].append(annotation)
        
        return data_by_frame
    
    def _create_video_element(self) -> str:
        """Create video element HTML."""
        # Use Panel's static file serving to avoid base64 encoding
        # This assumes the video is in a location that can be served statically
        
        # If the video is in the cache directory, we can serve it via a static route
        cache_dir = Path.home() / '.cache' / 'collab_env_dashboard'
        
        if str(self.video_path).startswith(str(cache_dir)):
            # Video is in cache, use just the filename for static serving
            filename = self.video_path.name
            video_url = f"/cache/{filename}"  # No /static/ prefix needed
            logger.info(f"Using static cache path: {video_url} for file: {self.video_path}")
        else:
            # Fallback to base64 for videos outside cache
            logger.warning(f"Video not in cache dir, using base64 encoding")
            video_content = self.video_path.read_bytes()
            video_b64 = base64.b64encode(video_content).decode('utf-8')
            video_url = f"data:video/mp4;base64,{video_b64}"
        
        return f'<source src="{video_url}" type="video/mp4">'
    
    def _create_component(self) -> pn.pane.HTML:
        """Create the Panel component with working overlay."""
        bbox_data_json = json.dumps(self.bbox_data, separators=(',', ':'))
        
        # Get video source
        video_source = self._create_video_element()
        
        # Get stats
        track_ids = set()
        for frame_data in self.bbox_data.values():
            for annotation in frame_data:
                track_ids.add(annotation['track_id'])
        
        num_tracks = len(track_ids)
        num_frames = len(self.bbox_data)
        
        # Create the HTML with embedded styles and JavaScript
        html_content = f"""
        <div id="video-bbox-viewer" style="font-family: Arial, sans-serif;">
            <style>
                .vbv-container {{
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .vbv-info {{
                    background: #e3f2fd;
                    padding: 10px;
                    border-radius: 5px;
                    margin-bottom: 15px;
                    border-left: 4px solid #2196f3;
                }}
                .vbv-controls {{
                    margin: 15px 0;
                    padding: 10px;
                    background: #f8f9fa;
                    border-radius: 5px;
                }}
                .vbv-video-container {{
                    position: relative;
                    display: inline-block;
                    margin: 10px 0;
                    border: 3px solid #333;
                    border-radius: 5px;
                }}
                .vbv-video {{
                    display: block;
                    width: 800px;
                    height: 450px;
                }}
                .vbv-canvas {{
                    position: absolute;
                    top: 0;
                    left: 0;
                    pointer-events: none;
                    z-index: 10;
                }}
                .vbv-status {{
                    margin: 10px 0;
                    padding: 8px;
                    background: #d4edda;
                    color: #155724;
                    border-radius: 5px;
                    font-family: monospace;
                    font-size: 12px;
                }}
            </style>
            
            <div class="vbv-container">
                <div class="vbv-info">
                    <strong>Video:</strong> {self.video_path.name} | 
                    <strong>Format:</strong> {self.data_format} | 
                    <strong>Tracks:</strong> {num_tracks} | 
                    <strong>Frames:</strong> {num_frames} | 
                    <strong>FPS:</strong> {self.fps}
                </div>
                
                <div class="vbv-controls">
                    <label style="margin-right: 20px;">
                        <input type="checkbox" id="vbv-show-ids" checked> Show Track IDs
                    </label>
                    <label style="margin-right: 20px;">
                        <input type="checkbox" id="vbv-show-trails"> Show Trails
                    </label>
                    <label>
                        Opacity: 
                        <input type="range" id="vbv-opacity" min="0.1" max="1" step="0.1" value="0.8" style="width: 150px;">
                        <span id="vbv-opacity-value">0.8</span>
                    </label>
                </div>
                
                <div class="vbv-video-container">
                    <video id="vbv-video" class="vbv-video" controls>
                        {video_source}
                        Your browser does not support the video tag.
                    </video>
                    <canvas id="vbv-canvas" class="vbv-canvas" width="800" height="450"></canvas>
                </div>
                
                <div id="vbv-status" class="vbv-status">Initializing...</div>
            </div>
            
            <script>
            // Wrap in IIFE to avoid global scope pollution
            (function() {{
                console.log('🎥 Panel Video Bbox Viewer initializing...');
                
                let retryCount = 0;
                const maxRetries = 50; // Try for 5 seconds
                
                // Wait for elements to be ready
                function initViewer() {{
                    retryCount++;
                    console.log(`Attempt ${{retryCount}}: Looking for elements...`);
                    
                    const video = document.getElementById('vbv-video');
                    const canvas = document.getElementById('vbv-canvas');
                    const status = document.getElementById('vbv-status');
                    
                    console.log('Found elements:', {{
                        video: video ? 'YES' : 'NO',
                        canvas: canvas ? 'YES' : 'NO', 
                        status: status ? 'YES' : 'NO'
                    }});
                    
                    if (!video || !canvas) {{
                        if (retryCount < maxRetries) {{
                            console.log(`Elements not ready (attempt ${{retryCount}}/${{maxRetries}}), retrying...`);
                            setTimeout(initViewer, 100);
                        }} else {{
                            console.error('❌ FAILED to find elements after', maxRetries, 'attempts');
                            console.log('Available elements with vbv prefix:', 
                                Array.from(document.querySelectorAll('[id^="vbv-"]')).map(el => el.id));
                        }}
                        return;
                    }}
                    
                    console.log('✅ Elements found, initializing viewer');
                    
                    const ctx = canvas.getContext('2d');
                    const showIdsCheck = document.getElementById('vbv-show-ids');
                    const showTrailsCheck = document.getElementById('vbv-show-trails');
                    const opacitySlider = document.getElementById('vbv-opacity');
                    const opacityValue = document.getElementById('vbv-opacity-value');
                    
                    // Data
                    const bboxData = {bbox_data_json};
                    const fps = {self.fps};
                    let trails = {{}};
                    
                    // Settings
                    let settings = {{
                        showIds: true,
                        showTrails: false,
                        opacity: 0.8
                    }};
                    
                    console.log('📊 Loaded data for', Object.keys(bboxData).length, 'frames');
                    
                    function updateStatus(message) {{
                        if (status) status.textContent = message;
                    }}
                    
                    function drawOverlay() {{
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        
                        // Overlay indicator
                        ctx.fillStyle = 'rgba(0, 255, 0, 0.8)';
                        ctx.fillRect(10, 10, 150, 25);
                        ctx.fillStyle = 'black';
                        ctx.font = 'bold 14px Arial';
                        ctx.fillText('✓ OVERLAY ACTIVE', 15, 28);
                        
                        if (!video.videoWidth || !video.videoHeight) return;
                        
                        const currentTime = video.currentTime;
                        const currentFrame = Math.floor(currentTime * fps);
                        const frameData = bboxData[currentFrame] || [];
                        
                        if (frameData.length === 0) return;
                        
                        const scaleX = canvas.width / video.videoWidth;
                        const scaleY = canvas.height / video.videoHeight;
                        
                        ctx.globalAlpha = settings.opacity;
                        
                        const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#F7DC6F', '#BB8FCE'];
                        
                        frameData.forEach(annotation => {{
                            const trackId = annotation.track_id;
                            const color = colors[trackId % colors.length];
                            
                            ctx.strokeStyle = color;
                            ctx.fillStyle = color;
                            ctx.lineWidth = 3;
                            
                            if (annotation.type === 'bbox') {{
                                const x1 = annotation.x1 * scaleX;
                                const y1 = annotation.y1 * scaleY;
                                const x2 = annotation.x2 * scaleX;
                                const y2 = annotation.y2 * scaleY;
                                const width = x2 - x1;
                                const height = y2 - y1;
                                
                                ctx.strokeRect(x1, y1, width, height);
                                
                                const centerX = x1 + width / 2;
                                const centerY = y1 + height / 2;
                                
                                if (settings.showTrails) {{
                                    if (!trails[trackId]) trails[trackId] = [];
                                    trails[trackId].push({{x: centerX, y: centerY, frame: currentFrame}});
                                    trails[trackId] = trails[trackId].filter(pt => currentFrame - pt.frame < 30);
                                    
                                    if (trails[trackId].length > 1) {{
                                        ctx.beginPath();
                                        ctx.moveTo(trails[trackId][0].x, trails[trackId][0].y);
                                        for (let i = 1; i < trails[trackId].length; i++) {{
                                            ctx.lineTo(trails[trackId][i].x, trails[trackId][i].y);
                                        }}
                                        ctx.stroke();
                                    }}
                                }}
                                
                                if (settings.showIds) {{
                                    ctx.font = 'bold 16px Arial';
                                    ctx.fillText(`ID: ${{trackId}}`, x1, y1 - 8);
                                }}
                                
                            }} else if (annotation.type === 'centroid') {{
                                const x = annotation.x * scaleX;
                                const y = annotation.y * scaleY;
                                
                                ctx.beginPath();
                                ctx.arc(x, y, 8, 0, 2 * Math.PI);
                                ctx.fill();
                                
                                if (settings.showTrails) {{
                                    if (!trails[trackId]) trails[trackId] = [];
                                    trails[trackId].push({{x: x, y: y, frame: currentFrame}});
                                    trails[trackId] = trails[trackId].filter(pt => currentFrame - pt.frame < 30);
                                    
                                    if (trails[trackId].length > 1) {{
                                        ctx.beginPath();
                                        ctx.moveTo(trails[trackId][0].x, trails[trackId][0].y);
                                        for (let i = 1; i < trails[trackId].length; i++) {{
                                            ctx.lineTo(trails[trackId][i].x, trails[trackId][i].y);
                                        }}
                                        ctx.stroke();
                                    }}
                                }}
                                
                                if (settings.showIds) {{
                                    ctx.font = 'bold 16px Arial';
                                    ctx.fillText(`ID: ${{trackId}}`, x + 12, y - 12);
                                }}
                            }}
                        }});
                        
                        ctx.globalAlpha = 1.0;
                        
                        if (frameData.length > 0) {{
                            updateStatus(`Frame ${{currentFrame}}: ${{frameData.length}} objects tracked`);
                        }}
                    }}
                    
                    // Event listeners
                    video.addEventListener('timeupdate', drawOverlay);
                    video.addEventListener('seeked', drawOverlay);
                    video.addEventListener('loadeddata', drawOverlay);
                    video.addEventListener('loadedmetadata', drawOverlay);
                    
                    // Control listeners
                    if (showIdsCheck) {{
                        showIdsCheck.addEventListener('change', function() {{
                            settings.showIds = this.checked;
                            drawOverlay();
                        }});
                    }}
                    
                    if (showTrailsCheck) {{
                        showTrailsCheck.addEventListener('change', function() {{
                            settings.showTrails = this.checked;
                            if (!this.checked) trails = {{}};
                            drawOverlay();
                        }});
                    }}
                    
                    if (opacitySlider) {{
                        opacitySlider.addEventListener('input', function() {{
                            settings.opacity = parseFloat(this.value);
                            if (opacityValue) opacityValue.textContent = this.value;
                            drawOverlay();
                        }});
                    }}
                    
                    updateStatus('Video overlay ready! ✓');
                    setTimeout(drawOverlay, 100);
                    
                    console.log('✅ Panel Video Bbox Viewer initialized successfully');
                }}
                
                // Start initialization with multiple triggers
                // Try immediately
                initViewer();
                
                // Try when DOM is ready (if not already)
                if (document.readyState === 'loading') {{
                    document.addEventListener('DOMContentLoaded', initViewer);
                }}
                
                // Try when page is fully loaded
                window.addEventListener('load', initViewer);
                
                // Try after a longer delay (Panel sometimes takes time)
                setTimeout(initViewer, 1000);
                setTimeout(initViewer, 2000);
            }})();
            </script>
        </div>
        """
        
        return pn.pane.HTML(html_content, width=900, height=700)
    
    def panel(self) -> pn.pane.HTML:
        """Return the Panel component."""
        return self.component


def create_video_bbox_viewer(video_path: str, bbox_df: pd.DataFrame, fps: float = 30.0) -> pn.pane.HTML:
    """
    Create a working video bbox viewer for Panel.
    
    Args:
        video_path: Path to the MP4 video file
        bbox_df: DataFrame with bbox/centroid data
        fps: Video frame rate
        
    Returns:
        Panel HTML component ready for display
    """
    viewer = WorkingVideoBboxViewer(video_path, bbox_df, fps)
    return viewer.panel()


# Test data function
def create_test_data() -> pd.DataFrame:
    """Create sample bbox data for testing."""
    import numpy as np
    
    data = []
    for frame in range(300):
        for track_id in range(3):
            base_x = 100 + track_id * 200 + np.sin(frame * 0.1 + track_id) * 80
            base_y = 100 + np.cos(frame * 0.05 + track_id) * 60
            
            data.append({
                'track_id': track_id,
                'frame': frame,
                'x1': base_x - 30,
                'y1': base_y - 30,
                'x2': base_x + 30,
                'y2': base_y + 30
            })
    
    return pd.DataFrame(data)