#!/usr/bin/env python3
"""
Simple, robust video overlay web application using Flask.
This WILL work on any modern browser.
"""

from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import json
import argparse
from pathlib import Path
import numpy as np

app = Flask(__name__)

# Global variables to store data
video_path = None
bbox_data = None
fps = 30.0

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Bbox Overlay</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f0f0f0;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .video-container {
            position: relative;
            display: inline-block;
            margin: 20px 0;
            border: 3px solid #333;
            border-radius: 5px;
        }
        video {
            display: block;
            width: 800px;
            height: 450px;
        }
        canvas {
            position: absolute;
            top: 0;
            left: 0;
            pointer-events: none;
            z-index: 10;
            width: 800px;
            height: 450px;
        }
        .controls {
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
            border: 1px solid #dee2e6;
        }
        .control-group {
            margin: 10px 0;
        }
        label {
            display: inline-block;
            margin-right: 20px;
            font-weight: bold;
        }
        input[type="range"] {
            width: 200px;
            margin: 0 10px;
        }
        .status {
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
            font-size: 14px;
        }
        .status.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .info {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 4px solid #2196f3;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎥 Video Bbox Overlay Viewer</h1>
        
        <div class="info">
            <h3>Video Information</h3>
            <p><strong>File:</strong> {{ video_name }}</p>
            <p><strong>Data Format:</strong> {{ data_format }}</p>
            <p><strong>Tracks:</strong> {{ num_tracks }}</p>
            <p><strong>Frames with data:</strong> {{ num_frames }}</p>
            <p><strong>FPS:</strong> {{ fps }}</p>
        </div>

        <div class="controls">
            <div class="control-group">
                <label>
                    <input type="checkbox" id="showIds" checked>
                    Show Track IDs
                </label>
                <label>
                    <input type="checkbox" id="showTrails">
                    Show Movement Trails
                </label>
                <label>
                    <input type="checkbox" id="showDebug">
                    Show Coordinate Debug
                </label>
            </div>
            <div class="control-group">
                <label>
                    Opacity: 
                    <input type="range" id="opacity" min="0.1" max="1" step="0.1" value="0.8">
                    <span id="opacityValue">0.8</span>
                </label>
            </div>
        </div>

        <div class="video-container">
            <video id="videoPlayer" controls>
                <source src="/video" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            <canvas id="overlayCanvas" width="800" height="450"></canvas>
        </div>

        <div id="status" class="status">Loading...</div>
    </div>

    <script>
        console.log('🚀 Video overlay app starting');

        // Get elements
        const video = document.getElementById('videoPlayer');
        const canvas = document.getElementById('overlayCanvas');
        const ctx = canvas.getContext('2d');
        const status = document.getElementById('status');
        
        // Controls
        const showIdsCheck = document.getElementById('showIds');
        const showTrailsCheck = document.getElementById('showTrails');
        const showDebugCheck = document.getElementById('showDebug');
        const opacitySlider = document.getElementById('opacity');
        const opacityValue = document.getElementById('opacityValue');

        // Settings
        let settings = {
            showIds: true,
            showTrails: false,
            showDebug: false,
            opacity: 0.8
        };

        // Data
        let bboxData = {{ bbox_data | safe }};
        let fps = {{ fps }};
        let trails = {};

        console.log('📊 Loaded bbox data for', Object.keys(bboxData).length, 'frames');

        // Update status
        function updateStatus(message, isError = false) {
            status.textContent = message;
            status.className = 'status ' + (isError ? 'error' : 'success');
        }

        // Draw overlay
        function drawOverlay() {
            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Always show overlay is working
            ctx.fillStyle = 'rgba(0, 255, 0, 0.8)';
            ctx.fillRect(10, 10, 180, 30);
            ctx.fillStyle = 'black';
            ctx.font = 'bold 16px Arial';
            ctx.fillText('✓ OVERLAY ACTIVE', 15, 30);

            // Check if video is ready
            if (!video.videoWidth || !video.videoHeight) {
                return;
            }

            // Get current frame
            const currentTime = video.currentTime;
            const currentFrame = Math.floor(currentTime * fps);
            const frameData = bboxData[currentFrame] || [];

            if (frameData.length === 0) {
                return;
            }

            // Calculate scaling accounting for letterboxing/pillarboxing
            const videoRect = video.getBoundingClientRect();
            const canvasRect = canvas.getBoundingClientRect();
            
            // Calculate the actual displayed video area (accounting for aspect ratio)
            const videoAspect = video.videoWidth / video.videoHeight;
            const containerAspect = canvas.width / canvas.height;
            
            let displayWidth, displayHeight, offsetX, offsetY;
            
            if (videoAspect > containerAspect) {
                // Video is wider - letterboxing (black bars top/bottom)
                displayWidth = canvas.width;
                displayHeight = canvas.width / videoAspect;
                offsetX = 0;
                offsetY = (canvas.height - displayHeight) / 2;
            } else {
                // Video is taller - pillarboxing (black bars left/right)  
                displayWidth = canvas.height * videoAspect;
                displayHeight = canvas.height;
                offsetX = (canvas.width - displayWidth) / 2;
                offsetY = 0;
            }
            
            // Calculate scale based on actual video display area
            const scaleX = displayWidth / video.videoWidth;
            const scaleY = displayHeight / video.videoHeight;
            
            // Debug coordinate system (only when debug is enabled)
            if (settings.showDebug) {
                console.log('Video dimensions:', video.videoWidth, 'x', video.videoHeight);
                console.log('Canvas dimensions:', canvas.width, 'x', canvas.height);
                console.log('Display area:', displayWidth, 'x', displayHeight);
                console.log('Offsets:', offsetX, offsetY);
                console.log('Scale factors:', scaleX, scaleY);
                
                // Draw coordinate system debug info on canvas
                ctx.fillStyle = 'rgba(255, 255, 0, 0.8)';
                ctx.fillRect(200, 10, 450, 100);
                ctx.fillStyle = 'black';
                ctx.font = '12px monospace';
                ctx.fillText(`Video: ${video.videoWidth}x${video.videoHeight}`, 210, 30);
                ctx.fillText(`Canvas: ${canvas.width}x${canvas.height}`, 210, 45);
                ctx.fillText(`Display: ${displayWidth.toFixed(0)}x${displayHeight.toFixed(0)}`, 210, 60);
                ctx.fillText(`Offset: ${offsetX.toFixed(0)}, ${offsetY.toFixed(0)}`, 210, 75);
                ctx.fillText(`Scale: ${scaleX.toFixed(3)}, ${scaleY.toFixed(3)}`, 210, 90);
                
                // Draw the actual video display area
                ctx.strokeStyle = 'yellow';
                ctx.lineWidth = 2;
                ctx.strokeRect(offsetX, offsetY, displayWidth, displayHeight);
            }

            // Set opacity
            ctx.globalAlpha = settings.opacity;

            // Colors for different tracks
            const colors = [
                '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
                '#FFEAA7', '#DDA0DD', '#F7DC6F', '#BB8FCE'
            ];

            // Draw annotations
            frameData.forEach(annotation => {
                const trackId = annotation.track_id;
                const color = colors[trackId % colors.length];
                
                ctx.strokeStyle = color;
                ctx.fillStyle = color;
                ctx.lineWidth = 3;

                if (annotation.type === 'bbox') {
                    // Draw bounding box with proper offset for letterboxing
                    const x1 = annotation.x1 * scaleX + offsetX;
                    const y1 = annotation.y1 * scaleY + offsetY;
                    const x2 = annotation.x2 * scaleX + offsetX;
                    const y2 = annotation.y2 * scaleY + offsetY;
                    const width = x2 - x1;
                    const height = y2 - y1;

                    ctx.strokeRect(x1, y1, width, height);

                    // Center point for trails
                    const centerX = x1 + width / 2;
                    const centerY = y1 + height / 2;

                    // Handle trails
                    if (settings.showTrails) {
                        if (!trails[trackId]) trails[trackId] = [];
                        trails[trackId].push({x: centerX, y: centerY, frame: currentFrame});
                        
                        // Keep only recent points
                        trails[trackId] = trails[trackId].filter(point => 
                            currentFrame - point.frame < 30
                        );

                        // Draw trail
                        if (trails[trackId].length > 1) {
                            ctx.beginPath();
                            ctx.moveTo(trails[trackId][0].x, trails[trackId][0].y);
                            for (let i = 1; i < trails[trackId].length; i++) {
                                ctx.lineTo(trails[trackId][i].x, trails[trackId][i].y);
                            }
                            ctx.stroke();
                        }
                    }

                    // Draw ID label
                    if (settings.showIds) {
                        ctx.font = 'bold 16px Arial';
                        ctx.fillStyle = color;
                        const labelY = y1 > 25 ? y1 - 8 : y2 + 20;
                        ctx.fillText(`ID: ${trackId}`, x1, labelY);
                    }

                } else if (annotation.type === 'centroid') {
                    // Draw centroid as bounding box for better visual alignment
                    const centerX = annotation.x * scaleX + offsetX;
                    const centerY = annotation.y * scaleY + offsetY;
                    
                    // Create a bounding box around the centroid (adjustable size)
                    const boxSize = 40; // Adjust this size as needed
                    const x1 = centerX - boxSize/2;
                    const y1 = centerY - boxSize/2;
                    const width = boxSize;
                    const height = boxSize;
                    
                    // Draw bounding box
                    ctx.strokeRect(x1, y1, width, height);
                    
                    // Draw crosshair at exact centroid position
                    ctx.beginPath();
                    ctx.moveTo(centerX - 10, centerY);
                    ctx.lineTo(centerX + 10, centerY);
                    ctx.moveTo(centerX, centerY - 10);
                    ctx.lineTo(centerX, centerY + 10);
                    ctx.stroke();
                    
                    // Draw center dot
                    ctx.beginPath();
                    ctx.arc(centerX, centerY, 3, 0, 2 * Math.PI);
                    ctx.fill();

                    // Handle trails
                    if (settings.showTrails) {
                        if (!trails[trackId]) trails[trackId] = [];
                        trails[trackId].push({x: centerX, y: centerY, frame: currentFrame});
                        
                        trails[trackId] = trails[trackId].filter(point => 
                            currentFrame - point.frame < 30
                        );

                        if (trails[trackId].length > 1) {
                            ctx.beginPath();
                            ctx.moveTo(trails[trackId][0].x, trails[trackId][0].y);
                            for (let i = 1; i < trails[trackId].length; i++) {
                                ctx.lineTo(trails[trackId][i].x, trails[trackId][i].y);
                            }
                            ctx.stroke();
                        }
                    }

                    // Draw ID label
                    if (settings.showIds) {
                        ctx.font = 'bold 14px Arial';
                        ctx.fillStyle = color;
                        ctx.fillText(`ID: ${trackId}`, x1, y1 - 5);
                        
                        // Also show exact coordinates for debugging
                        if (settings.showDebug) {
                            ctx.font = '10px monospace';
                            ctx.fillStyle = color;
                            ctx.fillText(`(${annotation.x}, ${annotation.y})`, x1, y1 + height + 15);
                        }
                    }
                }
            });

            ctx.globalAlpha = 1.0;

            // Update status
            if (frameData.length > 0) {
                updateStatus(`Frame ${currentFrame}: ${frameData.length} objects tracked`);
            }
        }

        // Event listeners
        video.addEventListener('timeupdate', drawOverlay);
        video.addEventListener('seeked', drawOverlay);
        video.addEventListener('loadeddata', drawOverlay);
        video.addEventListener('loadedmetadata', drawOverlay);

        // Control event listeners
        showIdsCheck.addEventListener('change', function() {
            settings.showIds = this.checked;
            drawOverlay();
        });

        showTrailsCheck.addEventListener('change', function() {
            settings.showTrails = this.checked;
            if (!this.checked) {
                trails = {}; // Clear trails
            }
            drawOverlay();
        });

        showDebugCheck.addEventListener('change', function() {
            settings.showDebug = this.checked;
            drawOverlay();
        });

        opacitySlider.addEventListener('input', function() {
            settings.opacity = parseFloat(this.value);
            opacityValue.textContent = this.value;
            drawOverlay();
        });

        // Initial setup
        updateStatus('Video overlay initialized successfully! ✓');
        
        // Initial draw
        setTimeout(drawOverlay, 100);

        console.log('✅ Video overlay app ready');
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main page with video overlay."""
    global video_path, bbox_data, fps
    
    if not video_path or not bbox_data:
        return "No video or bbox data loaded. Run with --video and --csv arguments."
    
    # Detect data format
    if 'x1' in bbox_data[0] if bbox_data else False:
        data_format = "Bounding Boxes (x1, y1, x2, y2)"
    else:
        data_format = "Centroids (x, y)"
    
    # Get track info
    track_ids = set()
    for frame_data in bbox_data.values():
        for annotation in frame_data:
            track_ids.add(annotation['track_id'])
    
    return render_template_string(HTML_TEMPLATE,
        video_name=Path(video_path).name,
        data_format=data_format,
        num_tracks=len(track_ids),
        num_frames=len(bbox_data),
        fps=fps,
        bbox_data=json.dumps(bbox_data)
    )

@app.route('/video')
def serve_video():
    """Serve the video file."""
    from flask import send_file
    return send_file(video_path, mimetype='video/mp4')

def prepare_bbox_data(df: pd.DataFrame) -> dict:
    """Convert DataFrame to frame-indexed dictionary."""
    data_by_frame = {}
    
    # Detect format
    if 'x1' in df.columns:
        data_format = "bbox"
        required_cols = ['x1', 'y1', 'x2', 'y2']
    else:
        data_format = "centroid"
        required_cols = ['x', 'y']
    
    for _, row in df.iterrows():
        frame = int(row['frame'])
        track_id = row['track_id']
        
        if frame not in data_by_frame:
            data_by_frame[frame] = []
        
        if data_format == "bbox":
            annotation = {
                'track_id': track_id,
                'x1': float(row['x1']),
                'y1': float(row['y1']),
                'x2': float(row['x2']),
                'y2': float(row['y2']),
                'type': 'bbox'
            }
        else:
            # Convert centroid to bounding box with estimated size
            center_x = float(row['x'])
            center_y = float(row['y'])
            
            # Estimate bounding box size (adjust these values based on your objects)
            # You can make this dynamic based on track_id or other factors
            bbox_width = 50   # Estimated width of tracked objects
            bbox_height = 50  # Estimated height of tracked objects
            
            annotation = {
                'track_id': track_id,
                'x1': center_x - bbox_width/2,
                'y1': center_y - bbox_height/2, 
                'x2': center_x + bbox_width/2,
                'y2': center_y + bbox_height/2,
                'type': 'bbox'  # Treat as bbox now
            }
        
        data_by_frame[frame].append(annotation)
    
    return data_by_frame

def create_test_data() -> pd.DataFrame:
    """Create sample bbox data for testing."""
    data = []
    for frame in range(300):  # More frames for testing
        for track_id in range(3):
            # Simulate moving objects
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

def main():
    global video_path, bbox_data, fps
    
    parser = argparse.ArgumentParser(description='Video Bbox Overlay Web App')
    parser.add_argument('--video', required=True, help='Path to MP4 video file')
    parser.add_argument('--csv', help='Path to CSV bbox file (optional, will use synthetic data if not provided)')
    parser.add_argument('--fps', type=float, default=30.0, help='Video FPS')
    parser.add_argument('--port', type=int, default=5009, help='Port to run on')
    parser.add_argument('--host', default='localhost', help='Host to run on')
    
    args = parser.parse_args()
    
    # Set global variables
    video_path = args.video
    fps = args.fps
    
    # Validate video file
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        return 1
    
    # Load bbox data
    if args.csv:
        if not Path(args.csv).exists():
            print(f"❌ CSV file not found: {args.csv}")
            return 1
        df = pd.read_csv(args.csv)
        print(f"📊 Loaded {len(df)} annotations from {args.csv}")
    else:
        df = create_test_data()
        print(f"📊 Generated {len(df)} synthetic annotations")
    
    bbox_data = prepare_bbox_data(df)
    
    print(f"🎥 Video: {Path(video_path).name}")
    print(f"📈 Tracks: {len(set(df['track_id']))}")
    print(f"🎞️  Frames with data: {len(bbox_data)}")
    print(f"🚀 Starting server at http://{args.host}:{args.port}")
    
    app.run(host=args.host, port=args.port, debug=False)
    
    return 0

if __name__ == '__main__':
    exit(main())