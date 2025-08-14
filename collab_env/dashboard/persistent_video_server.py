#!/usr/bin/env python3
"""
Persistent Flask server for video bbox overlay viewing.
Supports dynamic loading of different video/CSV combinations.
"""

from flask import Flask, render_template_string, jsonify, send_file
import pandas as pd
from pathlib import Path
from threading import Lock

app = Flask(__name__)

# Thread-safe storage for multiple video/CSV combinations
videos_data = {}
data_lock = Lock()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Bbox Overlay Viewer</title>
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
        .header {
            background: #2196f3;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .video-selector {
            margin-bottom: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
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
        select {
            padding: 8px;
            border-radius: 4px;
            border: 1px solid #ddd;
            font-size: 14px;
        }
        button {
            padding: 8px 16px;
            margin-left: 10px;
            border: none;
            border-radius: 4px;
            background: #007bff;
            color: white;
            cursor: pointer;
        }
        button:hover {
            background: #0056b3;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎥 Persistent Video Bbox Overlay Viewer</h1>
            <p>Dynamic viewer for multiple video/tracking combinations</p>
        </div>
        
        <div class="video-selector">
            <label for="videoSelect"><strong>Select Video:</strong></label>
            <select id="videoSelect" onchange="loadVideo()">
                <option value="">Loading videos...</option>
            </select>
            <button onclick="refreshVideoList()">🔄 Refresh</button>
        </div>
        
        <div id="videoContent" style="display: none;">
            <div id="videoInfo" class="info"></div>
            
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
                    Your browser does not support the video tag.
                </video>
                <canvas id="overlayCanvas" width="800" height="450"></canvas>
            </div>

            <div id="status" class="status">Ready</div>
        </div>
    </div>

    <script>
        console.log('🚀 Persistent video overlay viewer starting');

        // Global state
        let currentVideoId = null;
        let currentBboxData = {};
        let currentFps = 30.0;
        let trails = {};

        // Get elements
        const videoSelect = document.getElementById('videoSelect');
        const videoContent = document.getElementById('videoContent');
        const videoInfo = document.getElementById('videoInfo');
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

        function updateStatus(message, isError = false) {
            status.textContent = message;
            status.className = 'status ' + (isError ? 'error' : 'success');
        }

        async function refreshVideoList() {
            try {
                const response = await fetch('/api/videos');
                const videos = await response.json();
                
                // Clear and repopulate dropdown
                videoSelect.innerHTML = '<option value="">Choose a video...</option>';
                
                videos.forEach(video => {
                    const option = document.createElement('option');
                    option.value = video.id;
                    option.textContent = `${video.name} (${video.tracks} tracks)`;
                    videoSelect.appendChild(option);
                });
                
                updateStatus(`Refreshed video list: ${videos.length} video(s) available`);
                
            } catch (error) {
                console.error('Error refreshing video list:', error);
                updateStatus('Error refreshing video list', true);
            }
        }

        async function loadVideo() {
            const videoId = videoSelect.value;
            if (!videoId) {
                videoContent.style.display = 'none';
                return;
            }

            try {
                updateStatus('Loading video data...');
                
                // Fetch video data
                const response = await fetch(`/api/video/${videoId}`);
                if (!response.ok) {
                    throw new Error(`Failed to load video: ${response.statusText}`);
                }
                
                const data = await response.json();
                currentVideoId = videoId;
                currentBboxData = data.bbox_data;
                currentFps = data.fps;
                
                // Update video info
                videoInfo.innerHTML = `
                    <div style="background: #e3f2fd; padding: 15px; border-radius: 5px; margin: 10px 0;">
                        <h3>📹 ${data.name}</h3>
                        <p><strong>Data Format:</strong> ${data.data_format}</p>
                        <p><strong>Tracks:</strong> ${data.num_tracks}</p>
                        <p><strong>Frames with data:</strong> ${data.num_frames}</p>
                        <p><strong>FPS:</strong> ${data.fps}</p>
                    </div>
                `;
                
                // Set video source
                video.src = `/api/video/${videoId}/stream`;
                
                // Show video content
                videoContent.style.display = 'block';
                
                // Reset trails
                trails = {};
                
                updateStatus(`Video loaded: ${data.name}`);
                console.log('📊 Loaded bbox data for', Object.keys(currentBboxData).length, 'frames');
                
            } catch (error) {
                console.error('Error loading video:', error);
                updateStatus(`Error: ${error.message}`, true);
            }
        }

        // Draw overlay function (same as before)
        function drawOverlay() {
            if (!currentVideoId) return;
            
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
            const currentFrame = Math.floor(currentTime * currentFps);
            const frameData = currentBboxData[currentFrame] || [];

            if (frameData.length === 0) {
                return;
            }

            // Calculate scaling accounting for letterboxing/pillarboxing
            const videoAspect = video.videoWidth / video.videoHeight;
            const containerAspect = canvas.width / canvas.height;
            
            let displayWidth, displayHeight, offsetX, offsetY;
            
            if (videoAspect > containerAspect) {
                displayWidth = canvas.width;
                displayHeight = canvas.width / videoAspect;
                offsetX = 0;
                offsetY = (canvas.height - displayHeight) / 2;
            } else {
                displayWidth = canvas.height * videoAspect;
                displayHeight = canvas.height;
                offsetX = (canvas.width - displayWidth) / 2;
                offsetY = 0;
            }
            
            const scaleX = displayWidth / video.videoWidth;
            const scaleY = displayHeight / video.videoHeight;
            
            // Debug coordinate system
            if (settings.showDebug) {
                ctx.fillStyle = 'rgba(255, 255, 0, 0.8)';
                ctx.fillRect(200, 10, 450, 100);
                ctx.fillStyle = 'black';
                ctx.font = '12px monospace';
                ctx.fillText(`Video: ${video.videoWidth}x${video.videoHeight}`, 210, 30);
                ctx.fillText(`Canvas: ${canvas.width}x${canvas.height}`, 210, 45);
                ctx.fillText(`Display: ${displayWidth.toFixed(0)}x${displayHeight.toFixed(0)}`, 210, 60);
                ctx.fillText(`Offset: ${offsetX.toFixed(0)}, ${offsetY.toFixed(0)}`, 210, 75);
                ctx.fillText(`Scale: ${scaleX.toFixed(3)}, ${scaleY.toFixed(3)}`, 210, 90);
                
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
                    const x1 = annotation.x1 * scaleX + offsetX;
                    const y1 = annotation.y1 * scaleY + offsetY;
                    const x2 = annotation.x2 * scaleX + offsetX;
                    const y2 = annotation.y2 * scaleY + offsetY;
                    const width = x2 - x1;
                    const height = y2 - y1;

                    ctx.strokeRect(x1, y1, width, height);

                    const centerX = x1 + width / 2;
                    const centerY = y1 + height / 2;

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
                        ctx.font = 'bold 16px Arial';
                        ctx.fillStyle = color;
                        const labelY = y1 > 25 ? y1 - 8 : y2 + 20;
                        ctx.fillText(`ID: ${trackId}`, x1, labelY);
                    }
                }
            });

            ctx.globalAlpha = 1.0;

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
                trails = {};
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

        // Initialize
        updateStatus('Loading available videos...');
        refreshVideoList();
        console.log('✅ Persistent video overlay viewer ready');
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main page with video selector and viewer."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/video/<video_id>')
def get_video_data(video_id):
    """Get video metadata and bbox data."""
    with data_lock:
        if video_id not in videos_data:
            return jsonify({'error': 'Video not found'}), 404
        
        video_info = videos_data[video_id]
        
        # Detect data format
        if video_info['bbox_data'] and 'x1' in list(video_info['bbox_data'].values())[0][0]:
            data_format = "Bounding Boxes (x1, y1, x2, y2)"
        else:
            data_format = "Centroids (x, y)"
        
        # Get track info
        track_ids = set()
        for frame_data in video_info['bbox_data'].values():
            for annotation in frame_data:
                track_ids.add(annotation['track_id'])
        
        return jsonify({
            'name': video_info['name'],
            'data_format': data_format,
            'num_tracks': len(track_ids),
            'num_frames': len(video_info['bbox_data']),
            'fps': video_info['fps'],
            'bbox_data': video_info['bbox_data']
        })

@app.route('/api/video/<video_id>/stream')
def stream_video(video_id):
    """Stream video file."""
    with data_lock:
        if video_id not in videos_data:
            return "Video not found", 404
        
        video_path = videos_data[video_id]['video_path']
        return send_file(video_path, mimetype='video/mp4')

@app.route('/api/add_video', methods=['POST'])
def api_add_video():
    """API endpoint to add a video/CSV combination."""
    try:
        from flask import request
        import traceback
        
        print(f"🔄 API add_video called")
        
        # Get JSON data
        data = request.get_json()
        if not data:
            error_msg = "No JSON data received"
            print(f"❌ {error_msg}")
            return jsonify({'error': error_msg}), 400
        
        print(f"📝 Received data: {data}")
        
        video_id = data.get('video_id')
        video_path = data.get('video_path')
        csv_path = data.get('csv_path')
        fps = data.get('fps', 30.0)
        
        print(f"📹 Video ID: {video_id}")
        print(f"🎥 Video path: {video_path}")
        print(f"📊 CSV path: {csv_path}")
        print(f"⏱️ FPS: {fps}")
        
        if not all([video_id, video_path, csv_path]):
            error_msg = f"Missing fields - video_id: {bool(video_id)}, video_path: {bool(video_path)}, csv_path: {bool(csv_path)}"
            print(f"❌ {error_msg}")
            return jsonify({'error': error_msg}), 400
        
        # Check if files exist
        if not Path(video_path).exists():
            error_msg = f"Video file not found: {video_path}"
            print(f"❌ {error_msg}")
            return jsonify({'error': error_msg}), 400
            
        if not Path(csv_path).exists():
            error_msg = f"CSV file not found: {csv_path}"
            print(f"❌ {error_msg}")
            return jsonify({'error': error_msg}), 400
        
        # Add video using internal function
        print(f"🚀 Adding video to server...")
        success = add_video(video_id, video_path, csv_path, fps)
        
        if success:
            print(f"✅ Video added successfully: {video_id}")
            return jsonify({'success': True, 'video_id': video_id})
        else:
            error_msg = "Failed to add video (unknown error)"
            print(f"❌ {error_msg}")
            return jsonify({'error': error_msg}), 500
            
    except Exception as e:
        error_msg = f"Exception in add_video: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"📜 Traceback: {traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500

@app.route('/api/videos')
def list_videos_api():
    """API endpoint to list all videos."""
    with data_lock:
        video_list = []
        for video_id, info in videos_data.items():
            video_list.append({
                'id': video_id,
                'name': info['name'],
                'tracks': len(set(
                    ann['track_id'] 
                    for frame_data in info['bbox_data'].values() 
                    for ann in frame_data
                ))
            })
        print(f"📝 API /api/videos returning {len(video_list)} videos")
        return jsonify(video_list)

@app.route('/api/health')
def health_check():
    """Simple health check endpoint."""
    with data_lock:
        return jsonify({
            'status': 'ok',
            'videos_count': len(videos_data),
            'videos': list(videos_data.keys())
        })

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
            annotation = {
                'track_id': track_id,
                'x': float(row['x']),
                'y': float(row['y']),
                'type': 'centroid'
            }
        
        data_by_frame[frame].append(annotation)
    
    return data_by_frame

def add_video(video_id: str, video_path: str, csv_path: str, fps: float = 30.0):
    """Add a video/CSV combination to the server."""
    try:
        # Load CSV data
        df = pd.read_csv(csv_path)
        bbox_data = prepare_bbox_data(df)
        
        with data_lock:
            videos_data[video_id] = {
                'name': Path(video_path).name,
                'video_path': video_path,
                'csv_path': csv_path,
                'fps': fps,
                'bbox_data': bbox_data
            }
        
        print(f"✅ Added video: {video_id} ({Path(video_path).name})")
        return True
        
    except Exception as e:
        print(f"❌ Failed to add video {video_id}: {e}")
        return False

def remove_video(video_id: str):
    """Remove a video from the server."""
    with data_lock:
        if video_id in videos_data:
            del videos_data[video_id]
            print(f"🗑️  Removed video: {video_id}")
            return True
        return False

def list_videos():
    """List all loaded videos."""
    with data_lock:
        return list(videos_data.keys())

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Persistent Video Bbox Overlay Server')
    parser.add_argument('--port', type=int, default=5050, help='Port to run on')
    parser.add_argument('--host', default='localhost', help='Host to run on')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting persistent video server at http://{args.host}:{args.port}")
    print("📝 API endpoints available:")
    print("  - GET  / (homepage)")
    print("  - POST /api/add_video (add video/CSV)")
    print("  - GET  /api/videos (list videos)")
    print("  - GET  /api/health (health check)")
    print("  - GET  /api/video/<id> (video data)")
    print("  - GET  /api/video/<id>/stream (video stream)")
    
    app.run(host=args.host, port=args.port, debug=True, threaded=True)  # Enable debug for better error messages