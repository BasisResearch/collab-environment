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
videos_data: dict[str, dict] = {}
data_lock = Lock()

# Thread-safe storage for 3D mesh/track combinations
meshes_data: dict[str, dict] = {}

def convert_camera_params_to_json(params):
    """Convert camera parameters to JSON-serializable format."""
    try:
        import numpy as np
        
        def convert_value(obj):
            """Recursively convert numpy arrays and other types to JSON-serializable format."""
            if obj is None:
                return None
            elif isinstance(obj, np.ndarray):
                return obj.tolist()  # Convert numpy arrays to lists
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()  # Convert numpy scalars to Python scalars
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_value(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_value(v) for v in obj]
            elif hasattr(obj, 'cpu') and hasattr(obj, 'numpy'):
                # PyTorch tensor
                return convert_value(obj.cpu().numpy())
            elif isinstance(obj, (int, float, str)):
                return obj  # Already JSON serializable
            else:
                # Try to convert unknown objects to string as fallback
                print(f"⚠️ Unknown type {type(obj)}, converting to string: {obj}")
                return str(obj)
        
        if params is None:
            print("⚠️ Input params is None")
            return None
        
        print(f"🔄 Starting conversion of params type: {type(params)}")
        result = convert_value(params)
        print(f"✅ Conversion successful. Result type: {type(result)}")
        
        # Additional validation
        if result is None:
            print("⚠️ Conversion returned None!")
        elif isinstance(result, dict) and len(result) == 0:
            print("⚠️ Conversion returned empty dict!")
        else:
            print(f"✅ Conversion result looks good: {len(str(result))} characters")
        
        return result
        
    except Exception as e:
        print(f"❌ Error converting camera params: {e}")
        import traceback
        traceback.print_exc()
        return None

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
        let videoActualFps = null;  // Will be set from video metadata
        let trails = {};
        
        // Clear all state when switching videos
        function clearVideoState() {
            currentVideoId = null;
            currentBboxData = {};
            currentFps = 30.0;
            videoActualFps = null;
            trails = {};
            
            // Clear canvas
            if (ctx) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
            
            // Reset video
            if (video) {
                video.src = '';
                video.load();
            }
            
            console.log('🧽 Cleared all video state');
        }

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
                clearVideoState();
                return;
            }
            
            // Clear previous video state to prevent contamination
            if (currentVideoId !== videoId) {
                console.log(`🔄 Switching from ${currentVideoId} to ${videoId}`);
                clearVideoState();
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
                currentFps = data.fps;  // This is the FPS we calculated from, keep as reference
                
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
                
                // Reset trails
                trails = {};
                
                // Show video content
                videoContent.style.display = 'block';
                
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

            // Get current frame - use the video's actual timing
            const currentTime = video.currentTime;
            const usedFps = videoActualFps || currentFps;
            const currentFrame = Math.floor(currentTime * usedFps);
            const frameData = currentBboxData[currentFrame] || [];
            
            // Debug: log timing info occasionally
            if (Math.floor(currentTime) !== Math.floor((window.lastLoggedTime || 0)) && settings.showDebug) {
                console.log(`🕰️ Time: ${currentTime.toFixed(2)}s, Frame: ${currentFrame}, FPS used: ${usedFps.toFixed(2)}, Data points: ${frameData.length}`);
                window.lastLoggedTime = currentTime;
            }

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

            // Update status with timing info
            if (frameData.length > 0) {
                const timingInfo = videoActualFps ? ` (${usedFps.toFixed(1)} fps)` : '';
                updateStatus(`Frame ${currentFrame}${timingInfo}: ${frameData.length} objects tracked`);
            } else if (settings.showDebug) {
                updateStatus(`Frame ${currentFrame}: No data (time: ${currentTime.toFixed(2)}s)`);
            }
        }

        // Event listeners
        video.addEventListener('timeupdate', drawOverlay);
        video.addEventListener('seeked', drawOverlay);
        video.addEventListener('loadeddata', drawOverlay);
        video.addEventListener('loadedmetadata', function() {
            // Try to detect the video's actual frame rate from duration and estimated frame count
            // This is an approximation, but better than using a fixed FPS
            if (video.duration && Object.keys(currentBboxData).length > 0) {
                const maxFrame = Math.max(...Object.keys(currentBboxData).map(f => parseInt(f)));
                const estimatedFps = maxFrame / video.duration;
                if (estimatedFps > 5 && estimatedFps < 120) {  // Sanity check
                    videoActualFps = estimatedFps;
                    console.log(`📽️ Estimated video FPS: ${videoActualFps.toFixed(2)} (duration: ${video.duration.toFixed(2)}s, max frame: ${maxFrame})`);
                } else {
                    videoActualFps = currentFps;
                    console.log(`⚠️ FPS estimation failed (${estimatedFps.toFixed(2)}), using CSV FPS: ${currentFps}`);
                }
            } else {
                videoActualFps = currentFps;
                console.log(`📊 Using CSV FPS: ${currentFps}`);
            }
            drawOverlay();
        });

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

MESH_3D_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Mesh Track Viewer</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            overflow: hidden;
            background: #1a1a1a;
        }
        #container {
            width: 100vw;
            height: 100vh;
            position: relative;
        }
        #canvas {
            width: 100%;
            height: 100%;
            display: block;
        }
        #controls {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            width: 300px;
            z-index: 100;
        }
        #controls h2 {
            margin-top: 0;
            color: #333;
        }
        .control-group {
            margin: 15px 0;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        .control-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        select, button {
            width: 100%;
            padding: 8px;
            margin: 5px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }
        button {
            background: #007bff;
            color: white;
            cursor: pointer;
            border: none;
        }
        button:hover {
            background: #0056b3;
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        input[type="range"] {
            width: 100%;
            margin: 10px 0;
        }
        input[type="checkbox"] {
            margin-right: 10px;
        }
        #status {
            margin-top: 10px;
            padding: 10px;
            background: #e8f5e9;
            border-left: 4px solid #4caf50;
            border-radius: 4px;
            color: #2e7d32;
            font-size: 13px;
        }
        #status.error {
            background: #ffebee;
            border-left-color: #f44336;
            color: #c62828;
        }
        #frame-info {
            font-family: monospace;
            background: #333;
            color: #0f0;
            padding: 5px 10px;
            border-radius: 3px;
            margin: 5px 0;
        }
        .loader {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 24px;
            color: white;
            z-index: 1000;
        }
    </style>
</head>
<body>
    <div id="container">
        <div class="loader" id="loader">Loading 3D Viewer...</div>
        <div id="canvas"></div>
        
        <div id="controls">
            <h2>🌍 3D Track Viewer</h2>
            
            <div class="control-group">
                <label>Select Mesh:</label>
                <select id="meshSelect">
                    <option value="">Loading meshes...</option>
                </select>
                <button onclick="refreshMeshList()">🔄 Refresh</button>
            </div>
            
            <div id="meshControls" style="display: none;">
                <div class="control-group">
                    <label>Playback Controls:</label>
                    <button id="playBtn" onclick="togglePlayback()">▶️ Play</button>
                    <button onclick="resetAnimation()">⏮️ Reset</button>
                    <div id="frame-info">Frame: 0 / 0</div>
                    <label>
                        Speed: <span id="speedValue">1.0x</span>
                        <input type="range" id="speedSlider" min="0.1" max="3" step="0.1" value="1">
                    </label>
                    <label>
                        Frame: <span id="frameValue">0</span>
                        <input type="range" id="frameSlider" min="0" max="100" value="0">
                    </label>
                </div>
                
                <div class="control-group">
                    <label>Display Options:</label>
                    <label>
                        <input type="checkbox" id="showTrails" onchange="updateTrailVisibility()">
                        Show Trails
                    </label>
                    <label>
                        <input type="checkbox" id="showIds" checked onchange="updateLabelVisibility()">
                        Show Track IDs
                    </label>
                    <label>
                        <input type="checkbox" id="showMesh" checked onchange="updateMeshVisibility()">
                        Show Mesh
                    </label>
                    <label>
                        <input type="checkbox" id="showCamera" onchange="updateCameraVisibility()" disabled>
                        Show Camera Frustum
                    </label>
                    <label>
                        Sphere Size: <span id="sphereSizeValue">0.01</span>
                        <input type="range" id="sphereSizeSlider" min="0.001" max="0.05" step="0.001" value="0.01">
                    </label>
                </div>
                
                <div id="status">Ready</div>
            </div>
        </div>
    </div>

    <script type="importmap">
        {
            "imports": {
                "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
                "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
            }
        }
    </script>
    
    <script type="module">
        import * as THREE from 'three';
        import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
        import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
        import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
        
        console.log('🚀 Starting 3D Track Viewer');
        
        // Global state
        let scene, camera, renderer, controls, labelRenderer;
        let meshObject = null;
        let spheres = {};  // track_id -> sphere mesh
        let trails = {};   // track_id -> line geometry
        let labels = {};   // track_id -> CSS2D label
        let cameraFrustum = null;
        
        let currentMeshId = null;
        let trackData = null;
        let currentFrame = 0;
        let maxFrame = 0;
        let isPlaying = false;
        let animationSpeed = 1.0;
        let lastAnimationTime = 0;
        
        // Track colors
        const trackColors = {};
        const colorPalette = [
            0xff6b6b, 0x4ecdc4, 0x45b7d1, 0xffd93d,
            0x6bcf7f, 0xa8e6cf, 0xffeaa7, 0xdfe4ea,
            0xff9ff3, 0xfeca57, 0x48dbfb, 0xff6348
        ];
        
        // Initialize Three.js scene
        function initScene() {
            const container = document.getElementById('canvas');
            
            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x2a2a2a);
            scene.fog = new THREE.Fog(0x2a2a2a, 0.1, 100);
            
            // Camera
            camera = new THREE.PerspectiveCamera(
                60, 
                window.innerWidth / window.innerHeight, 
                0.001, 
                1000
            );
            camera.position.set(0.5, 0.5, 0.5);
            camera.lookAt(0, 0, 0);
            
            // Renderer
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            container.appendChild(renderer.domElement);
            
            // CSS2D Renderer for labels
            labelRenderer = new CSS2DRenderer();
            labelRenderer.setSize(window.innerWidth, window.innerHeight);
            labelRenderer.domElement.style.position = 'absolute';
            labelRenderer.domElement.style.top = '0px';
            labelRenderer.domElement.style.pointerEvents = 'none';
            container.appendChild(labelRenderer.domElement);
            
            // Controls
            controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.screenSpacePanning = false;
            controls.minDistance = 0.01;
            controls.maxDistance = 10;
            
            // Lights
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(1, 1, 1);
            directionalLight.castShadow = true;
            directionalLight.shadow.camera.near = 0.01;
            directionalLight.shadow.camera.far = 10;
            directionalLight.shadow.camera.left = -1;
            directionalLight.shadow.camera.right = 1;
            directionalLight.shadow.camera.top = 1;
            directionalLight.shadow.camera.bottom = -1;
            scene.add(directionalLight);
            
            // Grid helper
            const gridHelper = new THREE.GridHelper(2, 20, 0x444444, 0x222222);
            gridHelper.rotation.x = Math.PI / 2;
            scene.add(gridHelper);
            
            // Axes helper
            const axesHelper = new THREE.AxesHelper(0.2);
            scene.add(axesHelper);
            
            // Handle window resize
            window.addEventListener('resize', onWindowResize);
            
            // Hide loader
            document.getElementById('loader').style.display = 'none';
            
            console.log('✅ Scene initialized');
        }
        
        function onWindowResize() {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
            labelRenderer.setSize(window.innerWidth, window.innerHeight);
        }
        
        // Load mesh from server
        async function loadMesh(meshId) {
            try {
                updateStatus('Loading mesh data...');
                
                // Clear existing objects
                clearScene();
                
                // Fetch mesh metadata
                const response = await fetch(`/api/mesh/${meshId}`);
                if (!response.ok) throw new Error('Failed to load mesh metadata');
                
                const data = await response.json();
                trackData = data.tracks_by_frame;
                maxFrame = data.num_frames - 1;
                
                // Debug: Log sample of track data structure
                console.log('Track data loaded. Sample frames:', Object.keys(trackData).slice(0, 5));
                if (Object.keys(trackData).length > 0) {
                    const firstFrame = Object.keys(trackData)[0];
                    console.log(`Sample frame ${firstFrame} data:`, trackData[firstFrame]);
                }
                
                // Update UI
                document.getElementById('frameSlider').max = maxFrame;
                document.getElementById('meshControls').style.display = 'block';
                
                // Load PLY mesh
                const loader = new PLYLoader();
                const meshUrl = `/api/mesh/${meshId}/file`;
                
                updateStatus('Loading 3D mesh...');
                loader.load(
                    meshUrl,
                    (geometry) => {
                        // Process geometry
                        geometry.computeVertexNormals();
                        
                        // Create mesh material
                        const material = new THREE.MeshPhongMaterial({
                            vertexColors: geometry.attributes.color ? true : false,
                            color: geometry.attributes.color ? 0xffffff : 0x8888ff,
                            specular: 0x111111,
                            shininess: 10,
                            opacity: 0.9,
                            transparent: true,
                            side: THREE.DoubleSide
                        });
                        
                        // Remove previous mesh if exists
                        if (window.loadedMesh) {
                            scene.remove(window.loadedMesh);
                            if (window.loadedMesh.geometry) window.loadedMesh.geometry.dispose();
                            if (window.loadedMesh.material) window.loadedMesh.material.dispose();
                        }
                        
                        // Create mesh
                        meshObject = new THREE.Mesh(geometry, material);
                        meshObject.castShadow = true;
                        meshObject.receiveShadow = true;
                        scene.add(meshObject);
                        
                        // Store mesh reference for visibility toggle
                        window.loadedMesh = meshObject;
                        
                        // Center camera on mesh and calculate appropriate sphere size
                        const box = new THREE.Box3().setFromObject(meshObject);
                        const center = box.getCenter(new THREE.Vector3());
                        const size = box.getSize(new THREE.Vector3());
                        
                        // Calculate appropriate sphere size based on mesh extent
                        const meshDiameter = Math.max(size.x, size.y, size.z);
                        const autoSphereSize = meshDiameter * 0.005; // 0.5% of mesh size
                        
                        // Update sphere size slider with calculated value
                        const sphereSizeSlider = document.getElementById('sphereSizeSlider');
                        sphereSizeSlider.min = autoSphereSize * 0.1;
                        sphereSizeSlider.max = autoSphereSize * 10;
                        sphereSizeSlider.value = autoSphereSize;
                        document.getElementById('sphereSizeValue').textContent = autoSphereSize.toFixed(4);
                        
                        camera.position.set(
                            center.x + size.x * 0.5,
                            center.y + size.y * 0.5,
                            center.z + size.z * 0.5
                        );
                        controls.target.copy(center);
                        controls.update();
                        
                        updateStatus(`Mesh loaded: ${data.name} (sphere size: ${autoSphereSize.toFixed(4)})`);
                        
                        // Initialize track spheres with calculated size
                        initializeTrackSpheres(data.num_tracks, autoSphereSize);
                        
                        // Store mesh info for other functions
                        window.meshCenter = center;
                        window.meshSize = size;
                        
                        // Show camera frustum if available
                        console.log('Camera params from server:', data.camera_params);
                        if (data.camera_params) {
                            addCameraFrustum(data.camera_params);
                        } else {
                            console.log('No camera parameters available - camera frustum disabled');
                        }
                    },
                    (progress) => {
                        const percent = (progress.loaded / progress.total * 100).toFixed(0);
                        updateStatus(`Loading mesh: ${percent}%`);
                    },
                    (error) => {
                        console.error('Error loading mesh:', error);
                        updateStatus('Error loading mesh', true);
                    }
                );
                
                currentMeshId = meshId;
                
            } catch (error) {
                console.error('Error:', error);
                updateStatus(error.message, true);
            }
        }
        
        // Initialize spheres for tracks
        function initializeTrackSpheres(numTracks, sphereSize = null) {
            if (sphereSize === null) {
                sphereSize = parseFloat(document.getElementById('sphereSizeSlider').value);
            }
            const sphereGeometry = new THREE.SphereGeometry(sphereSize, 16, 12);
            
            // Clear existing spheres
            Object.values(spheres).forEach(sphere => {
                if (sphere) {
                    scene.remove(sphere);
                    sphere.geometry.dispose();
                    sphere.material.dispose();
                }
            });
            spheres = {};
            
            // Extract all unique track IDs from the track data
            const uniqueTrackIds = new Set();
            Object.values(trackData || {}).forEach(frameData => {
                frameData.forEach(track => {
                    uniqueTrackIds.add(track.track_id);
                });
            });
            
            console.log('Unique track IDs found:', Array.from(uniqueTrackIds).sort());
            
            // Create spheres for each actual track ID
            let colorIndex = 0;
            uniqueTrackIds.forEach(trackId => {
                const color = trackColors[trackId] || colorPalette[colorIndex % colorPalette.length];
                trackColors[trackId] = color;
                colorIndex++;
                
                const material = new THREE.MeshPhongMaterial({
                    color: color,
                    emissive: color,
                    emissiveIntensity: 0.3,
                    shininess: 100
                });
                
                const sphere = new THREE.Mesh(sphereGeometry, material);
                sphere.castShadow = true;
                sphere.visible = false;
                spheres[trackId] = sphere;  // Use track ID as key
                scene.add(sphere);
                
                // Create label for this track
                const labelDiv = document.createElement('div');
                labelDiv.className = 'label';
                labelDiv.textContent = trackId.toString();
                labelDiv.style.marginTop = '-1em';
                labelDiv.style.color = `#${color.toString(16).padStart(6, '0')}`;
                labelDiv.style.fontFamily = 'Arial, sans-serif';
                labelDiv.style.fontSize = '12px';
                labelDiv.style.fontWeight = 'bold';
                labelDiv.style.textShadow = '1px 1px 2px rgba(0,0,0,0.8)';
                labelDiv.style.pointerEvents = 'none';
                
                const label = new CSS2DObject(labelDiv);
                label.position.set(0, 0, 0);
                label.visible = document.getElementById('showIds').checked;
                labels[trackId] = label;
                scene.add(label);
            });
            
            console.log(`Created ${uniqueTrackIds.size} spheres for track IDs: ${Array.from(uniqueTrackIds).sort()}`);
        }
        
        // Update sphere positions for current frame
        function updateFrame(frame) {
            if (!trackData) return;
            
            currentFrame = frame;
            const frameData = trackData[frame] || [];
            
            // Hide all spheres and labels first
            Object.values(spheres).forEach(sphere => {
                if (sphere) sphere.visible = false;
            });
            Object.values(labels).forEach(label => {
                if (label) label.visible = false;
            });
            
            let visibleTracks = 0;
            // Update positions for visible tracks
            frameData.forEach(track => {
                if (track.x !== null && track.y !== null && track.z !== null) {
                    const sphere = spheres[track.track_id];
                    if (sphere) {
                        sphere.position.set(track.x, track.y, track.z);
                        sphere.visible = true;
                        visibleTracks++;
                        
                        // Update label position to be slightly above the sphere
                        const label = labels[track.track_id];
                        if (label) {
                            const sphereSize = parseFloat(document.getElementById('sphereSizeSlider').value);
                            label.position.set(track.x, track.y + sphereSize * 2, track.z);
                            label.visible = document.getElementById('showIds').checked;
                        }
                        
                        // Update trail for this track
                        updateTrail(track.track_id, track.x, track.y, track.z, frame);
                    } else {
                        console.warn(`No sphere found for track_id: ${track.track_id}`);
                    }
                }
            });
            
            // Hide trails for invisible tracks and clean up old trail data
            const visibleTrackIds = new Set(frameData.filter(t => t.x !== null && t.y !== null && t.z !== null).map(t => t.track_id));
            
            // Hide trails for tracks that are not visible in this frame
            Object.keys(trails).forEach(trackId => {
                if (!visibleTrackIds.has(parseInt(trackId))) {
                    if (trails[trackId]) {
                        trails[trackId].visible = false;
                    }
                }
            });
            
            // Clean up trail positions for tracks that haven't been seen for a while
            if (window.trailPositions) {
                const maxInactiveFrames = 60; // Clean up after 60 frames of inactivity
                Object.keys(window.trailPositions).forEach(trackId => {
                    const positions = window.trailPositions[trackId];
                    if (positions.length > 0) {
                        const lastFrame = positions[positions.length - 1].frame;
                        if (frame - lastFrame > maxInactiveFrames) {
                            // Clean up old trail
                            if (trails[trackId]) {
                                scene.remove(trails[trackId]);
                                trails[trackId].geometry.dispose();
                                trails[trackId].material.dispose();
                                delete trails[trackId];
                            }
                            delete window.trailPositions[trackId];
                        }
                    }
                });
            }
            
            // Debug: Log track updates occasionally
            if (frame % 30 === 0 || visibleTracks === 0) {
                console.log(`Frame ${frame}: ${frameData.length} track entries, ${visibleTracks} visible spheres`);
                if (frameData.length > 0) {
                    const sampleTrack = frameData[0];
                    console.log('Sample track:', sampleTrack);
                }
            }
            
            // Update UI
            document.getElementById('frameValue').textContent = frame;
            document.getElementById('frameSlider').value = frame;
            document.getElementById('frame-info').textContent = `Frame: ${frame} / ${maxFrame} | Tracks: ${frameData.length} | Visible: ${visibleTracks}`;
        }
        
        // Update trail for a track
        function updateTrail(trackId, x, y, z, frame) {
            const maxTrailLength = 100; // Keep last 100 positions for longer trails
            
            // Initialize trail data if needed
            if (!window.trailPositions) {
                window.trailPositions = {};
            }
            if (!window.trailPositions[trackId]) {
                window.trailPositions[trackId] = [];
            }
            
            // Add new position
            window.trailPositions[trackId].push({x, y, z, frame});
            
            // Keep only recent positions
            if (window.trailPositions[trackId].length > maxTrailLength) {
                window.trailPositions[trackId].shift();
            }
            
            // Update or create trail geometry
            if (window.trailPositions[trackId].length > 1) {
                const positions = [];
                window.trailPositions[trackId].forEach(pos => {
                    positions.push(pos.x, pos.y, pos.z);
                });
                
                // Remove old trail
                if (trails[trackId]) {
                    scene.remove(trails[trackId]);
                    trails[trackId].geometry.dispose();
                    trails[trackId].material.dispose();
                }
                
                // Get current sphere size for proportional trail width (1/3 of sphere radius)
                const currentSphereSize = parseFloat(document.getElementById('sphereSizeSlider').value);
                const trailRadius = Math.max(0.001, currentSphereSize / 3); // 1/3 of sphere radius
                
                // Use TubeGeometry instead of Line for visible width
                const curve = new THREE.CatmullRomCurve3(
                    window.trailPositions[trackId].map(pos => new THREE.Vector3(pos.x, pos.y, pos.z))
                );
                const tubeGeometry = new THREE.TubeGeometry(curve, 32, trailRadius, 8, false);
                const material = new THREE.MeshBasicMaterial({
                    color: trackColors[trackId] || 0xffffff,
                    opacity: 0.6,
                    transparent: true
                });
                
                const trail = new THREE.Mesh(tubeGeometry, material);
                trail.visible = document.getElementById('showTrails').checked;
                trails[trackId] = trail;
                scene.add(trail);
            }
        }
        
        // Animation loop
        function animate(time) {
            requestAnimationFrame(animate);
            
            // Update controls
            controls.update();
            
            // Handle playback
            if (isPlaying && trackData) {
                const deltaTime = time - lastAnimationTime;
                if (deltaTime > 33 / animationSpeed) {  // ~30fps adjusted by speed
                    currentFrame++;
                    if (currentFrame > maxFrame) {
                        currentFrame = 0;
                    }
                    updateFrame(currentFrame);
                    lastAnimationTime = time;
                }
            }
            
            // Render
            renderer.render(scene, camera);
            labelRenderer.render(scene, camera);
        }
        
        // Clear scene
        function clearScene() {
            if (meshObject) {
                scene.remove(meshObject);
                meshObject.geometry.dispose();
                meshObject.material.dispose();
                meshObject = null;
            }
            
            Object.values(spheres).forEach(sphere => {
                if (sphere) {
                    scene.remove(sphere);
                    sphere.geometry.dispose();
                    sphere.material.dispose();
                }
            });
            spheres = {};
            
            Object.values(trails).forEach(trail => {
                if (trail) {
                    scene.remove(trail);
                    trail.geometry.dispose();
                    trail.material.dispose();
                }
            });
            trails = {};
            
            Object.values(labels).forEach(label => {
                if (label) {
                    scene.remove(label);
                }
            });
            labels = {};
        }
        
        // Add camera frustum visualization
        function addCameraFrustum(cameraParams) {
            try {
                console.log('Adding camera frustum with params:', cameraParams);
                console.log('Camera params keys:', Object.keys(cameraParams || {}));
                
                if (!cameraParams) {
                    console.warn('No camera parameters provided');
                    return;
                }
                
                // Check for different possible parameter structures
                let K, c2w, width, height;
                
                if (cameraParams.K && cameraParams.c2w) {
                    // Standard format
                    K = cameraParams.K;
                    c2w = cameraParams.c2w;
                    width = cameraParams.width || 1920;
                    height = cameraParams.height || 1080;
                } else if (cameraParams.intrinsic && cameraParams.extrinsic) {
                    // Alternative format
                    K = cameraParams.intrinsic;
                    c2w = cameraParams.extrinsic;
                    width = cameraParams.width || 1920;
                    height = cameraParams.height || 1080;
                } else {
                    console.warn('Missing camera parameters. Available keys:', Object.keys(cameraParams));
                    console.warn('Expected: K + c2w OR intrinsic + extrinsic');
                    return;
                }
                
                console.log('Using K matrix:', K);
                console.log('Using c2w matrix:', c2w);
                
                // Create camera position from c2w matrix
                const cameraPosition = new THREE.Vector3(c2w[0][3], c2w[1][3], c2w[2][3]);
                
                // Calculate field of view from intrinsic matrix
                const fx = K[0][0];
                const fy = K[1][1];
                const fov = 2 * Math.atan(height / (2 * fy)) * 180 / Math.PI;
                const aspect = width / height;
                
                // Create frustum geometry
                const near = 0.01;
                const far = window.meshSize ? Math.max(window.meshSize.x, window.meshSize.y, window.meshSize.z) * 0.5 : 1.0;
                
                // Create frustum manually using line geometry for better control
                const frustumGeometry = new THREE.BufferGeometry();
                
                // Calculate frustum corners in camera space (z points forward)
                const nearHalfHeight = near * Math.tan(fov * Math.PI / 360);
                const nearHalfWidth = nearHalfHeight * aspect;
                const farHalfHeight = far * Math.tan(fov * Math.PI / 360);
                const farHalfWidth = farHalfHeight * aspect;
                
                // Frustum vertices in camera coordinate system (z forward, y up, x right)
                const vertices = [
                    // Camera center
                    0, 0, 0,
                    // Near plane corners
                    -nearHalfWidth, -nearHalfHeight, near,  // bottom-left
                    nearHalfWidth, -nearHalfHeight, near,   // bottom-right  
                    nearHalfWidth, nearHalfHeight, near,    // top-right
                    -nearHalfWidth, nearHalfHeight, near,   // top-left
                    // Far plane corners  
                    -farHalfWidth, -farHalfHeight, far,     // bottom-left
                    farHalfWidth, -farHalfHeight, far,      // bottom-right
                    farHalfWidth, farHalfHeight, far,       // top-right
                    -farHalfWidth, farHalfHeight, far,      // top-left
                ];
                
                // Line indices to connect the frustum
                const indices = [
                    // Lines from camera center to near plane
                    0, 1,  0, 2,  0, 3,  0, 4,
                    // Lines from camera center to far plane
                    0, 5,  0, 6,  0, 7,  0, 8,
                    // Near plane rectangle
                    1, 2,  2, 3,  3, 4,  4, 1,
                    // Far plane rectangle
                    5, 6,  6, 7,  7, 8,  8, 5,
                    // Connect near to far
                    1, 5,  2, 6,  3, 7,  4, 8
                ];
                
                frustumGeometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
                frustumGeometry.setIndex(indices);
                
                const frustumMaterial = new THREE.LineBasicMaterial({
                    color: 0xff0000,
                    linewidth: 2
                });
                
                const frustumLines = new THREE.LineSegments(frustumGeometry, frustumMaterial);
                
                // Apply camera transformation to frustum
                // Convert c2w matrix to Three.js format and apply to frustum
                const transformMatrix = new THREE.Matrix4().set(
                    c2w[0][0], c2w[0][1], c2w[0][2], c2w[0][3],
                    c2w[1][0], c2w[1][1], c2w[1][2], c2w[1][3],
                    c2w[2][0], c2w[2][1], c2w[2][2], c2w[2][3],
                    0, 0, 0, 1
                );
                
                frustumLines.applyMatrix4(transformMatrix);
                cameraFrustum = frustumLines;
                scene.add(cameraFrustum);
                
                // Store frustum reference globally for visibility toggle
                window.cameraFrustum = cameraFrustum;
                
                // Add camera position marker
                const cameraMarkerGeometry = new THREE.SphereGeometry(far * 0.02, 16, 12);
                const cameraMarkerMaterial = new THREE.MeshPhongMaterial({
                    color: 0xff0000,
                    emissive: 0x440000,
                    transparent: true,
                    opacity: 0.8
                });
                const cameraMarker = new THREE.Mesh(cameraMarkerGeometry, cameraMarkerMaterial);
                cameraMarker.position.copy(cameraPosition);
                scene.add(cameraMarker);
                
                // Store marker for visibility toggle
                window.cameraMarker = cameraMarker;
                
                // Show camera info in console
                console.log('Camera frustum added:', {
                    position: cameraPosition,
                    fov: fov.toFixed(1),
                    aspect: aspect.toFixed(2),
                    near, far: far.toFixed(2)
                });
                
                // Enable camera visibility checkbox
                document.getElementById('showCamera').disabled = false;
                document.getElementById('showCamera').checked = true;
                
            } catch (error) {
                console.error('Error adding camera frustum:', error);
            }
        }
        
        // UI Functions
        window.togglePlayback = function() {
            isPlaying = !isPlaying;
            document.getElementById('playBtn').textContent = isPlaying ? '⏸️ Pause' : '▶️ Play';
        };
        
        window.resetAnimation = function() {
            currentFrame = 0;
            
            // Clear all trails
            Object.values(trails).forEach(trail => {
                if (trail) {
                    scene.remove(trail);
                    trail.geometry.dispose();
                    trail.material.dispose();
                }
            });
            trails = {};
            window.trailPositions = {};
            
            updateFrame(0);
            isPlaying = false;
            document.getElementById('playBtn').textContent = '▶️ Play';
        };
        
        window.updateMeshVisibility = function() {
            if (meshObject) {
                meshObject.visible = document.getElementById('showMesh').checked;
            }
        };
        
        window.updateLabelVisibility = function() {
            const showLabels = document.getElementById('showIds').checked;
            console.log('Label visibility:', showLabels);
            
            // Toggle visibility of all track ID labels
            Object.values(labels).forEach(label => {
                if (label) {
                    label.visible = showLabels;
                }
            });
        };
        
        window.updateTrailVisibility = function() {
            const showTrails = document.getElementById('showTrails').checked;
            console.log('Trail visibility:', showTrails);
            
            // Toggle visibility of all trail lines
            Object.values(trails).forEach(trail => {
                if (trail) {
                    trail.visible = showTrails;
                }
            });
        };
        
        window.updateCameraVisibility = function() {
            const isVisible = document.getElementById('showCamera').checked;
            if (cameraFrustum) {
                cameraFrustum.visible = isVisible;
            }
            if (window.cameraMarker) {
                window.cameraMarker.visible = isVisible;
            }
        };
        
        window.refreshMeshList = async function() {
            try {
                const response = await fetch('/api/meshes');
                const meshes = await response.json();
                
                const select = document.getElementById('meshSelect');
                select.innerHTML = '<option value="">Choose a mesh...</option>';
                
                meshes.forEach(mesh => {
                    const option = document.createElement('option');
                    option.value = mesh.id;
                    option.textContent = `${mesh.name} (${mesh.tracks} tracks, ${mesh.frames} frames)`;
                    select.appendChild(option);
                });
                
                updateStatus(`Found ${meshes.length} mesh(es)`);
                
            } catch (error) {
                console.error('Error loading meshes:', error);
                updateStatus('Error loading mesh list', true);
            }
        };
        
        function updateStatus(message, isError = false) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = isError ? 'error' : '';
        }
        
        // Event listeners
        document.getElementById('meshSelect').addEventListener('change', (e) => {
            const meshId = e.target.value;
            if (meshId) {
                loadMesh(meshId);
            }
        });
        
        document.getElementById('speedSlider').addEventListener('input', (e) => {
            animationSpeed = parseFloat(e.target.value);
            document.getElementById('speedValue').textContent = animationSpeed.toFixed(1) + 'x';
        });
        
        document.getElementById('frameSlider').addEventListener('input', (e) => {
            if (!isPlaying) {
                updateFrame(parseInt(e.target.value));
            }
        });
        
        document.getElementById('sphereSizeSlider').addEventListener('input', (e) => {
            const size = parseFloat(e.target.value);
            document.getElementById('sphereSizeValue').textContent = size.toFixed(4);
            
            // Update all sphere geometries with new size
            Object.values(spheres).forEach(sphere => {
                if (sphere) {
                    const newGeometry = new THREE.SphereGeometry(size, 16, 12);
                    sphere.geometry.dispose(); // Clean up old geometry
                    sphere.geometry = newGeometry;
                }
            });
            
            console.log(`Updated ${Object.keys(spheres).length} spheres to size ${size.toFixed(4)}`);
        });
        
        // Initialize
        // Visibility control functions for UI toggles
        window.updateTrailVisibility = function() {
            const showTrails = document.getElementById('showTrails').checked;
            Object.values(trails).forEach(trail => {
                if (trail && trail.visible !== undefined) {
                    // Only show trails if the checkbox is checked AND the track is currently active
                    // This prevents showing all trails when toggling
                    trail.visible = showTrails && trail.parent; // Only visible trails in scene
                }
            });
        };
        
        window.updateLabelVisibility = function() {
            const showIds = document.getElementById('showIds').checked;
            Object.values(labels).forEach(label => {
                if (label && label.visible !== undefined) {
                    // Only show labels if checkbox is checked AND the track is currently active
                    // The updateFrame function will properly manage which labels should be visible
                    label.visible = showIds && label.parent; // Only visible labels in scene
                }
            });
            // Force update current frame to apply proper visibility
            if (typeof currentFrame !== 'undefined') {
                updateFrame(currentFrame);
            }
        };
        
        window.updateMeshVisibility = function() {
            const showMesh = document.getElementById('showMesh').checked;
            if (window.loadedMesh) {
                window.loadedMesh.visible = showMesh;
            }
        };
        
        window.updateCameraVisibility = function() {
            const showCamera = document.getElementById('showCamera').checked;
            if (window.cameraFrustum) {
                window.cameraFrustum.visible = showCamera;
            }
            if (window.cameraMarker) {
                window.cameraMarker.visible = showCamera;
            }
        };

        initScene();
        animate(0);
        refreshMeshList();
        
        console.log('✅ 3D Track Viewer ready');
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    """Main page with video selector and viewer."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/video/<video_id>")
def get_video_data(video_id):
    """Get video metadata and bbox data."""
    with data_lock:
        if video_id not in videos_data:
            return jsonify({"error": "Video not found"}), 404

        video_info = videos_data[video_id]

        # Detect data format
        if (
            video_info["bbox_data"]
            and "x1" in list(video_info["bbox_data"].values())[0][0]
        ):
            data_format = "Bounding Boxes (x1, y1, x2, y2)"
        else:
            data_format = "Centroids (x, y)"

        # Get track info
        track_ids = set()
        for frame_data in video_info["bbox_data"].values():
            for annotation in frame_data:
                track_ids.add(annotation["track_id"])

        return jsonify(
            {
                "name": video_info["name"],
                "data_format": data_format,
                "num_tracks": len(track_ids),
                "num_frames": len(video_info["bbox_data"]),
                "fps": video_info["fps"],
                "bbox_data": video_info["bbox_data"],
            }
        )


@app.route("/api/video/<video_id>/stream")
def stream_video(video_id):
    """Stream video file."""
    with data_lock:
        if video_id not in videos_data:
            return "Video not found", 404

        video_path = videos_data[video_id]["video_path"]
        return send_file(video_path, mimetype="video/mp4")


@app.route("/api/add_video", methods=["POST"])
def api_add_video():
    """API endpoint to add a video/CSV combination."""
    try:
        from flask import request
        import traceback

        print("🔄 API add_video called")

        # Get JSON data
        data = request.get_json()
        if not data:
            error_msg = "No JSON data received"
            print(f"❌ {error_msg}")
            return jsonify({"error": error_msg}), 400

        print(f"📝 Received data: {data}")

        video_id = data.get("video_id")
        video_path = data.get("video_path")
        csv_path = data.get("csv_path")
        fps = data.get("fps", 30.0)
        video_label = data.get("video_label")  # Meaningful video name/label

        print(f"📹 Video ID: {video_id}")
        print(f"🎥 Video path: {video_path}")
        print(f"📊 CSV path: {csv_path}")
        print(f"⏱️ FPS: {fps}")

        if not all([video_id, video_path, csv_path]):
            error_msg = f"Missing fields - video_id: {bool(video_id)}, video_path: {bool(video_path)}, csv_path: {bool(csv_path)}"
            print(f"❌ {error_msg}")
            return jsonify({"error": error_msg}), 400

        # Check if files exist
        if not Path(video_path).exists():
            error_msg = f"Video file not found: {video_path}"
            print(f"❌ {error_msg}")
            return jsonify({"error": error_msg}), 400

        if not Path(csv_path).exists():
            error_msg = f"CSV file not found: {csv_path}"
            print(f"❌ {error_msg}")
            return jsonify({"error": error_msg}), 400

        # Add video using internal function
        print("🚀 Adding video to server...")
        success = add_video(video_id, video_path, csv_path, fps, video_label)

        if success:
            print(f"✅ Video added successfully: {video_id}")
            return jsonify({"success": True, "video_id": video_id})
        else:
            error_msg = "Failed to add video (unknown error)"
            print(f"❌ {error_msg}")
            return jsonify({"error": error_msg}), 500

    except Exception as e:
        error_msg = f"Exception in add_video: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"📜 Traceback: {traceback.format_exc()}")
        return jsonify({"error": error_msg}), 500


@app.route("/api/videos")
def list_videos_api():
    """API endpoint to list all videos."""
    with data_lock:
        video_list = []
        for video_id, info in videos_data.items():
            video_list.append(
                {
                    "id": video_id,
                    "name": info["name"],
                    "tracks": len(
                        set(
                            ann["track_id"]
                            for frame_data in info["bbox_data"].values()
                            for ann in frame_data
                        )
                    ),
                }
            )
        print(f"📝 API /api/videos returning {len(video_list)} videos")
        return jsonify(video_list)


@app.route("/api/health")
def health_check():
    """Simple health check endpoint."""
    with data_lock:
        return jsonify(
            {
                "status": "ok",
                "videos_count": len(videos_data),
                "meshes_count": len(meshes_data),
                "videos": list(videos_data.keys()),
                "meshes": list(meshes_data.keys()),
            }
        )


@app.route("/api/clear", methods=["POST"])
def clear_all_data():
    """Clear all video and mesh data from the server."""
    with data_lock:
        old_video_count = len(videos_data)
        old_mesh_count = len(meshes_data)
        
        videos_data.clear()
        meshes_data.clear()
        
        total_cleared = old_video_count + old_mesh_count
        print(f"🧽 Cleared {old_video_count} videos and {old_mesh_count} meshes from server")
        
        return jsonify({
            "success": True, 
            "cleared_videos": old_video_count,
            "cleared_meshes": old_mesh_count,
            "total_cleared": total_cleared
        })


@app.route("/api/remove_video/<video_id>", methods=["DELETE"])
def remove_video_endpoint(video_id):
    """Remove a specific video from the server."""
    with data_lock:
        if video_id in videos_data:
            del videos_data[video_id]
            print(f"🗑️ Removed video: {video_id}")
            return jsonify({"success": True, "video_id": video_id})
        else:
            return jsonify({"error": "Video not found"}), 404


def prepare_bbox_data(df: pd.DataFrame) -> dict:
    """Convert DataFrame to frame-indexed dictionary."""
    data_by_frame: dict[int, list] = {}

    # Detect format
    if "x1" in df.columns:
        data_format = "bbox"
    else:
        data_format = "centroid"

    for _, row in df.iterrows():
        frame = int(row["frame"])
        track_id = row["track_id"]

        if frame not in data_by_frame:
            data_by_frame[frame] = []

        if data_format == "bbox":
            annotation = {
                "track_id": track_id,
                "x1": float(row["x1"]),
                "y1": float(row["y1"]),
                "x2": float(row["x2"]),
                "y2": float(row["y2"]),
                "type": "bbox",
            }
        else:
            annotation = {
                "track_id": track_id,
                "x": float(row["x"]),
                "y": float(row["y"]),
                "type": "centroid",
            }

        data_by_frame[frame].append(annotation)

    return data_by_frame


def add_video(
    video_id: str,
    video_path: str,
    csv_path: str,
    fps: float = 30.0,
    video_label: str | None = None,
):
    """Add a video/CSV combination to the server."""
    try:
        # Load CSV data
        df = pd.read_csv(csv_path)
        bbox_data = prepare_bbox_data(df)

        # Use meaningful label if provided, otherwise fall back to cache filename
        display_name = video_label if video_label else Path(video_path).name

        with data_lock:
            videos_data[video_id] = {
                "name": display_name,
                "video_path": video_path,
                "csv_path": csv_path,
                "fps": fps,
                "bbox_data": bbox_data,
            }

        print(f"✅ Added video: {video_id} ({display_name})")
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


# ============ 3D Mesh/Track Support ============

@app.route("/3d")
def mesh_viewer():
    """Main page for 3D mesh viewer."""
    return render_template_string(MESH_3D_TEMPLATE)


@app.route("/api/add_mesh", methods=["POST"])
def api_add_mesh():
    """Add a mesh/3D CSV combination."""
    try:
        from flask import request
        data = request.get_json()
        
        mesh_id = data.get("mesh_id")
        mesh_path = data.get("mesh_path")
        csv_3d_path = data.get("csv_3d_path")
        camera_params_path = data.get("camera_params_path")  # Optional
        mesh_label = data.get("mesh_label")
        
        if not all([mesh_id, mesh_path, csv_3d_path]):
            return jsonify({"error": "Missing required fields"}), 400
            
        # Check files exist
        if not Path(mesh_path).exists():
            return jsonify({"error": f"Mesh file not found: {mesh_path}"}), 400
        if not Path(csv_3d_path).exists():
            return jsonify({"error": f"3D CSV file not found: {csv_3d_path}"}), 400
            
        # Load 3D track data
        df = pd.read_csv(csv_3d_path)
        
        # Prepare track data by frame
        tracks_by_frame = {}
        for _, row in df.iterrows():
            frame = int(row["frame"])
            if frame not in tracks_by_frame:
                tracks_by_frame[frame] = []
            
            # Store 3D position for each track
            tracks_by_frame[frame].append({
                "track_id": int(row["track_id"]),
                "x": float(row["x"]) if pd.notna(row["x"]) else None,
                "y": float(row["y"]) if pd.notna(row["y"]) else None,
                "z": float(row["z"]) if pd.notna(row["z"]) else None,
            })
        
        # Load camera parameters if provided
        camera_params = None
        print(f"🔍 Camera params path provided: {camera_params_path}")
        
        if camera_params_path and Path(camera_params_path).exists():
            import pickle
            try:
                print(f"📷 Loading camera params from: {camera_params_path}")
                
                # Check if file is empty (indicates cache issue)
                file_size = Path(camera_params_path).stat().st_size
                print(f"📏 Camera params file size: {file_size} bytes")
                
                if file_size == 0:
                    print("⚠️ Camera params file is empty (cache issue) - skipping")
                    camera_params = None
                else:
                    with open(camera_params_path, "rb") as f:
                        raw_camera_params = pickle.load(f)
                        
                    print(f"✅ Loaded camera params. Type: {type(raw_camera_params)}")
                    print(f"📋 Raw camera params keys: {list(raw_camera_params.keys()) if isinstance(raw_camera_params, dict) else 'Not a dict'}")
                    print(f"📏 Raw camera params size: {len(str(raw_camera_params))} characters")
                    if isinstance(raw_camera_params, dict):
                        for key, value in raw_camera_params.items():
                            print(f"  - {key}: {type(value)} {getattr(value, 'shape', '') if hasattr(value, 'shape') else ''}")
                    
                    # Convert to JSON-serializable format
                    camera_params = convert_camera_params_to_json(raw_camera_params)
                    print(f"📊 Converted camera params. Keys: {list(camera_params.keys()) if isinstance(camera_params, dict) else 'Not a dict'}")
                    
                    if camera_params is None:
                        print("⚠️ Camera params conversion returned None")
                    else:
                        print(f"✅ Camera params ready for JSON: {len(str(camera_params))} characters")
                    print(f"📊 Sample camera params structure: {list(camera_params.keys()) if isinstance(camera_params, dict) else 'Not a dict'}")
                    
                    # Extra verification that data is valid
                    if isinstance(camera_params, dict):
                        for key, value in list(camera_params.items())[:3]:  # Show first 3 items
                            print(f"  📋 {key}: {type(value)} = {value}")
                    else:
                        print(f"⚠️ Unexpected camera_params type: {type(camera_params)}")
                
            except Exception as e:
                print(f"❌ Error loading camera params: {e}")
                import traceback
                traceback.print_exc()
                camera_params = None
        elif camera_params_path:
            print(f"❌ Camera params file not found: {camera_params_path}")
        else:
            print("ℹ️ No camera params path provided")
        
        with data_lock:
            meshes_data[mesh_id] = {
                "name": mesh_label if mesh_label else Path(mesh_path).name,
                "mesh_path": mesh_path,
                "csv_3d_path": csv_3d_path,
                "tracks_by_frame": tracks_by_frame,
                "camera_params": camera_params,
                "num_frames": len(tracks_by_frame),
                "num_tracks": len(df["track_id"].unique()) if "track_id" in df.columns else 0,
            }
        
        print(f"✅ Added mesh: {mesh_id}")
        return jsonify({"success": True, "mesh_id": mesh_id})
        
    except Exception as e:
        print(f"❌ Error adding mesh: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/meshes")
def list_meshes_api():
    """List all loaded meshes."""
    with data_lock:
        mesh_list = []
        for mesh_id, info in meshes_data.items():
            mesh_list.append({
                "id": mesh_id,
                "name": info["name"],
                "tracks": info["num_tracks"],
                "frames": info["num_frames"],
            })
        return jsonify(mesh_list)


@app.route("/api/mesh/<mesh_id>")
def get_mesh_data(mesh_id):
    """Get mesh metadata and track data."""
    with data_lock:
        if mesh_id not in meshes_data:
            return jsonify({"error": "Mesh not found"}), 404
        
        mesh_info = meshes_data[mesh_id]
        return jsonify({
            "name": mesh_info["name"],
            "num_tracks": mesh_info["num_tracks"],
            "num_frames": mesh_info["num_frames"],
            "tracks_by_frame": mesh_info["tracks_by_frame"],
            "camera_params": mesh_info["camera_params"],
        })


@app.route("/api/mesh/<mesh_id>/file")
def stream_mesh_file(mesh_id):
    """Stream the mesh PLY file."""
    with data_lock:
        if mesh_id not in meshes_data:
            return "Mesh not found", 404
        
        mesh_path = meshes_data[mesh_id]["mesh_path"]
        return send_file(mesh_path, mimetype="application/octet-stream")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Persistent Video Bbox Overlay Server")
    parser.add_argument("--port", type=int, default=5050, help="Port to run on")
    parser.add_argument("--host", default="localhost", help="Host to run on")

    args = parser.parse_args()

    print(f"🚀 Starting persistent video server at http://{args.host}:{args.port}")
    print("📝 API endpoints available:")
    print("  - GET  / (homepage)")
    print("  - GET  /3d (3D mesh viewer)")
    print("  - POST /api/add_video (add video/CSV)")
    print("  - POST /api/add_mesh (add mesh/3D CSV)")
    print("  - GET  /api/videos (list videos)")
    print("  - GET  /api/meshes (list meshes)")
    print("  - GET  /api/health (health check)")
    print("  - GET  /api/video/<id> (video data)")
    print("  - GET  /api/video/<id>/stream (video stream)")
    print("  - GET  /api/mesh/<id> (mesh data)")
    print("  - GET  /api/mesh/<id>/file (mesh PLY file)")

    app.run(
        host=args.host, port=args.port, debug=True, threaded=True
    )  # Enable debug for better error messages
