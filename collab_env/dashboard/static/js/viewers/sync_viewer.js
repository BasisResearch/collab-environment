/**
 * Synchronized 2D Video + 3D Mesh Viewer
 * Coordinates both viewers with shared frame management
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

import { ApiClient } from '../utils/api_client.js';
// Removed FrameManager import - using natural video playback instead
import { TrackColors } from '../utils/track_colors.js';
import { TrackTransformer } from '../utils/track_transformer.js';
import { TransformUI } from '../utils/transform_ui.js';

export class SyncViewer {
    constructor() {
        console.log('🚀 Starting Synchronized 2D + 3D Viewer');
        
        // Core components
        this.api = new ApiClient();
        // Natural video playback - no FrameManager needed
        this.trackColors = new TrackColors();
        this.trackTransformer = new TrackTransformer();
        this.transformUI = null;
        
        // Video viewer components
        this.videoPlayer = document.getElementById('videoPlayer');
        this.overlayCanvas = document.getElementById('overlayCanvas');
        this.overlayCtx = this.overlayCanvas.getContext('2d');
        this.videoMetadata = null;
        this.bboxData = {};
        this.videoTrails = {};
        
        // 3D viewer components
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.labelRenderer = null;
        this.controls = null;
        this.meshObject = null;
        this.spheres = {};
        this.trails3D = {};
        this.labels = {};
        this.cameraFrustum = null;
        this.trailPositions = {};
        
        // DOM elements
        this.videoSelect = document.getElementById('videoSelect');
        this.meshSelect = document.getElementById('meshSelect');
        this.refreshBtn = document.getElementById('refreshBtn');
        this.syncContent = document.getElementById('syncContent');
        this.meshCanvas = document.getElementById('meshCanvas');
        this.statusMessage = document.getElementById('statusMessage');
        
        // Status elements
        this.videoStatus = document.getElementById('videoStatus');
        this.meshStatus = document.getElementById('meshStatus');
        
        // Control elements
        this.playPauseBtn = document.getElementById('playPauseBtn');
        this.stepBackBtn = document.getElementById('stepBackBtn');
        this.stepForwardBtn = document.getElementById('stepForwardBtn');
        this.resetBtn = document.getElementById('resetBtn');
        this.frameSlider = document.getElementById('frameSlider');
        this.frameDisplay = document.getElementById('frameDisplay');
        this.speedSlider = document.getElementById('speedSlider');
        this.speedDisplay = document.getElementById('speedDisplay');
        
        // Display options
        this.showIds2D = document.getElementById('showIds2D');
        this.showTrails2D = document.getElementById('showTrails2D');
        this.showDebug2D = document.getElementById('showDebug2D');
        this.opacitySlider = document.getElementById('opacitySlider');
        
        this.showIds3D = document.getElementById('showIds3D');
        this.showTrails3D = document.getElementById('showTrails3D');
        this.showMesh3D = document.getElementById('showMesh3D');
        this.showCamera3D = document.getElementById('showCamera3D');
        this.sphereSizeSlider = document.getElementById('sphereSizeSlider');
        this.sphereSizeValue = document.getElementById('sphereSizeValue');
        
        // State
        this.currentVideoId = null;
        this.currentMeshId = null;
        this.trackData = null;
        this.maxFrame = 0;
        
        // Initialize
        this.init();
    }
    
    async init() {
        this.init3DScene();
        this.setupEventListeners();
        this.initTransformUI();
        await this.refreshDataLists();
        this.animate(0);

        console.log('✅ Synchronized Viewer ready');
    }
    
    init3DScene() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x2a2a2a);
        this.scene.fog = new THREE.Fog(0x2a2a2a, 0.1, 100);
        
        // Camera - use proper aspect ratio from container with improved controls
        const containerRect = this.meshCanvas.getBoundingClientRect();
        const aspect = containerRect.width > 0 ? containerRect.width / containerRect.height : 1;
        this.camera = new THREE.PerspectiveCamera(60, aspect, 0.001, 1000);
        this.camera.position.set(0.5, 0.5, 0.5);
        this.camera.lookAt(0, 0, 0);
        
        // Renderer - use container dimensions
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(containerRect.width, containerRect.height);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.meshCanvas.appendChild(this.renderer.domElement);
        
        // CSS2D Renderer for labels
        this.labelRenderer = new CSS2DRenderer();
        this.labelRenderer.setSize(containerRect.width, containerRect.height);
        this.labelRenderer.domElement.style.position = 'absolute';
        this.labelRenderer.domElement.style.top = '0px';
        this.labelRenderer.domElement.style.pointerEvents = 'none';
        this.meshCanvas.appendChild(this.labelRenderer.domElement);
        
        // Controls
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.screenSpacePanning = false;
        this.controls.minDistance = 0.01;
        this.controls.maxDistance = 10;
        this.controls.zoomSpeed = 1.0;
        this.controls.rotateSpeed = 1.0;
        this.controls.panSpeed = 1.0;
        
        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(1, 1, 1);
        directionalLight.castShadow = true;
        this.scene.add(directionalLight);
        
        // Grid helper
        const gridHelper = new THREE.GridHelper(2, 20, 0x444444, 0x222222);
        gridHelper.rotation.x = Math.PI / 2;
        this.scene.add(gridHelper);
        
        // Axes helper
        const axesHelper = new THREE.AxesHelper(0.2);
        this.scene.add(axesHelper);
        
        // Handle resize
        this.resizeRenderers();
        window.addEventListener('resize', () => this.resizeRenderers());
        
        console.log('✅ 3D Scene initialized');
    }
    
    setupEventListeners() {
        // Data selection
        this.videoSelect.addEventListener('change', (e) => {
            if (e.target.value) {
                this.loadVideo(e.target.value);
            }
        });
        
        this.meshSelect.addEventListener('change', (e) => {
            if (e.target.value) {
                this.loadMesh(e.target.value);
            }
        });
        
        this.refreshBtn.addEventListener('click', () => {
            this.refreshDataLists();
        });
        
        // Playback controls
        this.playPauseBtn.addEventListener('click', () => {
            if (this.videoPlayer.paused) {
                this.videoPlayer.play();
            } else {
                this.videoPlayer.pause();
            }
            this.updatePlayPauseButton();
        });
        
        this.stepBackBtn.addEventListener('click', () => {
            const fps = 30;
            this.videoPlayer.currentTime = Math.max(0, this.videoPlayer.currentTime - 1/fps);
        });
        
        this.stepForwardBtn.addEventListener('click', () => {
            const fps = 30;
            if (this.videoMetadata) {
                this.videoPlayer.currentTime = Math.min(this.videoMetadata.duration, this.videoPlayer.currentTime + 1/fps);
            }
        });
        
        this.resetBtn.addEventListener('click', () => {
            this.videoPlayer.currentTime = 0;
            this.clearTrails();
        });
        
        // Frame slider - manual seeking
        this.frameSlider.addEventListener('input', (e) => {
            const frame = parseInt(e.target.value);
            const fps = 30;
            this.videoPlayer.currentTime = frame / fps;
        });
        
        // Speed control
        this.speedSlider.addEventListener('input', (e) => {
            const speed = parseFloat(e.target.value);
            this.speedDisplay.textContent = speed.toFixed(2) + 'x';
            this.videoPlayer.playbackRate = speed;
        });
        
        
        // 2D display options - will redraw on next timeupdate
        this.showIds2D.addEventListener('change', () => this.onVideoTimeUpdate());
        this.showTrails2D.addEventListener('change', () => this.onVideoTimeUpdate());
        this.showDebug2D.addEventListener('change', () => this.onVideoTimeUpdate());
        this.opacitySlider.addEventListener('input', () => this.onVideoTimeUpdate());
        
        // 3D display options
        this.showIds3D.addEventListener('change', () => this.update3DVisibility());
        this.showTrails3D.addEventListener('change', () => this.update3DVisibility());
        this.showMesh3D.addEventListener('change', () => this.update3DVisibility());
        this.showCamera3D.addEventListener('change', () => this.update3DVisibility());
        
        this.sphereSizeSlider.addEventListener('input', (e) => {
            const size = parseFloat(e.target.value);
            this.sphereSizeValue.textContent = size.toFixed(4);
            this.updateSphereSize(size);
        });

        // Video metadata loaded
        this.videoPlayer.addEventListener('loadedmetadata', () => {
            this.onVideoMetadataLoaded();
        });
        
        // Video time update - sync overlays and 3D to video
        this.videoPlayer.addEventListener('timeupdate', () => {
            this.onVideoTimeUpdate();
        });
    }

    initTransformUI() {
        // Initialize transform UI in the sync controls panel
        const transformContainer = document.createElement('div');
        transformContainer.id = 'transformControls3D';
        transformContainer.className = 'control-group';

        // Find the sync controls panel and append
        const syncControlsPanel = document.querySelector('.sync-controls-panel');
        if (syncControlsPanel) {
            syncControlsPanel.appendChild(transformContainer);

            this.transformUI = new TransformUI(this.trackTransformer, transformContainer);

            // Listen for transform changes and trigger 3D frame update
            transformContainer.addEventListener('transform-changed', () => {
                // Clear existing trails to force rebuild with new transformation
                this.rebuildAll3DTrails();
                // Get current video frame and update 3D
                const fps = 30;
                const currentFrame = Math.floor(this.videoPlayer.currentTime * fps);
                this.update3DFrame(currentFrame);
            });
        }
    }
    
    async refreshDataLists() {
        try {
            this.showStatus('Loading data lists...', 'info');
            
            const [videos, meshes] = await Promise.all([
                this.api.getVideos(),
                this.api.getMeshes()
            ]);
            
            // Populate video selector
            this.videoSelect.innerHTML = '<option value="">Select a video...</option>';
            videos.forEach(video => {
                const option = document.createElement('option');
                option.value = video.id;
                option.textContent = video.name + (video.has_bboxes ? ' (with tracks)' : '');
                this.videoSelect.appendChild(option);
            });
            
            // Populate mesh selector
            this.meshSelect.innerHTML = '<option value="">Select a mesh...</option>';
            meshes.forEach(mesh => {
                const option = document.createElement('option');
                option.value = mesh.id;
                option.textContent = `${mesh.name} (${mesh.tracks} tracks, ${mesh.frames} frames)`;
                this.meshSelect.appendChild(option);
            });
            
            this.showStatus(`Found ${videos.length} videos and ${meshes.length} meshes`, 'success');
        } catch (error) {
            this.showStatus('Error loading data lists: ' + error.message, 'error');
        }
    }
    
    async loadVideo(videoId) {
        try {
            this.videoStatus.textContent = 'Loading video...';
            
            const data = await this.api.getVideoData(videoId);
            this.currentVideoId = videoId;
            this.bboxData = data.bbox_data || {};
            
            // Reset trails
            this.videoTrails = {};
            
            // Generate colors for all track IDs
            const allTrackIds = new Set();
            Object.values(this.bboxData).forEach(frameTracks => {
                frameTracks.forEach(track => allTrackIds.add(track.track_id));
            });
            
            if (allTrackIds.size > 0) {
                this.trackColors.generateForTracks(Array.from(allTrackIds));
            }
            
            // Load video file
            this.videoPlayer.src = this.api.getVideoFileUrl(videoId);
            
            this.videoStatus.textContent = `Loaded: ${data.name}`;
            this.checkForSyncReady();
            
        } catch (error) {
            this.videoStatus.textContent = 'Error loading video';
            this.showStatus('Error loading video: ' + error.message, 'error');
        }
    }
    
    async loadMesh(meshId) {
        try {
            this.meshStatus.textContent = 'Loading mesh...';
            
            const data = await this.api.getMeshData(meshId);
            this.currentMeshId = meshId;
            this.trackData = data.tracks_by_frame;
            this.maxFrame = data.num_frames - 1;
            
            // Load PLY mesh
            const loader = new PLYLoader();
            const meshUrl = this.api.getMeshFileUrl(meshId);
            
            await new Promise((resolve, reject) => {
                loader.load(
                    meshUrl,
                    (geometry) => {
                        try {
                            this.processMeshGeometry(geometry, data);
                            resolve();
                        } catch (error) {
                            reject(error);
                        }
                    },
                    (progress) => {
                        const percent = (progress.loaded / progress.total * 100).toFixed(0);
                        this.meshStatus.textContent = `Loading mesh: ${percent}%`;
                    },
                    reject
                );
            });
            
            this.meshStatus.textContent = `Loaded: ${data.name}`;
            this.checkForSyncReady();
            
        } catch (error) {
            this.meshStatus.textContent = 'Error loading mesh';
            this.showStatus('Error loading mesh: ' + error.message, 'error');
        }
    }
    
    processMeshGeometry(geometry, data) {
        // Clear existing mesh
        if (this.meshObject) {
            this.scene.remove(this.meshObject);
            this.meshObject.geometry.dispose();
            this.meshObject.material.dispose();
        }
        
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
        
        // Create mesh
        this.meshObject = new THREE.Mesh(geometry, material);
        this.meshObject.castShadow = true;
        this.meshObject.receiveShadow = true;
        this.scene.add(this.meshObject);
        
        // Position camera
        const box = new THREE.Box3().setFromObject(this.meshObject);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        
        this.camera.position.set(
            center.x + size.x * 0.5,
            center.y + size.y * 0.5,
            center.z + size.z * 0.5
        );
        this.controls.target.copy(center);
        this.controls.update();
        
        // Calculate appropriate sphere size
        const meshDiameter = Math.max(size.x, size.y, size.z);
        const autoSphereSize = meshDiameter * 0.005;
        
        this.sphereSizeSlider.min = autoSphereSize * 0.1;
        this.sphereSizeSlider.max = autoSphereSize * 10;
        this.sphereSizeSlider.value = autoSphereSize;
        this.sphereSizeValue.textContent = autoSphereSize.toFixed(4);
        
        // Initialize track spheres
        this.initializeTrackSpheres();
        
        // Add camera frustum if available
        if (data.camera_params) {
            this.addCameraFrustum(data.camera_params);
        }
    }
    
    initializeTrackSpheres() {
        // Clear existing spheres
        Object.values(this.spheres).forEach(sphere => {
            if (sphere) {
                this.scene.remove(sphere);
                sphere.geometry.dispose();
                sphere.material.dispose();
            }
        });
        this.spheres = {};
        
        Object.values(this.labels).forEach(label => {
            if (label) {
                this.scene.remove(label);
            }
        });
        this.labels = {};
        
        // Extract unique track IDs
        const uniqueTrackIds = new Set();
        Object.values(this.trackData || {}).forEach(frameData => {
            frameData.forEach(track => {
                uniqueTrackIds.add(track.track_id);
            });
        });
        
        // Ensure colors are generated
        if (uniqueTrackIds.size > 0) {
            this.trackColors.generateForTracks(Array.from(uniqueTrackIds));
        }
        
        const sphereSize = parseFloat(this.sphereSizeSlider.value);
        const sphereGeometry = new THREE.SphereGeometry(sphereSize, 16, 12);
        
        // Create spheres for each track ID
        uniqueTrackIds.forEach(trackId => {
            const color = this.trackColors.getColorHex(trackId);
            
            const material = new THREE.MeshPhongMaterial({
                color: color,
                emissive: color,
                emissiveIntensity: 0.3,
                shininess: 100
            });
            
            const sphere = new THREE.Mesh(sphereGeometry, material);
            sphere.castShadow = true;
            sphere.visible = false;
            this.spheres[trackId] = sphere;
            this.scene.add(sphere);
            
            // Create label
            const labelDiv = document.createElement('div');
            labelDiv.className = 'label';
            labelDiv.textContent = trackId.toString();
            labelDiv.style.marginTop = '-1em';
            labelDiv.style.color = this.trackColors.getColor(trackId);
            labelDiv.style.fontFamily = 'Arial, sans-serif';
            labelDiv.style.fontSize = '12px';
            labelDiv.style.fontWeight = 'bold';
            labelDiv.style.textShadow = '1px 1px 2px rgba(0,0,0,0.8)';
            labelDiv.style.pointerEvents = 'none';
            
            const label = new CSS2DObject(labelDiv);
            label.position.set(0, 0, 0);
            label.visible = this.showIds3D.checked;
            this.labels[trackId] = label;
            this.scene.add(label);
        });
        
        console.log(`Created ${uniqueTrackIds.size} track spheres`);
    }
    
    addCameraFrustum(cameraParams) {
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
            // Calculate far distance based on mesh size if available
            const meshSize = this.meshObject ? 
                new THREE.Box3().setFromObject(this.meshObject).getSize(new THREE.Vector3()) : 
                new THREE.Vector3(1, 1, 1);
            const far = Math.max(meshSize.x, meshSize.y, meshSize.z) * 0.5;
            
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
            this.cameraFrustum = frustumLines;
            this.scene.add(this.cameraFrustum);
            
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
            this.scene.add(cameraMarker);
            
            // Store marker for visibility toggle
            this.cameraMarker = cameraMarker;
            
            // Show camera info in console
            console.log('Camera frustum added:', {
                position: cameraPosition,
                fov: fov.toFixed(1),
                aspect: aspect.toFixed(2),
                near, far: far.toFixed(2)
            });
            
            // Enable camera visibility checkbox
            this.showCamera3D.disabled = false;
            this.showCamera3D.checked = true;
            
        } catch (error) {
            console.error('Error adding camera frustum:', error);
        }
    }
    
    checkForSyncReady() {
        if (this.currentVideoId && this.currentMeshId) {
            this.syncContent.style.display = 'block';
            this.resizeRenderers();
        }
    }
    
    
    onVideoMetadataLoaded() {
        this.videoMetadata = {
            duration: this.videoPlayer.duration,
            width: this.videoPlayer.videoWidth,
            height: this.videoPlayer.videoHeight
        };
        
        // Calculate total frames (assuming 30 fps)
        const fps = 30;
        const totalFrames = Math.floor(this.videoMetadata.duration * fps);
        
        // Update UI
        this.frameSlider.max = totalFrames;
        this.frameDisplay.textContent = `0 / ${totalFrames}`;
        
        this.resizeCanvas();
        this.checkForSyncReady();
    }
    
    onVideoTimeUpdate() {
        if (!this.videoMetadata) return;
        
        // Calculate current frame from video time
        const fps = 30;
        const currentFrame = Math.floor(this.videoPlayer.currentTime * fps);
        const totalFrames = Math.floor(this.videoMetadata.duration * fps);
        
        // Update UI
        this.frameSlider.value = currentFrame;
        this.frameDisplay.textContent = `${currentFrame} / ${totalFrames}`;
        
        // Update overlays and 3D scene
        this.updateFrame(currentFrame);
        
        // Update play/pause button state
        this.updatePlayPauseButton();
    }
    
    updateFrame(frame) {
        // Update 2D overlays
        this.draw2DOverlays(frame);
        
        // Update 3D scene
        this.update3DFrame(frame);
    }
    
    draw2DOverlays(frame) {
        // Clear canvas
        this.overlayCtx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
        
        if (!this.bboxData[frame] || !this.videoMetadata) return;
        
        // Set opacity
        this.overlayCtx.globalAlpha = parseFloat(this.opacitySlider.value);
        
        // Calculate video display area once per frame
        this.currentDisplayArea = this.getVideoDisplayArea();
        
        const tracks = this.bboxData[frame];
        
        tracks.forEach(track => {
            const color = this.trackColors.getColor(track.track_id);
            
            // Draw bounding box
            this.drawBoundingBox(track, color);
            
            // Update and draw trails
            if (this.showTrails2D.checked) {
                this.updateVideoTrail(track.track_id, track, frame);
                this.drawVideoTrail(track.track_id);
            }
            
            // Draw track ID
            if (this.showIds2D.checked) {
                this.drawTrackId(track, color);
            }
        });
        
        // Draw debug info
        if (this.showDebug2D.checked) {
            this.drawDebugInfo(frame, tracks.length);
        }
    }
    
    getVideoDisplayArea() {
        // Calculate the actual video display area within the canvas
        // considering object-fit: contain which maintains aspect ratio
        const canvasAspect = this.overlayCanvas.width / this.overlayCanvas.height;
        const videoAspect = this.videoMetadata.width / this.videoMetadata.height;
        
        let displayWidth, displayHeight, offsetX, offsetY;
        
        if (videoAspect > canvasAspect) {
            // Video is wider - fit to canvas width
            displayWidth = this.overlayCanvas.width;
            displayHeight = this.overlayCanvas.width / videoAspect;
            offsetX = 0;
            offsetY = (this.overlayCanvas.height - displayHeight) / 2;
        } else {
            // Video is taller - fit to canvas height
            displayWidth = this.overlayCanvas.height * videoAspect;
            displayHeight = this.overlayCanvas.height;
            offsetX = (this.overlayCanvas.width - displayWidth) / 2;
            offsetY = 0;
        }
        
        return { displayWidth, displayHeight, offsetX, offsetY };
    }
    
    drawBoundingBox(track, color) {
        this.overlayCtx.strokeStyle = color;
        this.overlayCtx.lineWidth = 2;
        
        // Use cached video display area
        const { displayWidth, displayHeight, offsetX, offsetY } = this.currentDisplayArea;
        
        // Scale coordinates to actual video display area
        const scaleX = displayWidth / this.videoMetadata.width;
        const scaleY = displayHeight / this.videoMetadata.height;
        
        const x1 = track.x1 * scaleX + offsetX;
        const y1 = track.y1 * scaleY + offsetY;
        const x2 = track.x2 * scaleX + offsetX;
        const y2 = track.y2 * scaleY + offsetY;
        
        this.overlayCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }
    
    drawTrackId(track, color) {
        this.overlayCtx.fillStyle = color;
        this.overlayCtx.font = 'bold 14px Arial';
        this.overlayCtx.textAlign = 'left';
        
        // Use cached video display area
        const { displayWidth, displayHeight, offsetX, offsetY } = this.currentDisplayArea;
        
        // Scale coordinates to actual video display area
        const scaleX = displayWidth / this.videoMetadata.width;
        const scaleY = displayHeight / this.videoMetadata.height;
        
        const x = track.x1 * scaleX + offsetX;
        const y = track.y1 * scaleY + offsetY - 5;
        
        const text = `ID: ${track.track_id}`;
        const metrics = this.overlayCtx.measureText(text);
        
        this.overlayCtx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        this.overlayCtx.fillRect(x - 2, y - 14, metrics.width + 4, 18);
        
        this.overlayCtx.fillStyle = color;
        this.overlayCtx.fillText(text, x, y);
    }
    
    updateVideoTrail(trackId, track, frame) {
        if (!this.videoTrails[trackId]) {
            this.videoTrails[trackId] = [];
        }
        
        const centerX = (track.x1 + track.x2) / 2;
        const centerY = (track.y1 + track.y2) / 2;
        
        this.videoTrails[trackId].push({ x: centerX, y: centerY, frame });
        
        if (this.videoTrails[trackId].length > 30) {
            this.videoTrails[trackId].shift();
        }
    }
    
    drawVideoTrail(trackId) {
        const trail = this.videoTrails[trackId];
        if (!trail || trail.length < 2) return;
        
        const color = this.trackColors.getColor(trackId);
        
        // Use cached video display area
        const { displayWidth, displayHeight, offsetX, offsetY } = this.currentDisplayArea;
        
        // Scale coordinates to actual video display area
        const scaleX = displayWidth / this.videoMetadata.width;
        const scaleY = displayHeight / this.videoMetadata.height;
        
        this.overlayCtx.strokeStyle = color;
        this.overlayCtx.lineWidth = 1;
        this.overlayCtx.globalAlpha = 0.5;
        
        this.overlayCtx.beginPath();
        this.overlayCtx.moveTo(trail[0].x * scaleX + offsetX, trail[0].y * scaleY + offsetY);
        
        for (let i = 1; i < trail.length; i++) {
            this.overlayCtx.lineTo(trail[i].x * scaleX + offsetX, trail[i].y * scaleY + offsetY);
        }
        
        this.overlayCtx.stroke();
        this.overlayCtx.globalAlpha = parseFloat(this.opacitySlider.value);
    }
    
    drawDebugInfo(frame, trackCount) {
        this.overlayCtx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        this.overlayCtx.fillRect(10, 10, 200, 60);
        
        this.overlayCtx.fillStyle = 'white';
        this.overlayCtx.font = '12px monospace';
        this.overlayCtx.textAlign = 'left';
        
        this.overlayCtx.fillText(`Frame: ${frame}`, 20, 30);
        this.overlayCtx.fillText(`Tracks: ${trackCount}`, 20, 45);
        this.overlayCtx.fillText(`FPS: 30`, 20, 60);
    }
    
    update3DFrame(frame) {
        if (!this.trackData) return;
        
        const frameData = this.trackData[frame] || [];
        
        // Hide all spheres and labels
        Object.values(this.spheres).forEach(sphere => {
            if (sphere) sphere.visible = false;
        });
        Object.values(this.labels).forEach(label => {
            if (label) label.visible = false;
        });
        
        // Update positions for visible tracks
        frameData.forEach(track => {
            if (track.x !== null && track.y !== null && track.z !== null) {
                const sphere = this.spheres[track.track_id];
                if (sphere) {
                    // Apply transformation if enabled
                    const pos = this.trackTransformer.applyTransform(track.x, track.y, track.z);

                    sphere.position.set(pos.x, pos.y, pos.z);
                    sphere.visible = true;

                    // Update label
                    const label = this.labels[track.track_id];
                    if (label) {
                        const sphereSize = parseFloat(this.sphereSizeSlider.value);
                        label.position.set(pos.x, pos.y + sphereSize * 2, pos.z);
                        label.visible = this.showIds3D.checked;
                    }

                    // Update trail (with ORIGINAL untransformed position)
                    if (this.showTrails3D.checked) {
                        this.update3DTrail(track.track_id, track.x, track.y, track.z, frame);
                    }
                }
            }
        });

        // Clean up trails for invisible tracks
        const visibleTrackIds = new Set(frameData.filter(t => t.x !== null && t.y !== null && t.z !== null).map(t => t.track_id));

        // Remove trails for tracks that are not visible in this frame
        Object.keys(this.trails3D).forEach(trackId => {
            if (!visibleTrackIds.has(parseInt(trackId))) {
                if (this.trails3D[trackId]) {
                    this.scene.remove(this.trails3D[trackId]);
                    this.trails3D[trackId].geometry.dispose();
                    this.trails3D[trackId].material.dispose();
                    delete this.trails3D[trackId];
                }
            }
        });
    }
    
    rebuildAll3DTrails() {
        // Remove all trail geometries from scene
        Object.values(this.trails3D).forEach(trail => {
            if (trail) {
                this.scene.remove(trail);
                trail.geometry.dispose();
                trail.material.dispose();
            }
        });
        this.trails3D = {};

        // Trail positions are kept (they contain original coordinates)
        // They will be rebuilt with new transformation on next frame update
    }

    update3DTrail(trackId, x, y, z, frame) {
        // Similar to mesh viewer trail logic
        const maxTrailLength = 100;

        if (!this.trailPositions[trackId]) {
            this.trailPositions[trackId] = [];
        }

        // Check if we already have this frame (avoid duplicates)
        const lastPos = this.trailPositions[trackId][this.trailPositions[trackId].length - 1];
        if (lastPos && lastPos.frame === frame) {
            // Update existing position instead of adding duplicate
            lastPos.x = x;
            lastPos.y = y;
            lastPos.z = z;
        } else {
            // Add new position
            this.trailPositions[trackId].push({x, y, z, frame});
        }

        if (this.trailPositions[trackId].length > maxTrailLength) {
            this.trailPositions[trackId].shift();
        }
        
        if (this.trailPositions[trackId].length > 1) {
            // Remove old trail
            if (this.trails3D[trackId]) {
                this.scene.remove(this.trails3D[trackId]);
                this.trails3D[trackId].geometry.dispose();
                this.trails3D[trackId].material.dispose();
            }

            // Create new trail
            const currentSphereSize = parseFloat(this.sphereSizeSlider.value);
            const trailRadius = Math.max(0.001, currentSphereSize / 3);

            // Apply transformation to trail positions when creating geometry
            const transformedPositions = this.trailPositions[trackId].map(pos => {
                const transformed = this.trackTransformer.applyTransform(pos.x, pos.y, pos.z);
                return new THREE.Vector3(transformed.x, transformed.y, transformed.z);
            });

            const curve = new THREE.CatmullRomCurve3(transformedPositions);
            const tubeGeometry = new THREE.TubeGeometry(curve, 32, trailRadius, 8, false);
            const material = new THREE.MeshBasicMaterial({
                color: this.trackColors.getColorHex(trackId),
                opacity: 0.6,
                transparent: true
            });
            
            const trail = new THREE.Mesh(tubeGeometry, material);
            trail.visible = this.showTrails3D.checked;
            this.trails3D[trackId] = trail;
            this.scene.add(trail);
        }
    }
    
    update3DVisibility() {
        // Update mesh visibility
        if (this.meshObject) {
            this.meshObject.visible = this.showMesh3D.checked;
        }
        
        // Update label visibility
        Object.values(this.labels).forEach(label => {
            if (label) {
                label.visible = this.showIds3D.checked;
            }
        });
        
        // Update trail visibility
        Object.values(this.trails3D).forEach(trail => {
            if (trail) {
                trail.visible = this.showTrails3D.checked;
            }
        });
        
        // Update camera visibility
        if (this.cameraFrustum) {
            this.cameraFrustum.visible = this.showCamera3D.checked;
        }
        if (this.cameraMarker) {
            this.cameraMarker.visible = this.showCamera3D.checked;
        }
    }
    
    updateSphereSize(size) {
        Object.values(this.spheres).forEach(sphere => {
            if (sphere) {
                const newGeometry = new THREE.SphereGeometry(size, 16, 12);
                sphere.geometry.dispose();
                sphere.geometry = newGeometry;
            }
        });
    }
    
    clearTrails() {
        // Clear 2D trails
        this.videoTrails = {};
        
        // Clear 3D trails
        Object.values(this.trails3D).forEach(trail => {
            if (trail) {
                this.scene.remove(trail);
                trail.geometry.dispose();
                trail.material.dispose();
            }
        });
        this.trails3D = {};
        this.trailPositions = {};
    }
    
    resizeCanvas() {
        if (this.videoPlayer && this.overlayCanvas) {
            const rect = this.videoPlayer.getBoundingClientRect();
            this.overlayCanvas.width = rect.width;
            this.overlayCanvas.height = rect.height;
        }
    }
    
    resizeRenderers() {
        if (!this.meshCanvas || !this.renderer) return;
        
        const rect = this.meshCanvas.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
        this.labelRenderer.setSize(width, height);
        
        this.resizeCanvas();
    }
    
    animate(time) {
        requestAnimationFrame((time) => this.animate(time));
        
        if (this.controls) {
            this.controls.update();
        }
        
        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }
        
        if (this.labelRenderer && this.scene && this.camera) {
            this.labelRenderer.render(this.scene, this.camera);
        }
    }
    
    updatePlayPauseButton() {
        if (this.videoPlayer && !this.videoPlayer.paused) {
            this.playPauseBtn.textContent = '⏸️ Pause';
        } else {
            this.playPauseBtn.textContent = '▶️ Play';
        }
    }
    
    showStatus(message, type = 'info') {
        this.statusMessage.textContent = message;
        this.statusMessage.className = `status ${type}`;
        this.statusMessage.style.display = 'block';
        
        if (type !== 'error') {
            setTimeout(() => {
                this.statusMessage.style.display = 'none';
            }, 3000);
        }
    }
}