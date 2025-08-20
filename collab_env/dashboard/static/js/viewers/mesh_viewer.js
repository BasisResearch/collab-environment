/**
 * 3D Mesh Viewer with track overlay
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

import { ApiClient } from '../utils/api_client.js';
import { FrameManager } from '../utils/frame_manager.js';
import { TrackColors } from '../utils/track_colors.js';

export class MeshViewer {
    constructor() {
        console.log('🚀 Starting 3D Track Viewer');
        
        // Core components
        this.api = new ApiClient();
        this.frameManager = null;
        this.trackColors = new TrackColors();
        
        // Three.js components
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.labelRenderer = null;
        this.controls = null;
        
        // Scene objects
        this.meshObject = null;
        this.spheres = {}; // track_id -> sphere mesh
        this.trails = {}; // track_id -> line geometry
        this.labels = {}; // track_id -> CSS2D label
        this.cameraFrustum = null;
        
        // DOM elements
        this.loader = document.getElementById('loader');
        this.canvas = document.getElementById('canvas');
        this.controls_panel = document.getElementById('controls');
        this.meshSelect = document.getElementById('meshSelect');
        this.meshControls = document.getElementById('meshControls');
        this.statusElement = document.getElementById('status');
        
        // Control elements
        this.playBtn = document.getElementById('playBtn');
        this.resetBtn = document.getElementById('resetBtn');
        this.frameInfo = document.getElementById('frameInfo');
        this.speedSlider = document.getElementById('speedSlider');
        this.speedValue = document.getElementById('speedValue');
        this.frameSlider = document.getElementById('frameSlider');
        this.frameValue = document.getElementById('frameValue');
        this.sphereSizeSlider = document.getElementById('sphereSizeSlider');
        this.sphereSizeValue = document.getElementById('sphereSizeValue');
        this.refreshBtn = document.getElementById('refreshBtn');
        
        // Display options
        this.showTrails = document.getElementById('showTrails');
        this.showIds = document.getElementById('showIds');
        this.showMesh = document.getElementById('showMesh');
        this.showCamera = document.getElementById('showCamera');
        
        // State
        this.currentMeshId = null;
        this.trackData = null;
        this.trailPositions = {};
        this.maxFrame = 0;
        this.meshCenter = null;
        this.meshSize = null;
        
        // Initialize
        this.init();
    }
    
    async init() {
        this.initScene();
        this.setupEventListeners();
        await this.refreshMeshList();
        this.animate(0);
        
        console.log('✅ 3D Track Viewer ready');
    }
    
    initScene() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x2a2a2a);
        this.scene.fog = new THREE.Fog(0x2a2a2a, 0.1, 100);
        
        // Camera - use container aspect ratio, but with improved controls
        const containerRect = this.canvas.getBoundingClientRect();
        this.camera = new THREE.PerspectiveCamera(
            60, 
            containerRect.width / containerRect.height, 
            0.001, 
            1000
        );
        this.camera.position.set(0.5, 0.5, 0.5);
        this.camera.lookAt(0, 0, 0);
        
        // Renderer - use container dimensions
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(containerRect.width, containerRect.height);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.canvas.appendChild(this.renderer.domElement);
        
        // CSS2D Renderer for labels
        this.labelRenderer = new CSS2DRenderer();
        this.labelRenderer.setSize(containerRect.width, containerRect.height);
        this.labelRenderer.domElement.style.position = 'absolute';
        this.labelRenderer.domElement.style.top = '0px';
        this.labelRenderer.domElement.style.pointerEvents = 'none';
        this.canvas.appendChild(this.labelRenderer.domElement);
        
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
        directionalLight.shadow.camera.near = 0.01;
        directionalLight.shadow.camera.far = 10;
        directionalLight.shadow.camera.left = -1;
        directionalLight.shadow.camera.right = 1;
        directionalLight.shadow.camera.top = 1;
        directionalLight.shadow.camera.bottom = -1;
        this.scene.add(directionalLight);
        
        // Grid helper
        const gridHelper = new THREE.GridHelper(2, 20, 0x444444, 0x222222);
        gridHelper.rotation.x = Math.PI / 2;
        this.scene.add(gridHelper);
        
        // Axes helper
        const axesHelper = new THREE.AxesHelper(0.2);
        this.scene.add(axesHelper);
        
        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());
        
        // Hide loader
        this.loader.style.display = 'none';
        
        console.log('✅ Scene initialized');
    }
    
    setupEventListeners() {
        // Mesh selection
        this.meshSelect.addEventListener('change', (e) => {
            const meshId = e.target.value;
            if (meshId) {
                this.loadMesh(meshId);
            }
        });
        
        // Refresh button
        this.refreshBtn.addEventListener('click', () => {
            this.refreshMeshList();
        });
        
        // Playback controls
        this.playBtn.addEventListener('click', () => {
            this.togglePlayback();
        });
        
        this.resetBtn.addEventListener('click', () => {
            this.resetAnimation();
        });
        
        // Speed control
        this.speedSlider.addEventListener('input', (e) => {
            const speed = parseFloat(e.target.value);
            this.speedValue.textContent = speed.toFixed(1) + 'x';
            if (this.frameManager) {
                this.frameManager.setSpeed(speed);
            }
        });
        
        // Frame control
        this.frameSlider.addEventListener('input', (e) => {
            if (this.frameManager && !this.frameManager.isPlaying) {
                this.frameManager.setFrame(parseInt(e.target.value));
            }
        });
        
        // Sphere size control
        this.sphereSizeSlider.addEventListener('input', (e) => {
            const size = parseFloat(e.target.value);
            this.sphereSizeValue.textContent = size.toFixed(4);
            this.updateSphereSize(size);
        });
        
        // Display options
        this.showTrails.addEventListener('change', () => {
            this.updateTrailVisibility();
        });
        
        this.showIds.addEventListener('change', () => {
            this.updateLabelVisibility();
        });
        
        this.showMesh.addEventListener('change', () => {
            this.updateMeshVisibility();
        });
        
        this.showCamera.addEventListener('change', () => {
            this.updateCameraVisibility();
        });
    }
    
    onWindowResize() {
        const containerRect = this.canvas.getBoundingClientRect();
        this.camera.aspect = containerRect.width / containerRect.height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(containerRect.width, containerRect.height);
        this.labelRenderer.setSize(containerRect.width, containerRect.height);
    }
    
    async refreshMeshList() {
        try {
            this.updateStatus('Loading mesh list...');
            const meshes = await this.api.getMeshes();
            
            // Clear and repopulate dropdown
            this.meshSelect.innerHTML = '<option value="">Choose a mesh...</option>';
            
            meshes.forEach(mesh => {
                const option = document.createElement('option');
                option.value = mesh.id;
                option.textContent = `${mesh.name} (${mesh.tracks} tracks, ${mesh.frames} frames)`;
                this.meshSelect.appendChild(option);
            });
            
            this.updateStatus(`Found ${meshes.length} mesh(es)`);
        } catch (error) {
            console.error('Error loading meshes:', error);
            this.updateStatus('Error loading mesh list', true);
        }
    }
    
    async loadMesh(meshId) {
        try {
            this.updateStatus('Loading mesh data...');
            
            // Clear existing objects
            this.clearScene();
            
            // Fetch mesh metadata
            const data = await this.api.getMeshData(meshId);
            this.trackData = data.tracks_by_frame;
            this.maxFrame = data.num_frames - 1;
            
            // Debug: Log sample of track data structure
            console.log('Track data loaded. Sample frames:', Object.keys(this.trackData).slice(0, 5));
            if (Object.keys(this.trackData).length > 0) {
                const firstFrame = Object.keys(this.trackData)[0];
                console.log(`Sample frame ${firstFrame} data:`, this.trackData[firstFrame]);
            }
            
            // Update UI
            this.frameSlider.max = this.maxFrame;
            this.meshControls.style.display = 'block';
            
            // Load PLY mesh
            const loader = new PLYLoader();
            const meshUrl = this.api.getMeshFileUrl(meshId);
            
            this.updateStatus('Loading 3D mesh...');
            
            return new Promise((resolve, reject) => {
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
                        this.updateStatus(`Loading mesh: ${percent}%`);
                    },
                    (error) => {
                        console.error('Error loading mesh:', error);
                        this.updateStatus('Error loading mesh', true);
                        reject(error);
                    }
                );
            });
            
        } catch (error) {
            console.error('Error:', error);
            this.updateStatus(error.message, true);
        }
    }
    
    processMeshGeometry(geometry, data) {
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
        if (this.meshObject) {
            this.scene.remove(this.meshObject);
            if (this.meshObject.geometry) this.meshObject.geometry.dispose();
            if (this.meshObject.material) this.meshObject.material.dispose();
        }
        
        // Create mesh
        this.meshObject = new THREE.Mesh(geometry, material);
        this.meshObject.castShadow = true;
        this.meshObject.receiveShadow = true;
        this.scene.add(this.meshObject);
        
        // Store mesh reference for visibility toggle
        window.loadedMesh = this.meshObject;
        
        // Center camera on mesh and calculate appropriate sphere size
        const box = new THREE.Box3().setFromObject(this.meshObject);
        this.meshCenter = box.getCenter(new THREE.Vector3());
        this.meshSize = box.getSize(new THREE.Vector3());
        
        // Calculate appropriate sphere size based on mesh extent
        const meshDiameter = Math.max(this.meshSize.x, this.meshSize.y, this.meshSize.z);
        const autoSphereSize = meshDiameter * 0.005; // 0.5% of mesh size
        
        // Update sphere size slider with calculated value
        this.sphereSizeSlider.min = autoSphereSize * 0.1;
        this.sphereSizeSlider.max = autoSphereSize * 10;
        this.sphereSizeSlider.value = autoSphereSize;
        this.sphereSizeValue.textContent = autoSphereSize.toFixed(4);
        
        this.camera.position.set(
            this.meshCenter.x + this.meshSize.x * 0.5,
            this.meshCenter.y + this.meshSize.y * 0.5,
            this.meshCenter.z + this.meshSize.z * 0.5
        );
        this.controls.target.copy(this.meshCenter);
        this.controls.update();
        
        this.updateStatus(`Mesh loaded: ${data.name} (sphere size: ${autoSphereSize.toFixed(4)})`);
        
        // Initialize track spheres with calculated size
        this.initializeTrackSpheres(data.num_tracks, autoSphereSize);
        
        // Initialize frame manager
        if (this.frameManager) {
            this.frameManager.destroy();
        }
        this.frameManager = new FrameManager(this.maxFrame);
        this.frameManager.onFrameChange((frame) => this.onFrameChange(frame));
        
        // Store mesh info for other functions
        window.meshCenter = this.meshCenter;
        window.meshSize = this.meshSize;
        
        // Show camera frustum if available
        console.log('Camera params from server:', data.camera_params);
        if (data.camera_params) {
            this.addCameraFrustum(data.camera_params);
        } else {
            console.log('No camera parameters available - camera frustum disabled');
        }
    }
    
    initializeTrackSpheres(numTracks, sphereSize = null) {
        if (sphereSize === null) {
            sphereSize = parseFloat(this.sphereSizeSlider.value);
        }
        const sphereGeometry = new THREE.SphereGeometry(sphereSize, 16, 12);
        
        // Clear existing spheres
        Object.values(this.spheres).forEach(sphere => {
            if (sphere) {
                this.scene.remove(sphere);
                sphere.geometry.dispose();
                sphere.material.dispose();
            }
        });
        this.spheres = {};
        
        // Extract all unique track IDs from the track data
        const uniqueTrackIds = new Set();
        Object.values(this.trackData || {}).forEach(frameData => {
            frameData.forEach(track => {
                uniqueTrackIds.add(track.track_id);
            });
        });
        
        console.log('Unique track IDs found:', Array.from(uniqueTrackIds).sort());
        
        // Generate colors for all track IDs
        this.trackColors.clear();
        this.trackColors.generateForTracks(Array.from(uniqueTrackIds));
        
        // Create spheres for each actual track ID
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
            this.spheres[trackId] = sphere; // Use track ID as key
            this.scene.add(sphere);
            
            // Create label for this track
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
            label.visible = this.showIds.checked;
            this.labels[trackId] = label;
            this.scene.add(label);
        });
        
        console.log(`Created ${uniqueTrackIds.size} spheres for track IDs: ${Array.from(uniqueTrackIds).sort()}`);
    }
    
    onFrameChange(frame) {
        this.updateFrame(frame);
        
        // Update UI
        this.frameValue.textContent = frame;
        this.frameSlider.value = frame;
    }
    
    updateFrame(frame) {
        if (!this.trackData) return;
        
        const frameData = this.trackData[frame] || [];
        
        // Hide all spheres and labels first
        Object.values(this.spheres).forEach(sphere => {
            if (sphere) sphere.visible = false;
        });
        Object.values(this.labels).forEach(label => {
            if (label) label.visible = false;
        });
        
        let visibleTracks = 0;
        // Update positions for visible tracks
        frameData.forEach(track => {
            if (track.x !== null && track.y !== null && track.z !== null) {
                const sphere = this.spheres[track.track_id];
                if (sphere) {
                    sphere.position.set(track.x, track.y, track.z);
                    sphere.visible = true;
                    visibleTracks++;
                    
                    // Update label position to be slightly above the sphere
                    const label = this.labels[track.track_id];
                    if (label) {
                        const sphereSize = parseFloat(this.sphereSizeSlider.value);
                        label.position.set(track.x, track.y + sphereSize * 2, track.z);
                        label.visible = this.showIds.checked;
                    }
                    
                    // Update trail for this track
                    this.updateTrail(track.track_id, track.x, track.y, track.z, frame);
                } else {
                    console.warn(`No sphere found for track_id: ${track.track_id}`);
                }
            }
        });
        
        // Hide trails for invisible tracks and clean up old trail data
        const visibleTrackIds = new Set(frameData.filter(t => t.x !== null && t.y !== null && t.z !== null).map(t => t.track_id));
        
        // Hide trails for tracks that are not visible in this frame
        Object.keys(this.trails).forEach(trackId => {
            if (!visibleTrackIds.has(parseInt(trackId))) {
                if (this.trails[trackId]) {
                    this.trails[trackId].visible = false;
                }
            }
        });
        
        // Clean up trail positions for tracks that haven't been seen for a while
        if (this.trailPositions) {
            const maxInactiveFrames = 60; // Clean up after 60 frames of inactivity
            Object.keys(this.trailPositions).forEach(trackId => {
                const positions = this.trailPositions[trackId];
                if (positions.length > 0) {
                    const lastFrame = positions[positions.length - 1].frame;
                    if (frame - lastFrame > maxInactiveFrames) {
                        // Clean up old trail
                        if (this.trails[trackId]) {
                            this.scene.remove(this.trails[trackId]);
                            this.trails[trackId].geometry.dispose();
                            this.trails[trackId].material.dispose();
                            delete this.trails[trackId];
                        }
                        delete this.trailPositions[trackId];
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
        this.frameInfo.textContent = `Frame: ${frame} / ${this.maxFrame} | Tracks: ${frameData.length} | Visible: ${visibleTracks}`;
    }
    
    updateTrail(trackId, x, y, z, frame) {
        const maxTrailLength = 100; // Keep last 100 positions for longer trails
        
        // Initialize trail data if needed
        if (!this.trailPositions[trackId]) {
            this.trailPositions[trackId] = [];
        }
        
        // Add new position
        this.trailPositions[trackId].push({x, y, z, frame});
        
        // Keep only recent positions
        if (this.trailPositions[trackId].length > maxTrailLength) {
            this.trailPositions[trackId].shift();
        }
        
        // Update or create trail geometry
        if (this.trailPositions[trackId].length > 1) {
            const positions = [];
            this.trailPositions[trackId].forEach(pos => {
                positions.push(pos.x, pos.y, pos.z);
            });
            
            // Remove old trail
            if (this.trails[trackId]) {
                this.scene.remove(this.trails[trackId]);
                this.trails[trackId].geometry.dispose();
                this.trails[trackId].material.dispose();
            }
            
            // Get current sphere size for proportional trail width (1/3 of sphere radius)
            const currentSphereSize = parseFloat(this.sphereSizeSlider.value);
            const trailRadius = Math.max(0.001, currentSphereSize / 3); // 1/3 of sphere radius
            
            // Use TubeGeometry instead of Line for visible width
            const curve = new THREE.CatmullRomCurve3(
                this.trailPositions[trackId].map(pos => new THREE.Vector3(pos.x, pos.y, pos.z))
            );
            const tubeGeometry = new THREE.TubeGeometry(curve, 32, trailRadius, 8, false);
            const material = new THREE.MeshBasicMaterial({
                color: this.trackColors.getColorHex(trackId),
                opacity: 0.6,
                transparent: true
            });
            
            const trail = new THREE.Mesh(tubeGeometry, material);
            trail.visible = this.showTrails.checked;
            this.trails[trackId] = trail;
            this.scene.add(trail);
        }
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
            const far = this.meshSize ? Math.max(this.meshSize.x, this.meshSize.y, this.meshSize.z) * 0.5 : 1.0;
            
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
            
            // Store frustum reference globally for visibility toggle
            window.cameraFrustum = this.cameraFrustum;
            
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
            window.cameraMarker = cameraMarker;
            
            // Show camera info in console
            console.log('Camera frustum added:', {
                position: cameraPosition,
                fov: fov.toFixed(1),
                aspect: aspect.toFixed(2),
                near, far: far.toFixed(2)
            });
            
            // Enable camera visibility checkbox
            this.showCamera.disabled = false;
            this.showCamera.checked = true;
            
        } catch (error) {
            console.error('Error adding camera frustum:', error);
        }
    }
    
    clearScene() {
        if (this.meshObject) {
            this.scene.remove(this.meshObject);
            this.meshObject.geometry.dispose();
            this.meshObject.material.dispose();
            this.meshObject = null;
        }
        
        Object.values(this.spheres).forEach(sphere => {
            if (sphere) {
                this.scene.remove(sphere);
                sphere.geometry.dispose();
                sphere.material.dispose();
            }
        });
        this.spheres = {};
        
        Object.values(this.trails).forEach(trail => {
            if (trail) {
                this.scene.remove(trail);
                trail.geometry.dispose();
                trail.material.dispose();
            }
        });
        this.trails = {};
        
        Object.values(this.labels).forEach(label => {
            if (label) {
                this.scene.remove(label);
            }
        });
        this.labels = {};
        
        this.trailPositions = {};
    }
    
    animate(time) {
        requestAnimationFrame((time) => this.animate(time));
        
        // Update controls
        this.controls.update();
        
        // Render
        this.renderer.render(this.scene, this.camera);
        this.labelRenderer.render(this.scene, this.camera);
    }
    
    // UI Control Methods
    togglePlayback() {
        if (this.frameManager) {
            this.frameManager.togglePlayPause();
            this.playBtn.textContent = this.frameManager.isPlaying ? '⏸️ Pause' : '▶️ Play';
        }
    }
    
    resetAnimation() {
        if (this.frameManager) {
            this.frameManager.jumpToStart();
            this.frameManager.pause();
            this.playBtn.textContent = '▶️ Play';
        }
        
        // Clear all trails
        Object.values(this.trails).forEach(trail => {
            if (trail) {
                this.scene.remove(trail);
                trail.geometry.dispose();
                trail.material.dispose();
            }
        });
        this.trails = {};
        this.trailPositions = {};
    }
    
    updateSphereSize(size) {
        // Update all sphere geometries with new size
        Object.values(this.spheres).forEach(sphere => {
            if (sphere) {
                const newGeometry = new THREE.SphereGeometry(size, 16, 12);
                sphere.geometry.dispose(); // Clean up old geometry
                sphere.geometry = newGeometry;
            }
        });
        
        console.log(`Updated ${Object.keys(this.spheres).length} spheres to size ${size.toFixed(4)}`);
    }
    
    updateMeshVisibility() {
        if (this.meshObject) {
            this.meshObject.visible = this.showMesh.checked;
        }
    }
    
    updateLabelVisibility() {
        const showLabels = this.showIds.checked;
        console.log('Label visibility:', showLabels);
        
        // Toggle visibility of all track ID labels
        Object.values(this.labels).forEach(label => {
            if (label) {
                label.visible = showLabels;
            }
        });
        
        // Force update current frame to apply proper visibility
        if (this.frameManager) {
            this.updateFrame(this.frameManager.currentFrame);
        }
    }
    
    updateTrailVisibility() {
        const showTrails = this.showTrails.checked;
        console.log('Trail visibility:', showTrails);
        
        // Toggle visibility of all trail lines
        Object.values(this.trails).forEach(trail => {
            if (trail) {
                trail.visible = showTrails;
            }
        });
    }
    
    updateCameraVisibility() {
        const isVisible = this.showCamera.checked;
        if (this.cameraFrustum) {
            this.cameraFrustum.visible = isVisible;
        }
        if (window.cameraMarker) {
            window.cameraMarker.visible = isVisible;
        }
    }
    
    updateStatus(message, isError = false) {
        this.statusElement.textContent = message;
        this.statusElement.className = isError ? 'mesh-status error' : 'mesh-status';
    }
}