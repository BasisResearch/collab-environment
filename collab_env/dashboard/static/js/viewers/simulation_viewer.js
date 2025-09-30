/**
 * Simulation Viewer - extends MeshViewer for boid simulation episode playback
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

import { ApiClient } from '../utils/api_client.js';
import { FrameManager } from '../utils/frame_manager.js';
import { TrackColors } from '../utils/track_colors.js';

export class SimulationViewer {
    constructor() {
        console.log('🚀 Starting Simulation Viewer');

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
        this.sceneMesh = null;
        this.targetMesh = null;
        this.spheres = {}; // track_id -> sphere mesh
        this.trails = {}; // track_id -> line geometry
        this.labels = {}; // track_id -> CSS2D label

        // DOM elements
        this.loader = document.getElementById('loader');
        this.canvas = document.getElementById('canvas');
        this.simulationControls = document.getElementById('simulationControls');
        this.playbackControls = document.getElementById('controls');
        this.statusElement = document.getElementById('status');

        // Simulation control elements
        this.simulationSelect = document.getElementById('simulationSelect');
        this.episodeSelect = document.getElementById('episodeSelect');
        this.loadBtn = document.getElementById('loadBtn');
        this.refreshBtn = document.getElementById('refreshBtn');
        this.configDisplay = document.getElementById('configDisplay');
        this.configInfo = document.getElementById('configInfo');
        this.meshStatus = document.getElementById('meshStatus');

        // Playback control elements
        this.playBtn = document.getElementById('playBtn');
        this.resetBtn = document.getElementById('resetBtn');
        this.frameInfo = document.getElementById('frameInfo');
        this.speedSlider = document.getElementById('speedSlider');
        this.speedValue = document.getElementById('speedValue');
        this.frameSlider = document.getElementById('frameSlider');
        this.frameValue = document.getElementById('frameValue');
        this.sphereSizeSlider = document.getElementById('sphereSizeSlider');
        this.sphereSizeValue = document.getElementById('sphereSizeValue');

        // Display options
        this.showTrails = document.getElementById('showTrails');
        this.showIds = document.getElementById('showIds');
        this.showSceneMesh = document.getElementById('showSceneMesh');
        this.showTargetMesh = document.getElementById('showTargetMesh');

        // State
        this.currentSimulationId = null;
        this.currentEpisodeId = null;
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
        await this.refreshSimulations();
        this.animate(0);

        console.log('✅ Simulation Viewer ready');
    }

    initScene() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x2a2a2a);
        this.scene.fog = new THREE.Fog(0x2a2a2a, 0.1, 100);

        // Camera
        const containerRect = this.canvas.getBoundingClientRect();
        this.camera = new THREE.PerspectiveCamera(
            60,
            containerRect.width / containerRect.height,
            0.001,
            1000
        );
        this.camera.position.set(0.5, 0.5, 0.5);
        this.camera.lookAt(0, 0, 0);

        // Renderer
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

        if (false) {
            // Grid helper
            const gridHelper = new THREE.GridHelper(2, 20, 0x444444, 0x222222);
            gridHelper.rotation.x = Math.PI / 2;
            this.scene.add(gridHelper);

            // Axes helper
            const axesHelper = new THREE.AxesHelper(0.2);
            this.scene.add(axesHelper);
        }

        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());

        // Hide loader
        this.loader.style.display = 'none';

        console.log('✅ Scene initialized');
    }

    setupEventListeners() {
        // Simulation selection
        this.simulationSelect.addEventListener('change', (e) => {
            const simulationId = e.target.value;
            if (simulationId) {
                this.loadEpisodes(simulationId);
            } else {
                this.episodeSelect.innerHTML = '<option value="">Select simulation first</option>';
                this.episodeSelect.disabled = true;
                this.loadBtn.disabled = true;
                this.configDisplay.style.display = 'none';
            }
        });

        // Episode selection
        this.episodeSelect.addEventListener('change', (e) => {
            const episodeId = e.target.value;
            this.loadBtn.disabled = episodeId === '';
        });

        // Load button
        this.loadBtn.addEventListener('click', () => {
            const simulationId = this.simulationSelect.value;
            const episodeId = parseInt(this.episodeSelect.value);
            if (simulationId && episodeId !== '') {
                this.loadEpisode(simulationId, episodeId);
            }
        });

        // Refresh button
        this.refreshBtn.addEventListener('click', () => {
            this.refreshSimulations();
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
        this.showTrails.addEventListener('change', (e) => {
            this.setTrailsVisible(e.target.checked);
        });

        this.showIds.addEventListener('change', (e) => {
            this.setLabelsVisible(e.target.checked);
        });

        this.showSceneMesh.addEventListener('change', (e) => {
            if (this.sceneMesh) {
                this.sceneMesh.visible = e.target.checked;
            }
        });

        this.showTargetMesh.addEventListener('change', (e) => {
            if (this.targetMesh) {
                this.targetMesh.visible = e.target.checked;
            }
        });
    }

    async refreshSimulations() {
        try {
            const response = await fetch('/api/simulations');
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const simulations = await response.json();

            // Update simulation dropdown
            this.simulationSelect.innerHTML = '';
            if (simulations.length === 0) {
                this.simulationSelect.innerHTML = '<option value="">No simulations available</option>';
            } else {
                this.simulationSelect.innerHTML = '<option value="">Select simulation...</option>';
                simulations.forEach(sim => {
                    const option = document.createElement('option');
                    option.value = sim.id;
                    option.textContent = `${sim.name} (${sim.num_episodes} episodes)`;
                    this.simulationSelect.appendChild(option);
                });
            }

            console.log(`Loaded ${simulations.length} simulations`, simulations);
            console.log('Dropdown element:', this.simulationSelect, 'Options count:', this.simulationSelect.children.length);

        } catch (error) {
            console.error('Error loading simulations:', error);
            this.simulationSelect.innerHTML = '<option value="">Error loading simulations</option>';
        }
    }

    async loadEpisodes(simulationId) {
        try {
            const response = await fetch(`/api/simulation/${simulationId}/episodes`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const episodes = await response.json();

            // Update episode dropdown
            this.episodeSelect.innerHTML = '';
            if (episodes.length === 0) {
                this.episodeSelect.innerHTML = '<option value="">No episodes available</option>';
            } else {
                this.episodeSelect.innerHTML = '<option value="">Select episode...</option>';
                episodes.forEach(episode => {
                    const option = document.createElement('option');
                    option.value = episode.id;
                    option.textContent = episode.name;
                    this.episodeSelect.appendChild(option);
                });
            }

            this.episodeSelect.disabled = false;

            // Show simulation info
            const simulations = await (await fetch('/api/simulations')).json();
            const simInfo = simulations.find(s => s.id === simulationId);
            if (simInfo) {
                this.displaySimulationConfig(simInfo);
            }

            console.log(`Loaded ${episodes.length} episodes for simulation ${simulationId}`);

        } catch (error) {
            console.error('Error loading episodes:', error);
            this.episodeSelect.innerHTML = '<option value="">Error loading episodes</option>';
        }
    }

    displaySimulationConfig(simInfo) {
        this.configInfo.innerHTML = `
            <strong>${simInfo.name}</strong><br>
            Agents: ${simInfo.num_agents} | Frames: ${simInfo.num_frames} | Episodes: ${simInfo.num_episodes}
        `;

        this.meshStatus.innerHTML = `
            <span class="status-indicator ${simInfo.mesh_status.scene_found ? 'status-found' : 'status-missing'}">
                Scene: ${simInfo.mesh_status.scene_found ? '✓' : '✗'}
            </span>
            <span class="status-indicator ${simInfo.mesh_status.target_found ? 'status-found' : 'status-missing'}">
                Target: ${simInfo.mesh_status.target_found ? '✓' : '✗'}
            </span>
        `;

        this.configDisplay.style.display = 'block';
    }

    async loadEpisode(simulationId, episodeId) {
        try {
            this.updateStatus('Loading episode...');

            const response = await fetch(`/api/simulation/${simulationId}/episode/${episodeId}`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();

            this.currentSimulationId = simulationId;
            this.currentEpisodeId = episodeId;
            this.trackData = data.frames;
            this.maxFrame = data.num_frames - 1;

            // Store episode config for coordinate transformations
            this.episodeConfig = data.config;

            // Load meshes
            await this.loadMeshes(simulationId, data.config.meshes);

            // Initialize spheres
            this.initializeTrackSpheres(data.num_tracks);

            // Initialize frame manager
            if (this.frameManager) {
                this.frameManager.destroy();
            }
            this.frameManager = new FrameManager(this.maxFrame);
            this.frameManager.onFrameChange((frame) => this.onFrameChange(frame));

            // Update UI
            this.frameSlider.max = this.maxFrame;
            this.frameSlider.value = 0;
            this.frameValue.textContent = '0';
            this.frameInfo.textContent = `Frame: 0 / ${this.maxFrame}`;

            // Show playback controls
            this.playbackControls.style.display = 'block';

            this.updateStatus(`Episode loaded: ${data.num_tracks} agents, ${data.num_frames} frames`);

            console.log(`Episode ${episodeId} loaded:`, data);

        } catch (error) {
            console.error('Error loading episode:', error);
            this.updateStatus(`Error: ${error.message}`);
        }
    }

    async loadMeshes(simulationId, meshPaths) {
        const plyLoader = new PLYLoader();

        // Load scene mesh
        if (meshPaths.scene_path) {
            try {
                const sceneUrl = `/api/simulation/${simulationId}/mesh/scene`;
                const sceneGeometry = await plyLoader.loadAsync(sceneUrl);

                // Remove existing scene mesh
                if (this.sceneMesh) {
                    this.scene.remove(this.sceneMesh);
                }

                // Check if geometry has vertex colors
                const hasVertexColors = sceneGeometry.attributes.color !== undefined;

                // Use vertex colors if available, otherwise use default material
                const sceneMaterial = hasVertexColors ?
                    new THREE.MeshStandardMaterial({
                        vertexColors: true,
                        transparent: true,
                        opacity: 0.7,
                        side: THREE.DoubleSide
                    }) :
                    new THREE.MeshStandardMaterial({
                        color: 0x888888,
                        transparent: true,
                        opacity: 0.3,
                        side: THREE.DoubleSide
                    });

                this.sceneMesh = new THREE.Mesh(sceneGeometry, sceneMaterial);

                // Get transformation parameters from config
                const sceneScale = this.episodeConfig?.scene_scale || 300.0;
                const scenePosition = this.episodeConfig?.scene_position || [0, 0, 0];
                const sceneAngle = this.episodeConfig?.scene_angle || [0, 0, 0];

                // Apply inverse rotation to scene mesh (same as target mesh)
                // Both meshes need inverse rotation to align with tracks
                // this.sceneMesh.rotation.set(
                //     THREE.MathUtils.degToRad(-sceneAngle[0]),
                //     THREE.MathUtils.degToRad(-sceneAngle[1]),
                //     THREE.MathUtils.degToRad(-sceneAngle[2])
                // );

                // 2. Translation (divide by scale to match agent coordinate system)
                this.sceneMesh.position.set(
                    scenePosition[0] / sceneScale,
                    scenePosition[1] / sceneScale,
                    scenePosition[2] / sceneScale
                );

                console.log('Scene mesh transformations applied:', {
                    rotation: [-sceneAngle[0], -sceneAngle[1], -sceneAngle[2]],
                    scale: sceneScale,
                    position: [scenePosition[0] / sceneScale, scenePosition[1] / sceneScale, scenePosition[2] / sceneScale],
                    hasVertexColors: hasVertexColors
                });

                this.scene.add(this.sceneMesh);

                console.log('Scene mesh loaded');

            } catch (error) {
                console.warn('Failed to load scene mesh:', error);
            }
        }

        // Load target mesh
        if (meshPaths.target_path) {
            try {
                const targetUrl = `/api/simulation/${simulationId}/mesh/target`;
                const targetGeometry = await plyLoader.loadAsync(targetUrl);

                // Remove existing target mesh
                if (this.targetMesh) {
                    this.scene.remove(this.targetMesh);
                }

                // Check if geometry has vertex colors
                const hasVertexColors = targetGeometry.attributes.color !== undefined;

                // Use vertex colors if available, otherwise use default green material
                const targetMaterial = hasVertexColors ?
                    new THREE.MeshStandardMaterial({
                        vertexColors: true,
                        transparent: true,
                        opacity: 0.9,
                        side: THREE.DoubleSide
                    }) :
                    new THREE.MeshStandardMaterial({
                        color: 0x00ff00,
                        transparent: true,
                        opacity: 0.8,
                        side: THREE.DoubleSide
                    });

                this.targetMesh = new THREE.Mesh(targetGeometry, targetMaterial);

                // Get transformation parameters from config
                const sceneScale = this.episodeConfig?.scene_scale || 300.0;
                const scenePosition = this.episodeConfig?.scene_position || [0, 0, 0];
                const sceneAngle = this.episodeConfig?.scene_angle || [0, 0, 0];

                // Apply inverse rotation to target mesh for alignment
                // The target mesh needs inverse rotation relative to scene mesh
                this.targetMesh.rotation.set(
                    THREE.MathUtils.degToRad(-sceneAngle[0]),
                    THREE.MathUtils.degToRad(-sceneAngle[1]),
                    THREE.MathUtils.degToRad(-sceneAngle[2])
                );

                // 2. Translation (divide by scale to match agent coordinate system)
                this.targetMesh.position.set(
                    scenePosition[0] / sceneScale,
                    scenePosition[1] / sceneScale,
                    scenePosition[2] / sceneScale
                );

                console.log('Target mesh transformations applied:', {
                    rotation: [-sceneAngle[0], -sceneAngle[1], -sceneAngle[2]],
                    scale: sceneScale,
                    position: [scenePosition[0] / sceneScale, scenePosition[1] / sceneScale, scenePosition[2] / sceneScale],
                    hasVertexColors: hasVertexColors
                });

                this.scene.add(this.targetMesh);

                console.log('Target mesh loaded');

            } catch (error) {
                console.warn('Failed to load target mesh:', error);
            }
        }

        // Calculate combined bounding box for camera positioning
        this.calculateSceneBounds();
    }

    calculateSceneBounds() {
        const boundingBox = new THREE.Box3();

        // Calculate bounding box in world space (including transformations)
        if (this.sceneMesh) {
            // Update world matrix to ensure transformations are applied
            this.sceneMesh.updateMatrixWorld();
            const box = new THREE.Box3().setFromObject(this.sceneMesh);
            boundingBox.union(box);
        }

        if (this.targetMesh) {
            // Update world matrix to ensure transformations are applied
            this.targetMesh.updateMatrixWorld();
            const box = new THREE.Box3().setFromObject(this.targetMesh);
            boundingBox.union(box);
        }

        if (!boundingBox.isEmpty()) {
            const center = boundingBox.getCenter(new THREE.Vector3());
            const size = boundingBox.getSize(new THREE.Vector3());

            // Position camera to see the entire scaled scene
            const maxDim = Math.max(size.x, size.y, size.z);
            const distance = maxDim * 1.5; // Distance multiplier for good view

            this.camera.position.set(
                center.x + distance,
                center.y + distance * 0.5,
                center.z + distance
            );

            this.camera.lookAt(center);
            this.controls.target.copy(center);
            this.controls.update();

            // Adjust camera clipping planes for large scene
            this.camera.near = maxDim * 0.001;
            this.camera.far = maxDim * 10;
            this.camera.updateProjectionMatrix();

            // Auto-adjust sphere size based on scaled mesh
            const autoSphereSize = maxDim * 0.002; // Small spheres for large scene
            this.sphereSizeSlider.min = autoSphereSize * 0.1;
            this.sphereSizeSlider.max = autoSphereSize * 10;
            this.sphereSizeSlider.value = autoSphereSize;
            this.sphereSizeValue.textContent = autoSphereSize.toFixed(4);

            console.log('Scene bounds updated:', {
                center: center,
                size: size,
                maxDim: maxDim,
                cameraDistance: distance
            });
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
            }
        });
        this.spheres = {};

        // Clear existing labels
        Object.values(this.labels).forEach(label => {
            if (label) {
                this.scene.remove(label);
            }
        });
        this.labels = {};

        // Clear existing trails
        Object.values(this.trails).forEach(trail => {
            if (trail) {
                this.scene.remove(trail);
            }
        });
        this.trails = {};
        this.trailPositions = {};

        console.log(`Initialized for ${numTracks} tracks with sphere size ${sphereSize.toFixed(4)}`);
    }

    onFrameChange(frame) {
        if (!this.trackData || !this.trackData[frame]) {
            return;
        }

        const tracks = this.trackData[frame];

        // Update frame display
        this.frameValue.textContent = frame.toString();
        this.frameInfo.textContent = `Frame: ${frame} / ${this.maxFrame}`;
        this.frameSlider.value = frame;

        // Update spheres and trails
        tracks.forEach(track => {
            this.updateTrack(track, frame);
        });

        // Hide inactive tracks
        Object.keys(this.spheres).forEach(trackId => {
            const trackFound = tracks.some(t => t.track_id.toString() === trackId);
            if (!trackFound && this.spheres[trackId]) {
                this.spheres[trackId].visible = false;
                if (this.labels[trackId]) {
                    this.labels[trackId].visible = false;
                }
            }
        });
    }

    updateTrack(track, frame) {
        const trackId = track.track_id.toString();

        // Divide track coordinates by scene_scale (but NOT relative to scene_position)
        const sceneScale = this.episodeConfig?.scene_scale || 300.0;
        const position = new THREE.Vector3(
            track.x / sceneScale,
            track.y / sceneScale,
            track.z / sceneScale
        );

        // Create or update sphere
        if (!this.spheres[trackId]) {
            const sphereGeometry = new THREE.SphereGeometry(parseFloat(this.sphereSizeSlider.value), 16, 12);
            const color = this.trackColors.getColor(track.track_id);
            const sphereMaterial = new THREE.MeshStandardMaterial({ color: color });

            this.spheres[trackId] = new THREE.Mesh(sphereGeometry, sphereMaterial);
            this.scene.add(this.spheres[trackId]);

            // Create label
            const labelDiv = document.createElement('div');
            labelDiv.textContent = trackId;
            labelDiv.style.color = `#${color.toString(16).padStart(6, '0')}`;
            labelDiv.style.fontSize = '12px';
            labelDiv.style.fontWeight = 'bold';
            labelDiv.style.textShadow = '1px 1px 2px rgba(0,0,0,0.8)';

            this.labels[trackId] = new CSS2DObject(labelDiv);
            this.labels[trackId].position.set(0, 0.05, 0);
            this.spheres[trackId].add(this.labels[trackId]);

            // Initialize trail
            this.trailPositions[trackId] = [];
        }

        // Update position
        this.spheres[trackId].position.copy(position);
        this.spheres[trackId].visible = true;

        if (this.labels[trackId]) {
            this.labels[trackId].visible = this.showIds.checked;
        }

        // Update trail
        if (this.showTrails.checked) {
            this.updateTrail(trackId, position);
        }
    }

    updateTrail(trackId, position) {
        if (!this.trailPositions[trackId]) {
            this.trailPositions[trackId] = [];
        }

        // Only add valid positions (avoid NaN/undefined that could cause rays from origin)
        if (position && !isNaN(position.x) && !isNaN(position.y) && !isNaN(position.z)) {
            // Also check that position is not at origin (0,0,0) which could cause rays
            if (position.length() > 0.001) {
                this.trailPositions[trackId].push(position.clone());
            }
        }

        // Limit trail length
        const maxTrailLength = 50;
        if (this.trailPositions[trackId].length > maxTrailLength) {
            this.trailPositions[trackId].shift();
        }

        // Update trail line
        if (this.trails[trackId]) {
            this.scene.remove(this.trails[trackId]);
        }

        if (this.trailPositions[trackId].length > 1) {
            const lineGeometry = new THREE.BufferGeometry().setFromPoints(this.trailPositions[trackId]);
            const color = this.trackColors.getColor(parseInt(trackId));
            const lineMaterial = new THREE.LineBasicMaterial({ color: color, opacity: 0.6, transparent: true });

            this.trails[trackId] = new THREE.Line(lineGeometry, lineMaterial);
            this.scene.add(this.trails[trackId]);
        }
    }

    updateSphereSize(size) {
        Object.values(this.spheres).forEach(sphere => {
            if (sphere && sphere.geometry) {
                sphere.geometry.dispose();
                sphere.geometry = new THREE.SphereGeometry(size, 16, 12);
            }
        });
    }

    setTrailsVisible(visible) {
        Object.values(this.trails).forEach(trail => {
            if (trail) {
                trail.visible = visible;
            }
        });

        if (!visible) {
            // Clear trail positions
            this.trailPositions = {};
        }
    }

    setLabelsVisible(visible) {
        Object.values(this.labels).forEach(label => {
            if (label) {
                label.visible = visible;
            }
        });
    }

    togglePlayback() {
        if (!this.frameManager) return;

        if (this.frameManager.isPlaying) {
            this.frameManager.pause();
            this.playBtn.innerHTML = '▶️ Play';
        } else {
            this.frameManager.play();
            this.playBtn.innerHTML = '⏸️ Pause';
        }
    }

    resetAnimation() {
        if (this.frameManager) {
            this.frameManager.pause();
            this.frameManager.jumpToStart();
            this.playBtn.innerHTML = '▶️ Play';
        }

        // Clear trails
        this.trailPositions = {};
        Object.values(this.trails).forEach(trail => {
            if (trail) {
                this.scene.remove(trail);
            }
        });
        this.trails = {};
    }

    updateStatus(message) {
        if (this.statusElement) {
            this.statusElement.textContent = message;
        }
        console.log('Status:', message);
    }

    onWindowResize() {
        const containerRect = this.canvas.getBoundingClientRect();
        this.camera.aspect = containerRect.width / containerRect.height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(containerRect.width, containerRect.height);
        this.labelRenderer.setSize(containerRect.width, containerRect.height);
    }

    animate(timestamp) {
        requestAnimationFrame((ts) => this.animate(ts));

        this.controls.update();

        // FrameManager has its own internal animation loop
        // No need to call update - it manages itself via play/pause

        this.renderer.render(this.scene, this.camera);
        this.labelRenderer.render(this.scene, this.camera);
    }
}