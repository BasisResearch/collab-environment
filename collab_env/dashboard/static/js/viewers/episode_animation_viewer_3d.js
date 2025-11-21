/**
 * Episode Animation Viewer 3D - client-side 3D animation for episode tracks
 *
 * Handles 3D visualization with smooth playback using Three.js WebGL.
 * Data is embedded directly from Panel - no HTTP requests needed.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { FrameManager } from '../utils/frame_manager.js';

export class EpisodeAnimationViewer {
    constructor(animationData, canvasId) {
        console.log('🚀 Starting Episode Animation Viewer 3D');

        this.data = animationData;
        this.canvasId = canvasId;

        // Validate data
        if (!this.data || !this.data.tracks || this.data.tracks.length === 0) {
            console.error('No track data available');
            return;
        }

        // Canvas element (or container)
        this.canvasElement = document.getElementById(canvasId);
        if (!this.canvasElement) {
            console.error(`Canvas element not found: ${canvasId}`);
            return;
        }

        // Determine if we have a canvas or container
        this.isCanvasElement = this.canvasElement.tagName.toLowerCase() === 'canvas';

        // Three.js components
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.labelRenderer = null;
        this.controls = null;

        // Scene objects
        this.spheres = {}; // agent_id -> sphere mesh
        this.trails = {}; // agent_id -> line/tube mesh
        this.labels = {}; // agent_id -> CSS2D label
        this.trailPositions = {}; // agent_id -> [{x, y, z, time}, ...]

        // Frame management
        const timeRange = this.data.time_range || [0, 100];
        this.frameManager = new FrameManager(timeRange[1] - timeRange[0]);
        this.frameManager.setSpeed(this.data.settings.playback_speed || 1.0);

        // Settings from Panel
        this.trailLength = this.data.settings.trail_length || 50;
        this.showIds = this.data.settings.show_agent_ids !== false;

        // Bounds from data
        this.xRange = this.data.bounds.x_range || [0, 1];
        this.yRange = this.data.bounds.y_range || [0, 1];
        this.zRange = this.data.bounds.z_range || [0, 1];

        // Agent colors
        this.agentColors = this.data.agent_colors || {};

        // Organize tracks by time and agent for efficient lookup
        this.tracksByTime = this._organizeTracks();

        // Calculate scene parameters
        this.sceneCenter = new THREE.Vector3(
            (this.xRange[0] + this.xRange[1]) / 2,
            (this.yRange[0] + this.yRange[1]) / 2,
            (this.zRange[0] + this.zRange[1]) / 2
        );
        this.sceneSize = Math.max(
            this.xRange[1] - this.xRange[0],
            this.yRange[1] - this.yRange[0],
            this.zRange[1] - this.zRange[0]
        );

        // Calculate appropriate sphere size (0.5% of scene size)
        this.sphereSize = this.sceneSize * 0.005;

        // Animation state
        this.animationId = null;

        // Initialize
        this.init();
    }

    init() {
        this.setupScene();
        this.setupCamera();
        this.setupRenderer();
        this.setupLights();
        this.setupHelpers();
        this.initializeAgents();
        this.setupFrameListener();
        this.startAnimation();

        console.log('✅ Episode Animation Viewer 3D ready');
        console.log(`   Canvas mode: ${this.isCanvasElement ? 'Using existing <canvas>' : 'Created new canvas in container'}`);
        console.log(`   Tracks: ${this.data.tracks.length} points`);
        console.log(`   Time range: ${this.data.time_range[0]} - ${this.data.time_range[1]}`);
        console.log(`   Bounds: X[${this.xRange[0]}, ${this.xRange[1]}], Y[${this.yRange[0]}, ${this.yRange[1]}], Z[${this.zRange[0]}, ${this.zRange[1]}]`);
        console.log(`   Scene size: ${this.sceneSize.toFixed(2)}, Sphere size: ${this.sphereSize.toFixed(4)}`);
    }

    setupScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a1a);
    }

    setupCamera() {
        const rect = this.canvasElement.getBoundingClientRect();
        this.camera = new THREE.PerspectiveCamera(
            60,
            rect.width / rect.height,
            this.sceneSize * 0.001,
            this.sceneSize * 10
        );

        // Position camera to view the entire scene
        const distance = this.sceneSize * 1.5;
        this.camera.position.set(
            this.sceneCenter.x + distance * 0.5,
            this.sceneCenter.y + distance * 0.5,
            this.sceneCenter.z + distance * 0.5
        );
        this.camera.lookAt(this.sceneCenter);
    }

    setupRenderer() {
        const rect = this.canvasElement.getBoundingClientRect();

        // WebGL renderer - use existing canvas if available
        if (this.isCanvasElement) {
            // Use the existing canvas element
            this.renderer = new THREE.WebGLRenderer({
                canvas: this.canvasElement,
                antialias: true
            });
        } else {
            // Create new canvas and append to container
            this.renderer = new THREE.WebGLRenderer({ antialias: true });
            this.canvasElement.appendChild(this.renderer.domElement);
        }

        this.renderer.setSize(rect.width, rect.height);
        this.renderer.setPixelRatio(window.devicePixelRatio);

        // CSS2D renderer for labels
        this.labelRenderer = new CSS2DRenderer();
        this.labelRenderer.setSize(rect.width, rect.height);
        this.labelRenderer.domElement.style.position = 'absolute';
        this.labelRenderer.domElement.style.top = '0px';
        this.labelRenderer.domElement.style.left = '0px';
        this.labelRenderer.domElement.style.pointerEvents = 'none';
        this.labelRenderer.domElement.style.width = '100%';
        this.labelRenderer.domElement.style.height = '100%';

        // Append label renderer to parent (works for both canvas and container)
        const parent = this.isCanvasElement ? this.canvasElement.parentElement : this.canvasElement;
        if (parent) {
            // Ensure parent has relative positioning for absolute child
            if (parent.style.position !== 'absolute' && parent.style.position !== 'relative') {
                parent.style.position = 'relative';
            }
            parent.appendChild(this.labelRenderer.domElement);
        }

        // OrbitControls
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.target.copy(this.sceneCenter);
        this.controls.update();

        // Handle window resize
        this.resizeHandler = () => this.onWindowResize();
        window.addEventListener('resize', this.resizeHandler);
    }

    setupLights() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);

        // Directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(
            this.sceneCenter.x + this.sceneSize,
            this.sceneCenter.y + this.sceneSize,
            this.sceneCenter.z + this.sceneSize
        );
        this.scene.add(directionalLight);
    }

    setupHelpers() {
        // Grid helper on XY plane
        const gridSize = this.sceneSize * 1.2;
        const gridDivisions = 20;
        const gridHelper = new THREE.GridHelper(gridSize, gridDivisions, 0x444444, 0x222222);
        gridHelper.position.copy(this.sceneCenter);
        // Rotate to XY plane (grid is normally on XZ)
        gridHelper.rotation.x = Math.PI / 2;
        this.scene.add(gridHelper);

        // Axes helper
        const axesHelper = new THREE.AxesHelper(this.sceneSize * 0.2);
        axesHelper.position.copy(this.sceneCenter);
        this.scene.add(axesHelper);
    }

    _organizeTracks() {
        /**
         * Organize tracks by time_index for fast lookup
         * Returns: {time_index: {agent_id: {x, y, z, speed, ...}, ...}, ...}
         */
        const tracksByTime = {};

        for (const track of this.data.tracks) {
            const time = track.time_index;
            if (!tracksByTime[time]) {
                tracksByTime[time] = {};
            }
            tracksByTime[time][track.agent_id] = track;
        }

        return tracksByTime;
    }

    initializeAgents() {
        /**
         * Create spheres and labels for all unique agents
         */
        // Extract all unique agent IDs
        const uniqueAgentIds = new Set();
        for (const track of this.data.tracks) {
            uniqueAgentIds.add(track.agent_id);
        }

        console.log(`Creating ${uniqueAgentIds.size} agent spheres`);

        // Shared geometry for efficiency
        const sphereGeometry = new THREE.SphereGeometry(this.sphereSize, 16, 12);

        for (const agentId of uniqueAgentIds) {
            const colorHex = this.agentColors[agentId] || '#888888';
            const color = new THREE.Color(colorHex);

            // Create sphere
            const material = new THREE.MeshPhongMaterial({
                color: color,
                emissive: color,
                emissiveIntensity: 0.3,
                shininess: 100
            });

            const sphere = new THREE.Mesh(sphereGeometry, material);
            sphere.castShadow = true;
            sphere.visible = false;
            this.spheres[agentId] = sphere;
            this.scene.add(sphere);

            // Create label
            if (this.showIds) {
                const labelDiv = document.createElement('div');
                labelDiv.className = 'agent-label';
                labelDiv.textContent = agentId.toString();
                labelDiv.style.color = colorHex;
                labelDiv.style.fontFamily = 'Arial, sans-serif';
                labelDiv.style.fontSize = '12px';
                labelDiv.style.fontWeight = 'bold';
                labelDiv.style.textShadow = '1px 1px 2px rgba(0,0,0,0.8)';
                labelDiv.style.pointerEvents = 'none';
                labelDiv.style.userSelect = 'none';

                const label = new CSS2DObject(labelDiv);
                label.visible = this.showIds;
                this.labels[agentId] = label;
                this.scene.add(label);
            }
        }
    }

    setupFrameListener() {
        // Register callback for frame changes
        this.frameManager.onFrameChange((frame) => {
            this.updateTrailHistory(frame);
            this.updateFrame(frame);
        });
    }

    updateTrailHistory(frame) {
        /**
         * Update trail history for the current frame
         * Maintains a sliding window of trail_length frames
         */
        const currentTime = this.data.time_range[0] + frame;
        const startTime = Math.max(this.data.time_range[0], currentTime - this.trailLength);

        // Clear trail history
        this.trailPositions = {};

        // Build trail history from startTime to currentTime
        for (let t = startTime; t <= currentTime; t++) {
            const tracksAtTime = this.tracksByTime[t];
            if (!tracksAtTime) continue;

            for (const agentId in tracksAtTime) {
                const track = tracksAtTime[agentId];

                if (!this.trailPositions[agentId]) {
                    this.trailPositions[agentId] = [];
                }

                this.trailPositions[agentId].push({
                    x: track.x,
                    y: track.y,
                    z: track.z,
                    time: t
                });
            }
        }
    }

    updateFrame(frame) {
        /**
         * Update scene for the current frame
         */
        const currentTime = this.data.time_range[0] + frame;
        const tracksAtTime = this.tracksByTime[currentTime];

        // Hide all spheres and labels first
        Object.values(this.spheres).forEach(sphere => {
            if (sphere) sphere.visible = false;
        });
        Object.values(this.labels).forEach(label => {
            if (label) label.visible = false;
        });

        let visibleCount = 0;

        // Update visible tracks
        if (tracksAtTime) {
            for (const agentId in tracksAtTime) {
                const track = tracksAtTime[agentId];
                const sphere = this.spheres[agentId];

                if (sphere && track.x !== null && track.y !== null && track.z !== null) {
                    sphere.position.set(track.x, track.y, track.z);
                    sphere.visible = true;
                    visibleCount++;

                    // Update label position
                    const label = this.labels[agentId];
                    if (label && this.showIds) {
                        // Center label on the agent
                        label.position.copy(sphere.position);
                        label.visible = true;
                    }
                }
            }
        }

        // Update trails
        this.updateTrails();
    }

    updateTrails() {
        /**
         * Update trail geometries for all agents with trail data
         */
        // Remove old trails
        Object.values(this.trails).forEach(trail => {
            if (trail) {
                this.scene.remove(trail);
                trail.geometry.dispose();
                trail.material.dispose();
            }
        });
        this.trails = {};

        // Create new trails
        for (const agentId in this.trailPositions) {
            const positions = this.trailPositions[agentId];
            if (positions.length < 2) continue;

            const colorHex = this.agentColors[agentId] || '#888888';
            const color = new THREE.Color(colorHex);

            // Create curve from positions
            const points = positions.map(pos => new THREE.Vector3(pos.x, pos.y, pos.z));
            const curve = new THREE.CatmullRomCurve3(points);

            // Use TubeGeometry for visible 3D trails
            const tubeRadius = this.sphereSize / 3;
            const tubeGeometry = new THREE.TubeGeometry(curve, 32, tubeRadius, 8, false);
            const tubeMaterial = new THREE.MeshBasicMaterial({
                color: color,
                opacity: 0.6,
                transparent: true
            });

            const trail = new THREE.Mesh(tubeGeometry, tubeMaterial);
            this.trails[agentId] = trail;
            this.scene.add(trail);
        }
    }

    onWindowResize() {
        const rect = this.canvasElement.getBoundingClientRect();
        this.camera.aspect = rect.width / rect.height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(rect.width, rect.height);
        this.labelRenderer.setSize(rect.width, rect.height);
    }

    startAnimation() {
        // Start render loop
        const animate = () => {
            this.controls.update();
            this.renderer.render(this.scene, this.camera);
            this.labelRenderer.render(this.scene, this.camera);
            this.animationId = requestAnimationFrame(animate);
        };
        animate();
    }

    destroy() {
        /**
         * Clean up resources
         */
        console.log('🛑 Destroying Episode Animation Viewer 3D');

        // Stop animation
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }

        // Stop frame manager
        if (this.frameManager) {
            this.frameManager.destroy();
        }

        // Dispose geometries and materials
        Object.values(this.spheres).forEach(sphere => {
            if (sphere) {
                this.scene.remove(sphere);
                sphere.geometry.dispose();
                sphere.material.dispose();
            }
        });

        Object.values(this.trails).forEach(trail => {
            if (trail) {
                this.scene.remove(trail);
                trail.geometry.dispose();
                trail.material.dispose();
            }
        });

        Object.values(this.labels).forEach(label => {
            if (label) {
                this.scene.remove(label);
            }
        });

        // Dispose renderer
        if (this.renderer) {
            this.renderer.dispose();
            // Only remove renderer's canvas if we created it (not using existing canvas)
            if (!this.isCanvasElement && this.renderer.domElement) {
                this.renderer.domElement.remove();
            }
        }

        if (this.labelRenderer && this.labelRenderer.domElement) {
            this.labelRenderer.domElement.remove();
        }

        // Remove resize handler
        if (this.resizeHandler) {
            window.removeEventListener('resize', this.resizeHandler);
        }

        // Dispose controls
        if (this.controls) {
            this.controls.dispose();
        }

        console.log('✅ Episode Animation Viewer 3D destroyed');
    }
}
// Cache bust: 1763706684
