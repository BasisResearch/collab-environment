/**
 * Episode Animation Viewer - client-side 2D animation for episode tracks
 *
 * Handles 2D visualization with smooth playback using Canvas API.
 * Data is embedded directly from Panel - no HTTP requests needed.
 */

import { FrameManager } from '../utils/frame_manager.js';

export class EpisodeAnimationViewer {
    constructor(animationData, canvasId) {
        console.log('🚀 Starting Episode Animation Viewer');

        this.data = animationData;
        this.canvasId = canvasId;

        // Validate data
        if (!this.data || !this.data.tracks || this.data.tracks.length === 0) {
            console.error('No track data available');
            return;
        }

        // Canvas and context
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) {
            console.error(`Canvas element not found: ${canvasId}`);
            return;
        }
        this.ctx = this.canvas.getContext('2d');

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

        // Agent colors
        this.agentColors = this.data.agent_colors || {};

        // Organize tracks by time and agent for efficient lookup
        this.tracksByTime = this._organizeTracks();

        // Trail history: {agent_id: [{x, y, time}, ...]}
        this.trailHistory = {};

        // Animation state
        this.animationId = null;

        // Initialize
        this.init();
    }

    init() {
        this.setupCanvas();
        this.setupFrameListener();
        this.startAnimation();

        console.log('✅ Episode Animation Viewer ready');
        console.log(`   Tracks: ${this.data.tracks.length} points`);
        console.log(`   Time range: ${this.data.time_range[0]} - ${this.data.time_range[1]}`);
        console.log(`   Bounds: X[${this.xRange[0]}, ${this.xRange[1]}], Y[${this.yRange[0]}, ${this.yRange[1]}]`);
    }

    setupCanvas() {
        // Set canvas size to match container
        const resizeCanvas = () => {
            const rect = this.canvas.getBoundingClientRect();
            this.canvas.width = rect.width;
            this.canvas.height = rect.height;
            this.render();  // Re-render after resize
        };

        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);
        this.resizeHandler = resizeCanvas;
    }

    _organizeTracks() {
        /**
         * Organize tracks by time_index for fast lookup
         * Returns: {time_index: {agent_id: {x, y, speed, ...}, ...}, ...}
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

    setupFrameListener() {
        // Register callback for frame changes
        this.frameManager.onFrameChange((frame) => {
            this.updateTrailHistory(frame);
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
        this.trailHistory = {};

        // Build trail history from startTime to currentTime
        for (let t = startTime; t <= currentTime; t++) {
            const tracksAtTime = this.tracksByTime[t];
            if (!tracksAtTime) continue;

            for (const agentId in tracksAtTime) {
                const track = tracksAtTime[agentId];

                if (!this.trailHistory[agentId]) {
                    this.trailHistory[agentId] = [];
                }

                this.trailHistory[agentId].push({
                    x: track.x,
                    y: track.y,
                    time: t
                });
            }
        }
    }

    startAnimation() {
        // Start render loop
        const animate = () => {
            this.render();
            this.animationId = requestAnimationFrame(animate);
        };
        animate();
    }

    render() {
        if (!this.ctx) return;

        // Clear canvas
        this.ctx.fillStyle = '#1a1a1a';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Get current frame and time
        const frame = this.frameManager.currentFrame;
        const currentTime = this.data.time_range[0] + frame;

        // Draw grid
        this.drawGrid();

        // Draw trails
        this.drawTrails();

        // Draw current positions
        this.drawCurrentPositions(currentTime);
    }

    drawGrid() {
        /**
         * Draw background grid for reference
         */
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        this.ctx.lineWidth = 1;

        // Vertical lines
        const numVerticalLines = 10;
        for (let i = 0; i <= numVerticalLines; i++) {
            const x = (i / numVerticalLines) * this.canvas.width;
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.canvas.height);
            this.ctx.stroke();
        }

        // Horizontal lines
        const numHorizontalLines = 10;
        for (let i = 0; i <= numHorizontalLines; i++) {
            const y = (i / numHorizontalLines) * this.canvas.height;
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.canvas.width, y);
            this.ctx.stroke();
        }
    }

    drawTrails() {
        /**
         * Draw trails for all agents
         */
        for (const agentId in this.trailHistory) {
            const trail = this.trailHistory[agentId];
            if (trail.length < 2) continue;

            const color = this.agentColors[agentId] || '#888888';

            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = 2;
            this.ctx.globalAlpha = 0.6;

            this.ctx.beginPath();
            const firstPoint = trail[0];
            const [x0, y0] = this.worldToCanvas(firstPoint.x, firstPoint.y);
            this.ctx.moveTo(x0, y0);

            for (let i = 1; i < trail.length; i++) {
                const point = trail[i];
                const [x, y] = this.worldToCanvas(point.x, point.y);
                this.ctx.lineTo(x, y);
            }

            this.ctx.stroke();
            this.ctx.globalAlpha = 1.0;
        }
    }

    drawCurrentPositions(currentTime) {
        /**
         * Draw current agent positions as colored circles
         */
        const tracksAtTime = this.tracksByTime[currentTime];
        if (!tracksAtTime) return;

        for (const agentId in tracksAtTime) {
            const track = tracksAtTime[agentId];
            const color = this.agentColors[agentId] || '#888888';

            const [x, y] = this.worldToCanvas(track.x, track.y);

            // Draw circle
            this.ctx.fillStyle = color;
            this.ctx.beginPath();
            this.ctx.arc(x, y, 8, 0, 2 * Math.PI);
            this.ctx.fill();

            // Draw border
            this.ctx.strokeStyle = '#ffffff';
            this.ctx.lineWidth = 2;
            this.ctx.stroke();

            // Draw agent ID if enabled
            if (this.showIds) {
                this.ctx.fillStyle = '#ffffff';
                this.ctx.font = 'bold 12px sans-serif';
                this.ctx.textAlign = 'center';
                this.ctx.textBaseline = 'middle';
                this.ctx.fillText(agentId.toString(), x, y);
            }
        }
    }

    worldToCanvas(x, y) {
        /**
         * Convert world coordinates to canvas coordinates
         * World coordinates are in the range [xRange, yRange]
         * Canvas coordinates are in pixels
         */
        const xNorm = (x - this.xRange[0]) / (this.xRange[1] - this.xRange[0]);
        const yNorm = (y - this.yRange[0]) / (this.yRange[1] - this.yRange[0]);

        const canvasX = xNorm * this.canvas.width;
        // Flip Y axis (canvas Y increases downward, world Y increases upward)
        const canvasY = (1 - yNorm) * this.canvas.height;

        return [canvasX, canvasY];
    }

    destroy() {
        /**
         * Clean up resources
         */
        console.log('🛑 Destroying Episode Animation Viewer');

        // Stop animation
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }

        // Stop frame manager
        if (this.frameManager) {
            this.frameManager.destroy();
        }

        // Remove resize handler
        if (this.resizeHandler) {
            window.removeEventListener('resize', this.resizeHandler);
        }

        console.log('✅ Episode Animation Viewer destroyed');
    }
}
