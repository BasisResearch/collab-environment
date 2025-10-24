/**
 * 2D Video Viewer with bbox overlay
 */

import { ApiClient } from '../utils/api_client.js';
import { FrameManager } from '../utils/frame_manager.js';
import { TrackColors } from '../utils/track_colors.js';

export class VideoViewer {
    constructor() {
        // Core components
        this.api = new ApiClient();
        this.frameManager = null;
        this.trackColors = new TrackColors();
        
        // DOM elements
        this.videoSelect = document.getElementById('videoSelect');
        this.videoContent = document.getElementById('videoContent');
        this.videoPlayer = document.getElementById('videoPlayer');
        this.canvas = document.getElementById('overlayCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.controlsPanel = document.getElementById('controlsPanel');
        this.statusMessage = document.getElementById('statusMessage');
        
        // Control elements
        this.playPauseBtn = document.getElementById('playPauseBtn');
        this.frameSlider = document.getElementById('frameSlider');
        this.frameDisplay = document.getElementById('frameDisplay');
        this.speedSlider = document.getElementById('speedSlider');
        this.speedDisplay = document.getElementById('speedDisplay');
        
        // Display options
        this.showIds = document.getElementById('showIds');
        this.showTrails = document.getElementById('showTrails');
        this.showDebug = document.getElementById('showDebug');
        this.opacitySlider = document.getElementById('opacitySlider');
        
        // State
        this.currentVideoId = null;
        this.videoData = null;
        this.bboxData = {};
        this.trails = {};
        this.videoMetadata = null;
        
        // Initialize
        this.init();
    }
    
    async init() {
        // Set up event listeners
        this.setupEventListeners();
        
        // Load initial video list
        await this.refreshVideoList();
    }
    
    setupEventListeners() {
        // Video selection
        this.videoSelect.addEventListener('change', (e) => {
            if (e.target.value) {
                this.loadVideo(e.target.value);
            }
        });
        
        // Refresh button
        document.getElementById('refreshBtn').addEventListener('click', () => {
            this.refreshVideoList();
        });
        
        // Playback controls
        this.playPauseBtn.addEventListener('click', () => {
            if (this.videoPlayer.paused) {
                this.videoPlayer.play();
                this.playPauseBtn.textContent = 'Pause';
            } else {
                this.videoPlayer.pause();
                this.playPauseBtn.textContent = 'Play';
            }
        });
        
        document.getElementById('stepBackBtn').addEventListener('click', () => {
            const fps = 30;
            this.videoPlayer.currentTime = Math.max(0, this.videoPlayer.currentTime - 1/fps);
        });
        
        document.getElementById('stepForwardBtn').addEventListener('click', () => {
            const fps = 30;
            this.videoPlayer.currentTime = Math.min(this.videoPlayer.duration, this.videoPlayer.currentTime + 1/fps);
        });
        
        document.getElementById('resetBtn').addEventListener('click', () => {
            this.videoPlayer.currentTime = 0;
            this.trails = {};
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
        
        // Canvas resize
        window.addEventListener('resize', () => this.resizeCanvas());
        
        // Video metadata loaded
        this.videoPlayer.addEventListener('loadedmetadata', () => {
            this.onVideoMetadataLoaded();
        });
        
        // Video time update - sync overlays to video
        this.videoPlayer.addEventListener('timeupdate', () => {
            this.onVideoTimeUpdate();
        });
    }
    
    async refreshVideoList() {
        try {
            this.showStatus('Loading video list...', 'info');
            const videos = await this.api.getVideos();
            
            // Clear and repopulate dropdown
            this.videoSelect.innerHTML = '<option value="">Select a video...</option>';
            
            videos.forEach(video => {
                const option = document.createElement('option');
                option.value = video.id;
                option.textContent = video.name + (video.has_bboxes ? ' (with tracks)' : '');
                this.videoSelect.appendChild(option);
            });
            
            this.showStatus(`Found ${videos.length} video(s)`, 'success');
        } catch (error) {
            this.showStatus('Error loading videos: ' + error.message, 'error');
        }
    }
    
    async loadVideo(videoId) {
        try {
            this.showStatus('Loading video data...', 'info');
            
            // Get video data from server
            this.videoData = await this.api.getVideoData(videoId);
            this.currentVideoId = videoId;
            this.bboxData = this.videoData.bbox_data || {};
            
            // Reset trails
            this.trails = {};
            
            // Generate colors for all track IDs
            const allTrackIds = new Set();
            Object.values(this.bboxData).forEach(frameTracks => {
                frameTracks.forEach(track => allTrackIds.add(track.track_id));
            });
            this.trackColors.clear();
            this.trackColors.generateForTracks(Array.from(allTrackIds));
            
            // Load video file
            this.videoPlayer.src = this.api.getVideoFileUrl(videoId);
            
            // Show UI elements
            this.videoContent.style.display = 'block';
            this.controlsPanel.style.display = 'block';
            
            this.showStatus('Video loaded', 'success');
        } catch (error) {
            this.showStatus('Error loading video: ' + error.message, 'error');
        }
    }
    
    onVideoMetadataLoaded() {
        // Store video metadata
        this.videoMetadata = {
            duration: this.videoPlayer.duration,
            width: this.videoPlayer.videoWidth,
            height: this.videoPlayer.videoHeight
        };
        
        // Calculate total frames (assuming 30 fps, can be made configurable)
        const fps = 30;
        const totalFrames = Math.floor(this.videoMetadata.duration * fps);
        
        // Update UI
        this.frameSlider.max = totalFrames;
        this.frameDisplay.textContent = `0 / ${totalFrames}`;
        
        // Resize canvas to match video
        this.resizeCanvas();
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
        
        // Draw overlays for current frame
        this.drawOverlays(currentFrame);
    }
    
    drawOverlays(frame) {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        if (!this.bboxData[frame]) return;
        
        // Set opacity
        this.ctx.globalAlpha = parseFloat(this.opacitySlider.value);
        
        // Calculate video display area once per frame
        this.currentDisplayArea = this.getVideoDisplayArea();
        
        // Get tracks for this frame
        const tracks = this.bboxData[frame];
        
        // Draw each track
        tracks.forEach(track => {
            const color = this.trackColors.getColor(track.track_id);
            
            // Draw bounding box
            this.drawBoundingBox(track, color);
            
            // Update trails
            if (this.showTrails.checked) {
                this.updateTrail(track.track_id, track, frame);
                this.drawTrail(track.track_id);
            }
            
            // Draw track ID
            if (this.showIds.checked) {
                this.drawTrackId(track, color);
            }
        });
        
        // Draw debug info
        if (this.showDebug.checked) {
            this.drawDebugInfo(frame, tracks.length);
        }
    }
    
    getVideoDisplayArea() {
        // Calculate the actual video display area within the canvas
        // considering object-fit: contain which maintains aspect ratio
        const canvasAspect = this.canvas.width / this.canvas.height;
        const videoAspect = this.videoMetadata.width / this.videoMetadata.height;
        
        let displayWidth, displayHeight, offsetX, offsetY;
        
        if (videoAspect > canvasAspect) {
            // Video is wider - fit to canvas width
            displayWidth = this.canvas.width;
            displayHeight = this.canvas.width / videoAspect;
            offsetX = 0;
            offsetY = (this.canvas.height - displayHeight) / 2;
        } else {
            // Video is taller - fit to canvas height
            displayWidth = this.canvas.height * videoAspect;
            displayHeight = this.canvas.height;
            offsetX = (this.canvas.width - displayWidth) / 2;
            offsetY = 0;
        }
        
        return { displayWidth, displayHeight, offsetX, offsetY };
    }
    
    drawBoundingBox(track, color) {
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 2;
        
        // Use cached video display area
        const { displayWidth, displayHeight, offsetX, offsetY } = this.currentDisplayArea;
        
        // Scale coordinates to actual video display area
        const scaleX = displayWidth / this.videoMetadata.width;
        const scaleY = displayHeight / this.videoMetadata.height;
        
        const x1 = track.x1 * scaleX + offsetX;
        const y1 = track.y1 * scaleY + offsetY;
        const x2 = track.x2 * scaleX + offsetX;
        const y2 = track.y2 * scaleY + offsetY;
        
        this.ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }
    
    drawTrackId(track, color) {
        this.ctx.fillStyle = color;
        this.ctx.font = 'bold 14px Arial';
        this.ctx.textAlign = 'left';
        
        // Use cached video display area
        const { displayWidth, displayHeight, offsetX, offsetY } = this.currentDisplayArea;
        
        // Scale coordinates to actual video display area
        const scaleX = displayWidth / this.videoMetadata.width;
        const scaleY = displayHeight / this.videoMetadata.height;
        
        const x = track.x1 * scaleX + offsetX;
        const y = track.y1 * scaleY + offsetY - 5;
        
        // Draw background for better visibility
        const text = `ID: ${track.track_id}`;
        const metrics = this.ctx.measureText(text);
        
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        this.ctx.fillRect(x - 2, y - 14, metrics.width + 4, 18);
        
        this.ctx.fillStyle = color;
        this.ctx.fillText(text, x, y);
    }
    
    updateTrail(trackId, track, frame) {
        if (!this.trails[trackId]) {
            this.trails[trackId] = [];
        }
        
        // Calculate center point in original video coordinates
        const centerX = (track.x1 + track.x2) / 2;
        const centerY = (track.y1 + track.y2) / 2;
        
        this.trails[trackId].push({ x: centerX, y: centerY, frame });
        
        // Keep only last 30 positions
        if (this.trails[trackId].length > 30) {
            this.trails[trackId].shift();
        }
    }
    
    drawTrail(trackId) {
        const trail = this.trails[trackId];
        if (!trail || trail.length < 2) return;
        
        const color = this.trackColors.getColor(trackId);
        
        // Use cached video display area
        const { displayWidth, displayHeight, offsetX, offsetY } = this.currentDisplayArea;
        
        // Scale coordinates to actual video display area
        const scaleX = displayWidth / this.videoMetadata.width;
        const scaleY = displayHeight / this.videoMetadata.height;
        
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 1;
        this.ctx.globalAlpha = 0.5;
        
        this.ctx.beginPath();
        this.ctx.moveTo(trail[0].x * scaleX + offsetX, trail[0].y * scaleY + offsetY);
        
        for (let i = 1; i < trail.length; i++) {
            this.ctx.lineTo(trail[i].x * scaleX + offsetX, trail[i].y * scaleY + offsetY);
        }
        
        this.ctx.stroke();
        this.ctx.globalAlpha = parseFloat(this.opacitySlider.value);
    }
    
    drawDebugInfo(frame, trackCount) {
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        this.ctx.fillRect(10, 10, 200, 60);
        
        this.ctx.fillStyle = 'white';
        this.ctx.font = '12px monospace';
        this.ctx.textAlign = 'left';
        
        this.ctx.fillText(`Frame: ${frame}`, 20, 30);
        this.ctx.fillText(`Tracks: ${trackCount}`, 20, 45);
        this.ctx.fillText(`FPS: ${(1000 / 33).toFixed(1)}`, 20, 60);
    }
    
    resizeCanvas() {
        // Match canvas size to video display size
        const rect = this.videoPlayer.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
        
        // Redraw current frame
        if (this.videoMetadata) {
            const fps = 30;
            const currentFrame = Math.floor(this.videoPlayer.currentTime * fps);
            this.drawOverlays(currentFrame);
        }
    }
    
    showStatus(message, type = 'info') {
        this.statusMessage.textContent = message;
        this.statusMessage.className = `status ${type}`;
        this.statusMessage.style.display = 'block';
        
        // Auto-hide after 3 seconds for success/info
        if (type !== 'error') {
            setTimeout(() => {
                this.statusMessage.style.display = 'none';
            }, 3000);
        }
    }
}