/**
 * Dual-Video Synchronized Viewer
 * Displays two videos side-by-side with synchronized playback and independent tracking overlays
 */

import { ApiClient } from '../utils/api_client.js';
import { TrackColors } from '../utils/track_colors.js';

export class DualVideoViewer {
    constructor() {
        // Core components
        this.api = new ApiClient();
        this.trackColors = new TrackColors(); // Shared color generator for consistent track colors

        // DOM elements - Selectors and containers
        this.video1Select = document.getElementById('video1Select');
        this.video2Select = document.getElementById('video2Select');
        this.dualVideoContent = document.getElementById('dualVideoContent');
        this.statusMessage = document.getElementById('statusMessage');

        // Video 1 elements
        this.video1Player = document.getElementById('video1Player');
        this.video1Status = document.getElementById('video1Status');
        this.canvas1 = document.getElementById('overlay1Canvas');
        this.ctx1 = this.canvas1.getContext('2d');

        // Video 2 elements
        this.video2Player = document.getElementById('video2Player');
        this.video2Status = document.getElementById('video2Status');
        this.canvas2 = document.getElementById('overlay2Canvas');
        this.ctx2 = this.canvas2.getContext('2d');

        // Playback controls
        this.playPauseBtn = document.getElementById('playPauseBtn');
        this.stepBackBtn = document.getElementById('stepBackBtn');
        this.stepForwardBtn = document.getElementById('stepForwardBtn');
        this.resetBtn = document.getElementById('resetBtn');
        this.timeSlider = document.getElementById('timeSlider');
        this.timeDisplay = document.getElementById('timeDisplay');
        this.frameDisplay = document.getElementById('frameDisplay');

        // Speed controls
        this.speedBtns = document.querySelectorAll('.speed-btn');
        this.speedDisplay = document.getElementById('speedDisplay');

        // Video 1 overlay controls
        this.showIds1 = document.getElementById('showIds1');
        this.showTrails1 = document.getElementById('showTrails1');
        this.showDebug1 = document.getElementById('showDebug1');
        this.opacity1Slider = document.getElementById('opacity1Slider');
        this.opacity1Value = document.getElementById('opacity1Value');

        // Video 2 overlay controls
        this.showIds2 = document.getElementById('showIds2');
        this.showTrails2 = document.getElementById('showTrails2');
        this.showDebug2 = document.getElementById('showDebug2');
        this.opacity2Slider = document.getElementById('opacity2Slider');
        this.opacity2Value = document.getElementById('opacity2Value');

        // State
        this.currentVideo1Id = null;
        this.currentVideo2Id = null;
        this.fps = 30; // Default FPS, can be made configurable
        this.isPlaying = false;
        this.currentSpeed = 1.0;

        // Video 1 data
        this.video1Data = null;
        this.bboxData1 = {};
        this.trails1 = {};
        this.metadata1 = null;

        // Video 2 data
        this.video2Data = null;
        this.bboxData2 = {};
        this.trails2 = {};
        this.metadata2 = null;

        // Synchronization state
        this.isSeeking = false;
        this.syncTolerance = 0.1; // Max time difference in seconds before forcing sync

        // Initialize
        this.init();
    }

    async init() {
        this.setupEventListeners();
        await this.refreshVideoLists();
    }

    setupEventListeners() {
        // Video selection
        this.video1Select.addEventListener('change', () => {
            if (this.video1Select.value && this.video2Select.value) {
                this.loadVideos();
            }
        });

        this.video2Select.addEventListener('change', () => {
            if (this.video1Select.value && this.video2Select.value) {
                this.loadVideos();
            }
        });

        // Refresh button
        document.getElementById('refreshBtn').addEventListener('click', () => {
            this.refreshVideoLists();
        });

        // Playback controls
        this.playPauseBtn.addEventListener('click', () => this.togglePlayPause());
        this.stepBackBtn.addEventListener('click', () => this.stepBack());
        this.stepForwardBtn.addEventListener('click', () => this.stepForward());
        this.resetBtn.addEventListener('click', () => this.reset());

        // Time slider - manual seeking
        this.timeSlider.addEventListener('input', (e) => {
            const time = parseFloat(e.target.value);
            this.seekTo(time);
        });

        // Speed controls
        this.speedBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const speed = parseFloat(btn.dataset.speed);
                this.setSpeed(speed);

                // Update active button
                this.speedBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            });
        });

        // Video 1 - Master video drives synchronization
        this.video1Player.addEventListener('loadedmetadata', () => this.onVideo1MetadataLoaded());
        this.video1Player.addEventListener('timeupdate', () => this.onMasterTimeUpdate());
        this.video1Player.addEventListener('play', () => this.onMasterPlay());
        this.video1Player.addEventListener('pause', () => this.onMasterPause());

        // Video 2 - Slave video
        this.video2Player.addEventListener('loadedmetadata', () => this.onVideo2MetadataLoaded());

        // Opacity sliders
        this.opacity1Slider.addEventListener('input', (e) => {
            this.opacity1Value.textContent = e.target.value;
        });
        this.opacity2Slider.addEventListener('input', (e) => {
            this.opacity2Value.textContent = e.target.value;
        });

        // Canvas resize
        window.addEventListener('resize', () => this.resizeCanvases());
    }

    async refreshVideoLists() {
        try {
            this.showStatus('Loading videos...', 'info');
            const videos = await this.api.getVideos();

            // Clear and repopulate dropdowns
            this.video1Select.innerHTML = '<option value="">Select video 1...</option>';
            this.video2Select.innerHTML = '<option value="">Select video 2...</option>';

            videos.forEach(video => {
                const option1 = document.createElement('option');
                option1.value = video.id;
                option1.textContent = video.name;
                this.video1Select.appendChild(option1);

                const option2 = document.createElement('option');
                option2.value = video.id;
                option2.textContent = video.name;
                this.video2Select.appendChild(option2);
            });

            this.showStatus(`Found ${videos.length} video(s)`, 'success');
        } catch (error) {
            this.showStatus('Error loading videos: ' + error.message, 'error');
        }
    }

    async loadVideos() {
        const video1Id = this.video1Select.value;
        const video2Id = this.video2Select.value;

        // Validation
        if (!video1Id || !video2Id) {
            return;
        }

        if (video1Id === video2Id) {
            this.showStatus('Please select different videos', 'error');
            return;
        }

        try {
            this.showStatus('Loading videos...', 'info');

            // Load both videos in parallel
            const [video1Data, video2Data] = await Promise.all([
                this.api.getVideoData(video1Id),
                this.api.getVideoData(video2Id)
            ]);

            this.currentVideo1Id = video1Id;
            this.currentVideo2Id = video2Id;

            // Extract video data
            this.video1Data = video1Data;
            this.video2Data = video2Data;

            this.bboxData1 = this.video1Data.bbox_data || {};
            this.bboxData2 = this.video2Data.bbox_data || {};

            // Reset trails
            this.trails1 = {};
            this.trails2 = {};

            // Generate colors for all unique track IDs across both videos
            const allTrackIds = new Set();
            Object.values(this.bboxData1).forEach(frameTracks => {
                frameTracks.forEach(track => allTrackIds.add(track.track_id));
            });
            Object.values(this.bboxData2).forEach(frameTracks => {
                frameTracks.forEach(track => allTrackIds.add(track.track_id));
            });

            this.trackColors.clear();
            this.trackColors.generateForTracks(Array.from(allTrackIds));

            // Load video files
            this.video1Player.src = this.api.getVideoFileUrl(video1Id);
            this.video2Player.src = this.api.getVideoFileUrl(video2Id);

            // Update status labels
            this.video1Status.textContent = this.video1Data.name;
            this.video2Status.textContent = this.video2Data.name;

            // Show UI
            this.dualVideoContent.style.display = 'block';

            this.showStatus('Videos loaded successfully', 'success');
        } catch (error) {
            this.showStatus('Error loading videos: ' + error.message, 'error');
        }
    }

    onVideo1MetadataLoaded() {
        this.metadata1 = {
            duration: this.video1Player.duration,
            width: this.video1Player.videoWidth,
            height: this.video1Player.videoHeight
        };

        // Update slider max
        this.timeSlider.max = this.metadata1.duration;
        this.timeSlider.step = 1 / this.fps;

        this.resizeCanvas(this.video1Player, this.canvas1);
        this.updateTimeDisplay();
    }

    onVideo2MetadataLoaded() {
        this.metadata2 = {
            duration: this.video2Player.duration,
            width: this.video2Player.videoWidth,
            height: this.video2Player.videoHeight
        };

        this.resizeCanvas(this.video2Player, this.canvas2);
    }

    onMasterTimeUpdate() {
        if (this.isSeeking) return;

        // Sync slave video to master
        const timeDiff = Math.abs(this.video2Player.currentTime - this.video1Player.currentTime);
        if (timeDiff > this.syncTolerance) {
            this.video2Player.currentTime = this.video1Player.currentTime;
        }

        // Update UI
        this.updateTimeDisplay();
        this.updateFrameDisplay();

        // Draw overlays for both videos
        const currentFrame = Math.floor(this.video1Player.currentTime * this.fps);
        this.drawOverlays1(currentFrame);
        this.drawOverlays2(currentFrame);
    }

    onMasterPlay() {
        this.isPlaying = true;
        this.playPauseBtn.textContent = '⏸️ Pause';

        // Ensure slave is also playing
        if (this.video2Player.paused) {
            this.video2Player.play().catch(e => console.error('Error playing video 2:', e));
        }
    }

    onMasterPause() {
        this.isPlaying = false;
        this.playPauseBtn.textContent = '▶️ Play';

        // Pause slave video
        if (!this.video2Player.paused) {
            this.video2Player.pause();
        }
    }

    togglePlayPause() {
        if (this.video1Player.paused) {
            this.video1Player.play();
        } else {
            this.video1Player.pause();
        }
    }

    stepBack() {
        const newTime = Math.max(0, this.video1Player.currentTime - 1/this.fps);
        this.seekTo(newTime);
    }

    stepForward() {
        const newTime = Math.min(this.metadata1.duration, this.video1Player.currentTime + 1/this.fps);
        this.seekTo(newTime);
    }

    reset() {
        this.seekTo(0);
        this.trails1 = {};
        this.trails2 = {};
    }

    seekTo(time) {
        this.isSeeking = true;
        this.video1Player.currentTime = time;
        this.video2Player.currentTime = time;
        this.timeSlider.value = time;

        // Reset seeking flag after a short delay
        setTimeout(() => { this.isSeeking = false; }, 100);
    }

    setSpeed(speed) {
        this.currentSpeed = speed;
        this.video1Player.playbackRate = speed;
        this.video2Player.playbackRate = speed;
        this.speedDisplay.textContent = `${speed.toFixed(2)}x`;
    }

    updateTimeDisplay() {
        if (!this.metadata1) return;

        const current = this.formatTime(this.video1Player.currentTime);
        const total = this.formatTime(this.metadata1.duration);
        this.timeDisplay.textContent = `${current} / ${total}`;

        this.timeSlider.value = this.video1Player.currentTime;
    }

    updateFrameDisplay() {
        if (!this.metadata1) return;

        const currentFrame = Math.floor(this.video1Player.currentTime * this.fps);
        const totalFrames = Math.floor(this.metadata1.duration * this.fps);
        this.frameDisplay.textContent = `Frame: ${currentFrame} / ${totalFrames} (FPS: ${this.fps})`;
    }

    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    }

    // Drawing methods for Video 1
    drawOverlays1(frame) {
        this.ctx1.clearRect(0, 0, this.canvas1.width, this.canvas1.height);

        if (!this.bboxData1[frame]) return;

        const opacity = parseFloat(this.opacity1Slider.value);
        this.ctx1.globalAlpha = opacity;

        const displayArea = this.getVideoDisplayArea(this.metadata1, this.canvas1);
        const tracks = this.bboxData1[frame];

        tracks.forEach(track => {
            const color = this.trackColors.getColor(track.track_id);

            this.drawBoundingBox(this.ctx1, track, color, displayArea, this.metadata1);

            if (this.showTrails1.checked) {
                this.updateTrail(this.trails1, track.track_id, track, frame);
                this.drawTrail(this.ctx1, this.trails1[track.track_id], color, displayArea, this.metadata1);
            }

            if (this.showIds1.checked) {
                this.drawTrackId(this.ctx1, track, color, displayArea, this.metadata1);
            }
        });

        if (this.showDebug1.checked) {
            this.drawDebugInfo(this.ctx1, frame, tracks.length, 1);
        }

        this.ctx1.globalAlpha = 1.0;
    }

    // Drawing methods for Video 2
    drawOverlays2(frame) {
        this.ctx2.clearRect(0, 0, this.canvas2.width, this.canvas2.height);

        if (!this.bboxData2[frame]) return;

        const opacity = parseFloat(this.opacity2Slider.value);
        this.ctx2.globalAlpha = opacity;

        const displayArea = this.getVideoDisplayArea(this.metadata2, this.canvas2);
        const tracks = this.bboxData2[frame];

        tracks.forEach(track => {
            const color = this.trackColors.getColor(track.track_id);

            this.drawBoundingBox(this.ctx2, track, color, displayArea, this.metadata2);

            if (this.showTrails2.checked) {
                this.updateTrail(this.trails2, track.track_id, track, frame);
                this.drawTrail(this.ctx2, this.trails2[track.track_id], color, displayArea, this.metadata2);
            }

            if (this.showIds2.checked) {
                this.drawTrackId(this.ctx2, track, color, displayArea, this.metadata2);
            }
        });

        if (this.showDebug2.checked) {
            this.drawDebugInfo(this.ctx2, frame, tracks.length, 2);
        }

        this.ctx2.globalAlpha = 1.0;
    }

    // Shared drawing utilities
    getVideoDisplayArea(metadata, canvas) {
        const canvasAspect = canvas.width / canvas.height;
        const videoAspect = metadata.width / metadata.height;

        let displayWidth, displayHeight, offsetX, offsetY;

        if (videoAspect > canvasAspect) {
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

        return { displayWidth, displayHeight, offsetX, offsetY };
    }

    drawBoundingBox(ctx, track, color, displayArea, metadata) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;

        const { displayWidth, displayHeight, offsetX, offsetY } = displayArea;
        const scaleX = displayWidth / metadata.width;
        const scaleY = displayHeight / metadata.height;

        const x1 = track.x1 * scaleX + offsetX;
        const y1 = track.y1 * scaleY + offsetY;
        const x2 = track.x2 * scaleX + offsetX;
        const y2 = track.y2 * scaleY + offsetY;

        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }

    drawTrackId(ctx, track, color, displayArea, metadata) {
        const { displayWidth, displayHeight, offsetX, offsetY } = displayArea;
        const scaleX = displayWidth / metadata.width;
        const scaleY = displayHeight / metadata.height;

        const x = track.x1 * scaleX + offsetX;
        const y = track.y1 * scaleY + offsetY - 5;

        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'left';

        const text = `ID: ${track.track_id}`;
        const metrics = ctx.measureText(text);

        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(x - 2, y - 14, metrics.width + 4, 18);

        ctx.fillStyle = color;
        ctx.fillText(text, x, y);
    }

    updateTrail(trailsObj, trackId, track, frame) {
        if (!trailsObj[trackId]) {
            trailsObj[trackId] = [];
        }

        const centerX = (track.x1 + track.x2) / 2;
        const centerY = (track.y1 + track.y2) / 2;

        trailsObj[trackId].push({ x: centerX, y: centerY, frame });

        if (trailsObj[trackId].length > 30) {
            trailsObj[trackId].shift();
        }
    }

    drawTrail(ctx, trail, color, displayArea, metadata) {
        if (!trail || trail.length < 2) return;

        const { displayWidth, displayHeight, offsetX, offsetY } = displayArea;
        const scaleX = displayWidth / metadata.width;
        const scaleY = displayHeight / metadata.height;

        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.globalAlpha = 0.5;

        ctx.beginPath();
        ctx.moveTo(trail[0].x * scaleX + offsetX, trail[0].y * scaleY + offsetY);

        for (let i = 1; i < trail.length; i++) {
            ctx.lineTo(trail[i].x * scaleX + offsetX, trail[i].y * scaleY + offsetY);
        }

        ctx.stroke();
    }

    drawDebugInfo(ctx, frame, trackCount, videoNum) {
        ctx.globalAlpha = 1.0;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(10, 10, 200, 75);

        ctx.fillStyle = 'white';
        ctx.font = '12px monospace';
        ctx.textAlign = 'left';

        ctx.fillText(`Video ${videoNum}`, 20, 30);
        ctx.fillText(`Frame: ${frame}`, 20, 45);
        ctx.fillText(`Tracks: ${trackCount}`, 20, 60);
        ctx.fillText(`Speed: ${this.currentSpeed.toFixed(2)}x`, 20, 75);
    }

    resizeCanvas(videoPlayer, canvas) {
        const rect = videoPlayer.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
    }

    resizeCanvases() {
        if (this.metadata1) {
            this.resizeCanvas(this.video1Player, this.canvas1);
            const frame = Math.floor(this.video1Player.currentTime * this.fps);
            this.drawOverlays1(frame);
        }

        if (this.metadata2) {
            this.resizeCanvas(this.video2Player, this.canvas2);
            const frame = Math.floor(this.video2Player.currentTime * this.fps);
            this.drawOverlays2(frame);
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
