/**
 * Frame Manager for controlling playback and synchronization
 */

export class FrameManager {
    constructor(maxFrame = 0) {
        this.currentFrame = 0;
        this.maxFrame = maxFrame;
        this.isPlaying = false;
        this.playbackSpeed = 1.0;
        this.fps = 30;
        
        // Registered listeners for frame changes
        this.listeners = [];
        
        // Animation state
        this.lastAnimationTime = 0;
        this.animationId = null;
    }
    
    /**
     * Register a callback for frame changes
     */
    onFrameChange(callback) {
        this.listeners.push(callback);
        // Immediately call with current frame
        callback(this.currentFrame);
    }
    
    /**
     * Remove a frame change listener
     */
    removeListener(callback) {
        const index = this.listeners.indexOf(callback);
        if (index > -1) {
            this.listeners.splice(index, 1);
        }
    }
    
    /**
     * Set the current frame
     */
    setFrame(frame) {
        frame = Math.max(0, Math.min(frame, this.maxFrame));
        if (frame !== this.currentFrame) {
            this.currentFrame = frame;
            this.notifyListeners();
        }
    }
    
    /**
     * Set the maximum frame
     */
    setMaxFrame(maxFrame) {
        this.maxFrame = maxFrame;
        if (this.currentFrame > maxFrame) {
            this.setFrame(maxFrame);
        }
    }
    
    /**
     * Start playback
     */
    play() {
        if (!this.isPlaying) {
            this.isPlaying = true;
            this.lastAnimationTime = performance.now();
            this.animate();
        }
    }
    
    /**
     * Pause playback
     */
    pause() {
        this.isPlaying = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }
    
    /**
     * Toggle play/pause
     */
    togglePlayPause() {
        if (this.isPlaying) {
            this.pause();
        } else {
            this.play();
        }
    }
    
    /**
     * Step forward one frame
     */
    stepForward() {
        this.pause();
        this.setFrame(this.currentFrame + 1);
    }
    
    /**
     * Step backward one frame
     */
    stepBackward() {
        this.pause();
        this.setFrame(this.currentFrame - 1);
    }
    
    /**
     * Jump to beginning
     */
    jumpToStart() {
        this.setFrame(0);
    }
    
    /**
     * Jump to end
     */
    jumpToEnd() {
        this.setFrame(this.maxFrame);
    }
    
    /**
     * Set playback speed (0.25x to 4x)
     */
    setSpeed(speed) {
        this.playbackSpeed = Math.max(0.25, Math.min(4, speed));
    }
    
    /**
     * Get current state
     */
    getState() {
        return {
            currentFrame: this.currentFrame,
            maxFrame: this.maxFrame,
            isPlaying: this.isPlaying,
            playbackSpeed: this.playbackSpeed,
            progress: this.maxFrame > 0 ? this.currentFrame / this.maxFrame : 0
        };
    }
    
    /**
     * Private: Notify all listeners of frame change
     */
    notifyListeners() {
        this.listeners.forEach(callback => {
            try {
                callback(this.currentFrame);
            } catch (error) {
                console.error('Error in frame listener:', error);
            }
        });
    }
    
    /**
     * Private: Animation loop
     */
    animate() {
        if (!this.isPlaying) return;
        
        const currentTime = performance.now();
        const deltaTime = currentTime - this.lastAnimationTime;
        
        // Target frame interval based on FPS and speed
        const frameInterval = 1000 / (this.fps * this.playbackSpeed);
        
        if (deltaTime >= frameInterval) {
            // Advance frame
            this.currentFrame++;
            
            // Loop or stop at end
            if (this.currentFrame > this.maxFrame) {
                this.currentFrame = 0; // Loop
                // Or stop: this.pause(); return;
            }
            
            this.notifyListeners();
            this.lastAnimationTime = currentTime - (deltaTime % frameInterval);
        }
        
        this.animationId = requestAnimationFrame(() => this.animate());
    }
    
    /**
     * Destroy the frame manager
     */
    destroy() {
        this.pause();
        this.listeners = [];
    }
}