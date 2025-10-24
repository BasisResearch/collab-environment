/**
 * Track color management utilities
 */

export class TrackColors {
    constructor() {
        this.colorMap = new Map();
        this.colorIndex = 0;
        
        // Predefined distinct colors for tracks
        this.palette = [
            '#FF6B6B', // Red
            '#4ECDC4', // Teal
            '#45B7D1', // Blue
            '#FFA07A', // Light Salmon
            '#98D8C8', // Mint
            '#FFD93D', // Yellow
            '#6BCF7F', // Green
            '#FF90B3', // Pink
            '#B4A7D6', // Lavender
            '#F4A261', // Sandy Brown
            '#2A9D8F', // Sea Green
            '#E76F51', // Terra Cotta
            '#264653', // Charcoal
            '#A8DADC', // Powder Blue
            '#E9C46A', // Maize
        ];
    }
    
    /**
     * Get color for a track ID (consistent across calls)
     */
    getColor(trackId) {
        if (!this.colorMap.has(trackId)) {
            const color = this.palette[this.colorIndex % this.palette.length];
            this.colorMap.set(trackId, color);
            this.colorIndex++;
        }
        return this.colorMap.get(trackId);
    }
    
    /**
     * Get color as RGB values (for Three.js)
     */
    getColorRGB(trackId) {
        const hex = this.getColor(trackId);
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16) / 255,
            g: parseInt(result[2], 16) / 255,
            b: parseInt(result[3], 16) / 255
        } : { r: 1, g: 1, b: 1 };
    }
    
    /**
     * Get color as Three.js hex number
     */
    getColorHex(trackId) {
        const hex = this.getColor(trackId);
        return parseInt(hex.replace('#', '0x'), 16);
    }
    
    /**
     * Set custom color for a track
     */
    setColor(trackId, color) {
        this.colorMap.set(trackId, color);
    }
    
    /**
     * Clear all color assignments
     */
    clear() {
        this.colorMap.clear();
        this.colorIndex = 0;
    }
    
    /**
     * Get all assigned colors
     */
    getAllColors() {
        return new Map(this.colorMap);
    }
    
    /**
     * Generate colors for a list of track IDs
     */
    generateForTracks(trackIds) {
        trackIds.forEach(id => this.getColor(id));
        return this.getAllColors();
    }
}