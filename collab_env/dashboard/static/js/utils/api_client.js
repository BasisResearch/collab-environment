/**
 * API Client for communicating with the Flask server
 */

export class ApiClient {
    constructor(baseUrl = window.API_BASE_URL || '') {
        this.baseUrl = baseUrl;
    }
    
    async getVideos() {
        const response = await fetch(`${this.baseUrl}/api/videos`);
        if (!response.ok) throw new Error('Failed to fetch videos');
        return response.json();
    }
    
    async getMeshes() {
        const response = await fetch(`${this.baseUrl}/api/meshes`);
        if (!response.ok) throw new Error('Failed to fetch meshes');
        return response.json();
    }
    
    async getVideoData(videoId) {
        const response = await fetch(`${this.baseUrl}/api/video/${videoId}`);
        if (!response.ok) throw new Error('Failed to fetch video data');
        return response.json();
    }
    
    async getMeshData(meshId) {
        const response = await fetch(`${this.baseUrl}/api/mesh/${meshId}`);
        if (!response.ok) throw new Error('Failed to fetch mesh data');
        return response.json();
    }
    
    async addVideo(videoId, videoPath, csvPath = null) {
        const response = await fetch(`${this.baseUrl}/api/add_video`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                video_id: videoId,
                video_path: videoPath,
                csv_path: csvPath
            })
        });
        
        if (!response.ok) throw new Error('Failed to add video');
        return response.json();
    }
    
    async addMesh(meshId, meshPath, csv3dPath, cameraParamsPath = null) {
        const response = await fetch(`${this.baseUrl}/api/add_mesh`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                mesh_id: meshId,
                mesh_path: meshPath,
                csv_3d_path: csv3dPath,
                camera_params_path: cameraParamsPath
            })
        });

        if (!response.ok) throw new Error('Failed to add mesh');
        return response.json();
    }

    async clearAll() {
        const response = await fetch(`${this.baseUrl}/api/clear`, {
            method: 'POST'
        });
        
        if (!response.ok) throw new Error('Failed to clear data');
        return response.json();
    }
    
    getVideoFileUrl(videoId) {
        return `${this.baseUrl}/api/video/${videoId}/file`;
    }
    
    getMeshFileUrl(meshId) {
        return `${this.baseUrl}/api/mesh/${meshId}/file`;
    }
}