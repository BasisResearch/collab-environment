# Product Requirements Document: Simulation Viewer Mode

## Executive Summary

Add a new standalone simulation viewer mode to the existing dashboard infrastructure for visualizing boid simulation outputs from `collab_env/sim/boids`. This viewer will enable playback of 3D agent tracks from parquet files alongside the scene mesh and target submesh visualization.

## Problem Statement

Current dashboard can view PLY meshes and 3D tracks from CSV files, but cannot handle:
- Multiple parquet episode files in simulation folders
- Simultaneous visualization of scene mesh and target submesh
- Agent-specific simulation parameters from config.yaml
- Batch episode selection and playback
- Local filesystem-based simulation data access

## Solution Overview

Create a new simulation viewing mode that:
1. Adds simulation-specific routes to existing `persistent_video_server.py` (not subclassing)
2. Extends mesh viewer components for simulation-specific features
3. Adds parquet data loader and episode management
4. Integrates config.yaml parsing with proper path resolution
5. Works exclusively with local filesystem (no GCS integration)

## Data Structure

### Input Directory Structure
```
simulated_data/hackathon/
├── hackathon-boid-small-200-align-cohesion_sim_run-started-*/
│   ├── config.yaml
│   ├── episode-0-completed-*.parquet
│   ├── episode-1-completed-*.parquet
│   └── ...
```

### Parquet File Schema
- `id`: Agent ID
- `type`: Agent type
- `time`: Frame number
- `x`, `y`, `z`: Position coordinates
- `v_x`, `v_y`, `v_z`: Velocity components
- Additional tracking metrics

### Config.yaml Key Fields

```yaml
meshes:
  mesh_scene: 'meshes/Open3dTSDFfusion_mesh.ply'  # Path relative to project root
  scene_angle: [-90.0, 0.0, 0.0]
  sub_mesh_target: ['meshes/labeled_meshes/query-tree_top-cluster.ply']  # Relative path

simulator:
  num_agents: 30
  num_frames: 3000
  num_episodes: 10
```

## Dashboard Integration Strategy

### Data Mode Separation

The dashboard will support three mutually exclusive data modes:

1. **GCS Mode (Default)**: Access `fieldwork_curated` and `fieldwork_processed` buckets via rclone
2. **Local Filesystem Mode**: Browse local directories with rclone local backend
3. **Simulation Mode**: Specialized mode for boid simulation data (local only)

Mode selection will be at startup via CLI arguments:
```bash
# GCS Mode (default)
python -m collab_env.dashboard.cli

# Local filesystem mode
python -m collab_env.dashboard.cli --mode local --data-dir /path/to/data

# Simulation mode
python -m collab_env.dashboard.cli --mode simulation --data-dir simulated_data/hackathon
```

### Local Filesystem Support

Rclone transparently supports local filesystem as a remote:
- Direct path usage: `/path/to/data`
- Or configured remote: `[local] type = local`
- Maintains compatibility with existing caching mechanisms
- No code changes needed for RcloneClient

### Path Resolution Strategy

All mesh paths in config.yaml are relative to project root and must be resolved:

```python
from collab_env.data.file_utils import expand_path, get_project_root

path = expand_path(rel_path, get_project_root())
```

## Architecture Design

### Component Refactoring

#### 1. Server Architecture
**Modified File:** `collab_env/dashboard/persistent_video_server.py`
- Add simulation mode flag and routes (not subclassing)
- New simulation-specific routes:
  - `/api/simulations` - List available simulation folders
  - `/api/simulation/<sim_id>/episodes` - List episodes
  - `/api/simulation/<sim_id>/config` - Get config data
  - `/api/simulation/<sim_id>/episode/<episode_id>` - Get episode data
- Conditional route registration based on mode

#### 2. Data Loading
**New File:** `collab_env/dashboard/utils/simulation_loader.py`
- `SimulationDataLoader` class:
  - Parse config.yaml
  - Load parquet files efficiently
  - Convert parquet to track format compatible with existing viewer
  - Cache loaded episodes

#### 3. Frontend Viewer
**Modified File:** `collab_env/dashboard/static/js/viewers/mesh_viewer.js`
**New File:** `collab_env/dashboard/static/js/viewers/simulation_viewer.js`
- Extends `MeshViewer` class
- Adds:
  - Episode selector dropdown
  - Multi-mesh loading (scene + target)
  - Agent variant visualization
  - Simulation-specific controls

#### 4. HTML Templates
**New File:** `collab_env/dashboard/templates/simulation_viewer.html`
- Based on `mesh_viewer.html`
- Adds:
  - Simulation folder selector
  - Episode selector
  - Config parameter display panel
  - Local path display for loaded meshes

### Shared Components (Refactored)

#### 1. Track Management
**Refactored:** Extract common track handling to `collab_env/dashboard/utils/track_manager.py`
- Unified interface for CSV and Parquet tracks
- Common track interpolation and frame management
- Shared color mapping for agents

#### 2. Mesh Loading
**Refactored:** Extract PLY loading to `collab_env/dashboard/utils/mesh_loader.py`
- Support multiple simultaneous meshes
- Mesh transformation (rotation/scale) from config
- Shared between video and simulation viewers

## Implementation Plan

### Phase 1: Infrastructure Refactoring
1. Add mode parameter to `persistent_video_server.py`
2. Extract common track/mesh handling utilities
3. Implement path resolution helpers for project-relative paths

### Phase 2: Simulation Server Routes
1. Add conditional simulation routes to `persistent_video_server.py`
2. Implement parquet data loader with proper caching
3. Config.yaml parser with mesh path resolution

### Phase 3: Frontend Implementation
1. Create simulation viewer JavaScript module
2. Add episode management UI
3. Implement multi-mesh rendering with proper path handling

### Phase 4: Integration
1. Update CLI to support mode selection
2. Integrate local filesystem support via rclone
3. Add development script for simulation viewer
4. Update dashboard app to handle mode switching

## API Specification

### Launch Server

```python
# Launch persistent server in simulation mode
python -m collab_env.dashboard.persistent_video_server --mode simulation --data-dir simulated_data/hackathon --port 5051

# Or via main dashboard with simulation mode
python -m collab_env.dashboard.cli --mode simulation --data-dir simulated_data/hackathon
```

### REST Endpoints

#### POST /api/add_simulation
Register a simulation folder with the server (called from dashboard):
```json
{
  "simulation_id": "sim_001",
  "folder_path": "simulated_data/hackathon/hackathon-boid-small-200-align-cohesion_sim_run-started-20250926-214330",
  "config_path": "simulated_data/hackathon/.../config.yaml"
}
```

#### GET /api/simulations
List registered simulations:
```json
[
  {
    "id": "sim_001",
    "name": "Boid Small 200 Align+Cohesion",
    "folder_path": "simulated_data/hackathon/hackathon-boid-small-200-align-cohesion_sim_run-started-20250926-214330",
    "num_episodes": 10,
    "num_agents": 30,
    "num_frames": 3000,
    "mesh_status": {
      "scene_found": true,
      "target_found": true
    }
  }
]
```

#### GET /api/simulation/{sim_id}/episodes
List available episodes for a simulation:
```json
[
  {"id": 0, "name": "Episode 0", "filename": "episode-0-completed-20250926-214410.parquet"},
  {"id": 1, "name": "Episode 1", "filename": "episode-1-completed-20250926-214449.parquet"}
]
```

#### GET /api/simulation/{sim_id}/episode/{episode_id}
Get episode track data:
```json
{
  "frames": {
    "0": [
      {"track_id": 1, "x": 750.0, "y": 200.0, "z": 650.0, "type": "agent"},
      {"track_id": 2, "x": 760.0, "y": 210.0, "z": 660.0, "type": "agent"}
    ]
  },
  "config": {
    "num_agents": 30,
    "num_frames": 3000,
    "meshes": {
      "scene_path": "/api/simulation/{sim_id}/mesh/scene",
      "target_path": "/api/simulation/{sim_id}/mesh/target"
    }
  }
}
```

## Dashboard Mode Configuration

### CLI Arguments

The dashboard will support mode selection via CLI:

```python
# Dashboard app configuration
parser.add_argument(
    "--mode",
    choices=["gcs", "local", "simulation"],
    default="gcs",
    help="Data access mode"
)
parser.add_argument(
    "--data-dir",
    help="Local directory path (required for local/simulation modes)"
)
```

### Mode-Specific UI Behavior

- **GCS Mode**: Shows session browser for curated/processed buckets
- **Local Mode**: Shows filesystem browser with caching support
- **Simulation Mode**: Shows simulation folder selector with episode list

## Dashboard-Server Communication

### Trigger Mechanism

Similar to the current video/mesh viewing workflow:

1. **User Action**: User browses to a simulation folder and clicks on `config.yaml`
2. **Dashboard Request**: Dashboard sends simulation folder path to persistent server via API
3. **Server Registration**: Server adds simulation to its memory with unique ID
4. **Launch Viewer**: Button appears "View Simulation" → Opens viewer with simulation ID
5. **Episode Selection**: Viewer UI shows two dropdowns populated from server data

### API Flow

```javascript
// 1. Dashboard detects config.yaml click
if (filename === 'config.yaml' && mode === 'simulation') {
    // 2. Register simulation with server
    response = await fetch('/api/add_simulation', {
        method: 'POST',
        body: JSON.stringify({
            simulation_id: generateId(),
            folder_path: currentFolderPath,  // e.g., simulated_data/hackathon/hackathon-boid-*/
            config_path: configPath
        })
    });

    // 3. Show "View Simulation" button
    showSimulationViewerButton(simulation_id);
}

// 4. When button clicked, open viewer
function openSimulationViewer(simulation_id) {
    window.open(`/simulation?id=${simulation_id}`);
}
```

### Server State Management

The persistent server maintains:
- Dictionary of registered simulations (id → folder_path)
- Parsed config data per simulation
- Episode list per simulation (discovered from .parquet files)
- Currently loaded episode data

## User Interface

### Main View Components

1. **Simulation Control Panel** (top/left)
   - **Simulation Dropdown**: Select from registered simulations (shows folder name)
   - **Episode Dropdown**: Select specific episode (e.g., "Episode 0", "Episode 1")
   - **Load Button**: Load selected episode data
   - Config parameter display (num_agents, num_frames, weights)
   - Mesh path status indicators (✓ scene mesh found, ✓ target mesh found)

2. **3D Viewport** (center)
   - Scene mesh (semi-transparent gray)
   - Target submesh (highlighted green)
   - Agent spheres with trails (color-coded by ID)
   - Camera controls (orbit/zoom/pan)
   - Grid and axes helpers

3. **Playback Controls** (bottom)
   - Play/Pause/Reset buttons
   - Frame slider with current/total display
   - Speed control (0.1x - 3.0x)
   - Display toggles (mesh/target/trails/IDs/grid)

## Success Criteria
1. Can load and display simulations from hackathon folder
2. Smooth playback of 3000+ frame episodes with 30+ agents
3. Clear visualization of scene mesh and target submesh
4. No code duplication - maximum reuse of existing components
5. Maintains compatibility with existing dashboard features

## Future Enhancements
- Side-by-side episode comparison
- Statistical analysis overlays
