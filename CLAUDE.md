# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`collab-environment` is an integration package for representing, modeling, and simulating behavior within 3D environments. It combines tracking, data processing, graph neural networks (GNN), and simulation modules for collaborative research.

## Development Commands

### Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Alternative: Using conda
conda env create -n collab-env -f env.yml
conda activate collab-env
```

### Testing and Quality Assurance
```bash
# Run tests (includes linting check first)
make test

# Run linting checks only
make lint

# Format code automatically
make format

# Test notebooks execution
make test-notebooks

# Format notebooks
make format-notebooks

# Lint notebooks
make lint-notebooks
```

### Direct script execution

```bash
# Run tests directly
./scripts/test.sh        # pytest tests/ -n auto
./scripts/lint.sh        # mypy, ruff check, ruff format --diff
./scripts/clean.sh       # code formatting

# Dashboard development
./scripts/dev_dashboard.sh  # panel serve dashboard with autoreload
```

## Code Architecture

### Package Structure

- `collab_env/` - Main package with five core subprojects:
  - `tracking/` - Animal tracking and thermal processing with model inference
  - `data/` - Data utilities including GCS integration and file processing
  - `gnn/` - Graph Neural Network components for behavioral modeling
  - `sim/` - Simulation modules including boids and gymnasium environments
  - `dashboard/` - Web-based data browser for GCS buckets via rclone

### Key Components

**Simulation (`sim/`)**
- `boids/` - Multi-agent boids simulation with 3D visualization
- `gymnasium_env/` - OpenAI Gym compatible environments for RL training
- Entry point: `python -m collab_env.sim.boids.runBoidsSimple`

**Tracking (`tracking/`)**
- Animal tracking with thermal processing capabilities
- Local model inference and tracking functionality
- GUI for alignment operations

**Data Processing (`data/`)**
- Google Cloud Storage utilities for data management
- File processing utilities including EXIF data handling
- Pre-trained data stored in `simulated_data/` and `trained_models/`

**GNN (`gnn/`)**
- Graph neural network implementations for behavioral analysis
- Utility functions for graph-based modeling

### Testing Requirements

All PRs must pass comprehensive testing:
- **Unit tests**: Focus on code logic, not notebooks (`tests/` directory)
- **Notebook execution tests**: Jupyter notebooks validated for execution
- **Linting**: mypy, ruff, formatting checks

### Environment Dependencies

Key external dependencies:
- PyTorch + PyTorch Geometric for deep learning
- Open3D for 3D processing
- OpenCV for computer vision
- Gymnasium for RL environments
- Google Cloud Storage (gcsfs) for data access
- ExifTool (system dependency) for image metadata
- ffmpeg (system dependency) for dashboard video conversion

### Data and Configuration

- `config-local/` - Contains Google Cloud service account keys (not in git)
- `data/` - Processed and curated datasets
- `meshes/` - 3D mesh files and alignment data
- Environment variables via `.env` file for API keys

### Dashboard (`dashboard/`)

Web-based data browser for GCS buckets and local simulation data via rclone integration:

**Core Features:**

- **Session Discovery**: Automatically discovers matching sessions across `fieldwork_curated` and `fieldwork_processed` buckets
- **Smart File Filtering**: Only displays files that can be viewed/edited by the dashboard
- **Multi-Format Viewer**: Text (YAML/XML/JSON/MD), tables (CSV/Parquet), video (MP4/AVI/MOV/MKV), 3D meshes (PLY)
- **Video Bbox Overlay Viewer**: Interactive video player with synchronized bounding box/tracking overlays from CSV data
- **3D Mesh Viewer**: Interactive PLY file viewer with VTK-based rendering, automatic camera positioning from pickle parameters
- **3D Track Viewer**: View 3D tracking data overlaid on meshes with playback controls and camera frustum display
- **Simulation Viewer**: Specialized mode for viewing boid simulation outputs with episode playback and multi-mesh support
- **Video Conversion**: Convert incompatible videos to browser-compatible H.264 format with upload
- **File Editing**: Edit and save text-based files directly back to GCS
- **Local Caching**: Automatic file caching with cache management UI
- **Progress Indicators**: Visual feedback during file loading with cache status icons
- **Read-Only Mode**: Optional mode that disables all upload/delete operations

**Development Setup:**

```bash
# Install system dependencies
# macOS:
brew install rclone ffmpeg

# Linux (Ubuntu/Debian):
sudo apt-get install rclone ffmpeg

# Configure rclone for GCS access
rclone config create collab-data "google cloud storage" service_account_file=/path/to/key.json

# Install Python dependencies
pip install -e ".[dev]"
```

**Entry Points:**

- **Dashboard (GCS Mode)**: `python -m collab_env.dashboard.cli`
- **Dashboard (Read-Only Mode)**: `python -m collab_env.dashboard.cli --read-only`
- **Development with Autoreload**: `./scripts/dev_dashboard.sh`
- **Simulation Viewer**: `python -m collab_env.dashboard.persistent_video_server --mode simulation --data-dir simulated_data/hackathon --port 5051`

**Development with Autoreload:**

```bash
# Best experience - refreshes existing browser tab
./scripts/dev_dashboard.sh

# Or manually
cd collab_env/dashboard
panel serve dashboard_app.py --dev --show --port 5007
```

**Video Bbox Overlay Workflow:**

1. **Browse Videos**: Navigate to any MP4/AVI/MOV/MKV file in the dashboard
2. **Auto-Detection**: If `*_bboxes.csv` files exist in same directory, "View with Overlays" button appears
3. **Launch Viewer**: Click button → Persistent server starts → Browser opens to video selector
4. **Select Video**: Choose your video from dropdown → Interactive overlay viewer loads
5. **Multiple Videos**: Add more videos from dashboard → Switch between them in same viewer
6. **Interactive Controls**: Toggle track IDs, trails, opacity, debug coordinates in real-time
7. **Cleanup**: Click "Stop Server" in dashboard when done to free resources

**CSV Format Requirements:**
- Must contain `track_id` and `frame` columns
- Either bounding box format: `x1, y1, x2, y2` (pixel coordinates)
- Or centroid format: `x, y` (center point coordinates)
- File naming: `*_bboxes.csv` in same directory as video file

**Video Bbox Overlay Viewer:**

Advanced video analysis feature for viewing tracking data overlays:

- **Auto-Detection**: Automatically detects `*_bboxes.csv` files in same directory as videos
- **Smart Activation**: "View with Overlays" button appears when tracking data is available
- **Persistent Server**: Single Flask server efficiently handles multiple video/CSV combinations
- **Interactive Controls**: Toggle track IDs, movement trails, opacity, coordinate debugging
- **Multi-Format Support**: Handles both bounding box (x1,y1,x2,y2) and centroid (x,y) CSV formats
- **Dynamic Loading**: Add videos from dashboard, switch between them via dropdown selector
- **Resource Efficient**: One server process regardless of number of videos viewed
- **Clean Lifecycle**: Start/stop server management integrated with dashboard

**Simulation Viewer Mode:**

The dashboard includes a specialized simulation viewer for boid simulation outputs:

- **Episode Management**: Browse and playback multiple simulation episodes from parquet files
- **Multi-Mesh Visualization**: Simultaneous display of scene mesh and target submesh
- **Agent Tracking**: Color-coded agent spheres with movement trails over time
- **Configuration Display**: Shows simulation parameters from config.yaml
- **Path Resolution**: Automatically resolves mesh paths relative to project root
- **Local Filesystem**: Works exclusively with local simulation data (no GCS)

**Simulation Mode Usage:**

```bash
# Start persistent server in simulation mode
python -m collab_env.dashboard.persistent_video_server --mode simulation --data-dir simulated_data/hackathon --port 5051

# Then open browser to http://localhost:5051/simulation
# The server will auto-discover and register all simulations in the data directory
```

**Data Structure Requirements:**

Simulation folders should contain:
- `config.yaml` - Configuration with mesh paths (relative to project root) and simulation parameters
- `episode-*.parquet` - Episode data files with columns: id, type, time, x, y, z, v_x, v_y, v_z

**Architecture Components:**

- `RcloneClient`: Interface to rclone for GCS operations and local filesystem access
- `SessionManager`: Session discovery and management across buckets
- `FileContentManager`: File viewing/editing with pluggable viewer registry
- `DataDashboard`: Main Panel/HoloViz reactive UI application
- `PersistentVideoServer`: Flask server for video/mesh viewing and simulation mode support
- `SimulationDataLoader`: Parquet episode loader and config parser for simulation mode

### Notebook Testing

Notebooks can be excluded from CI testing by:
1. Adding to `EXCLUDED_NOTEBOOKS` list in `scripts/test_notebooks.sh`
2. Using environment guards: `smoke_test = "CI" in os.environ`