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

### Data and Configuration

- `config-local/` - Contains Google Cloud service account keys (not in git)
- `data/` - Processed and curated datasets
- `meshes/` - 3D mesh files and alignment data
- Environment variables via `.env` file for API keys

### Dashboard (`dashboard/`)

Web-based data browser for GCS buckets via rclone integration:

**Core Features:**

- **Session Discovery**: Automatically discovers matching sessions across `fieldwork_curated` and `fieldwork_processed` buckets
- **Smart File Filtering**: Only displays files that can be viewed/edited by the dashboard
- **Multi-Format Viewer**: Text (YAML/XML/JSON/MD), tables (CSV/Parquet), video (MP4/AVI/MOV/MKV)
- **File Editing**: Edit and save text-based files directly back to GCS
- **Local Caching**: Automatic file caching with cache management UI
- **Progress Indicators**: Visual feedback during file loading with cache status icons

**Development Setup:**

```bash
# Install rclone and configure
rclone config create collab-data "google cloud storage" service_account_file=/path/to/key.json

# Install with dashboard dependencies
pip install -e ".[dev]"
```

**Entry Points:**

- **Production**: `python -m collab_env.dashboard.cli`
- **Development (recommended)**: `./scripts/dev_dashboard.sh`
- **Alternative**: `python -m collab_env.dashboard.run_dashboard`

**Development with Autoreload:**

```bash
# Best experience - refreshes existing browser tab
./scripts/dev_dashboard.sh

# Or manually
cd collab_env/dashboard
panel serve dashboard_app.py --dev --show --port 5007
```

**Architecture Components:**

- `RcloneClient`: Interface to rclone for GCS operations
- `SessionManager`: Session discovery and management across buckets
- `FileContentManager`: File viewing/editing with pluggable viewer registry
- `DataDashboard`: Main Panel/HoloViz reactive UI application

### Notebook Testing

Notebooks can be excluded from CI testing by:
1. Adding to `EXCLUDED_NOTEBOOKS` list in `scripts/test_notebooks.sh`
2. Using environment guards: `smoke_test = "CI" in os.environ`