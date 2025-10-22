
.. index-inclusion-marker

collab-environment
==================

`collab-environment` is an integration package across projects for representing, modeling, and simulating behavior within 3D environments.

------------

Setup
-----

**Note:**  For alignment functionality, we strongly recommend a CUDA 11.8 compatible GPU. Using a different CUDA version will alter COLMAP results. We recommend matching the CUDA version within `collab-splats`.

* Using pip / uv:

  For installation with pip / uv, all dependencies are installed via a shell script:

   .. code:: sh

      bash setup.sh


* Using conda

   We also provide a conda `env.yml` file that can be used to create a conda environment with the necessary dependencies. Run the following commands to create the environment:

   .. code:: sh
   
      conda env create -n collab-env -f env.yml
      conda activate collab-env
      bash setup.sh


* Building the Docker image

  We provide a prebuilt Docker image as well as the associated Dockerfile. To build the image, run the following commands:

   .. code:: sh

      docker build --platform=linux/amd64  --progress=plain -t IMAGE_NAME .
      docker push IMAGE_NAME:latest

  To pull and run the image, run the following commands:

    .. code:: sh
      docker image pull tommybotch/collab-environment:latest
      docker run -it --rm -p 8888:8888 tommybotch/collab-environment:latest

* Install exiftool

  Processing the FLIR (thermal) videos requires the installation of exiftool. This can be done with the following commands:

   .. code:: sh

      # For MacOS
      brew install exiftool

      # For Linux (Ubuntu/Debian)
      sudo apt-get install libimage-exiftool-perl

      # For Linux (RHEL/CentOS/Fedora) 
      sudo yum install perl-Image-ExifTool

* Install ffmpeg (Required for Dashboard Video Conversion)

  The dashboard's video conversion feature requires ffmpeg to convert videos to browser-compatible H.264 format:

   .. code:: sh

      # For MacOS
      brew install ffmpeg

      # For Linux (Ubuntu/Debian)
      sudo apt-get install ffmpeg

      # For Linux (RHEL/CentOS/Fedora)
      sudo dnf install ffmpeg

      # For Windows
      # Download from https://ffmpeg.org/download.html
      # Or using Chocolatey: choco install ffmpeg

      # Verify installation
      ffmpeg -version


* Using gcloud

   Use of gcloud data access requires API keys stored outside this repository. Please obtain the API keys and create a ```.env``` file
   in the root directory of this repository. See below for an example:

   .. code:: sh
   
      COLLAB_DATA_KEY=/path/to/api/key.json

Dashboard
---------

A web-based dashboard for browsing and editing data files from GCS buckets using rclone integration.

**Features:**

* **Session Discovery**: Automatically discovers matching sessions across ``fieldwork_curated`` and ``fieldwork_processed`` buckets
* **Multi-Format Viewer**: Built-in viewers for text files (YAML, TXT, XML, JSON, Markdown), tabular data (CSV, Parquet), and video files (MP4, AVI, MOV, MKV)
* **Video Bbox Overlay Viewer**: Interactive video player with synchronized bounding box/tracking overlays from CSV data - automatically detects ``*_bboxes.csv`` files and provides real-time visualization controls
* **3D Mesh Viewer**: Interactive PLY file viewer with VTK-based rendering for point clouds and 3D meshes, includes automatic camera positioning from pickle parameters
* **3D Track Viewer**: View 3D tracking data overlaid on meshes with playback controls, automatic detection of ``*_3d.csv`` files, and camera parameter integration
* **Video Conversion**: Convert incompatible videos (e.g., OpenCV MPEG-4) to browser-compatible H.264 format with one-click upload
* **File Editing**: Edit and save text-based files directly back to GCS
* **Local Caching**: Automatically caches downloaded files locally for faster access with cache management UI
* **Enhanced UI**: Enlarged navigation panel with file tree, cache status icons, and progress indicators
* **Read-Only Mode**: Optional mode that disables all upload/delete operations to prevent accidental modifications to cloud storage

**Read-Only Mode:**

The dashboard supports a read-only mode for safer browsing when you want to prevent any accidental modifications to cloud storage:

* **Disabled in read-only mode**: Upload to cloud (Replace in Cloud button), Delete from cloud and cache, status shows [READ-ONLY MODE]
* **Still enabled**: All viewing operations, local file editing and caching, video conversion (local), Download Original from cloud, cache management operations
* **Usage**: Add ``--read-only`` flag to any dashboard command

**Prerequisites:**

Install `rclone <https://rclone.org/>`_ and `ffmpeg` for your platform:

.. code:: sh

    # Install rclone (see https://rclone.org/install/)
    
    # Configure for GCS access
    rclone config create collab-data "google cloud storage" service_account_file=/path/to/api/key.json
    
    # Install ffmpeg (required for video conversion feature)
    # See ffmpeg installation instructions above

**Usage:**

.. code:: sh

    # Basic usage (GCS buckets via rclone remote)
    python -m collab_env.dashboard.cli

    # Custom port and buckets
    python -m collab_env.dashboard.cli --port 8080 --curated-bucket my-curated --processed-bucket my-processed

    # Don't auto-open browser
    python -m collab_env.dashboard.cli --no-browser

    # Read-only mode (disables upload/delete functionality)
    python -m collab_env.dashboard.cli --read-only

    # Combine multiple options
    python -m collab_env.dashboard.cli --port 8080 --read-only --no-browser

    # Show all available options
    python -m collab_env.dashboard.cli --help

**Local Filesystem Mode:**

The dashboard can browse local directories using rclone's local backend:

.. code:: sh

    # First, configure an rclone local remote (one-time setup)
    rclone config create local-data local

    # Then browse local directories as if they were buckets
    python -m collab_env.dashboard.cli --remote-name local-data --curated-bucket /path/to/data/curated --processed-bucket /path/to/data/processed

    # Example with absolute paths
    python -m collab_env.dashboard.cli --remote-name local-data --curated-bucket /Users/username/research/curated --processed-bucket /Users/username/research/processed

    # Or use relative paths from current directory
    python -m collab_env.dashboard.cli --remote-name local-data --curated-bucket ./data/curated --processed-bucket ./data/processed

**Note:** In local mode, the dashboard works the same way as with GCS - it can browse, view, edit, cache, and manage files. All features (video conversion, 3D visualization, etc.) work with local files.

**Development with Autoreload:**

For the best development experience with autoreload that refreshes existing browser tabs:

.. code:: sh

    # Easy way: Use the provided script
    ./scripts/dev_dashboard.sh

    # Or manually navigate to dashboard directory
    cd collab_env/dashboard
    panel serve dashboard_app.py --dev --show --port 5007

    # With read-only mode
    panel serve dashboard_app.py --dev --show --port 5007 --args --read-only

**Dashboard Interface:**

* 💾 = File is cached locally (fast access)
* 📝 = File is cached and modified locally (needs upload)
* 📡 = File needs to be downloaded from remote
* Full file paths are displayed for precise identification
* Cache management with size and file count display
* Cache location: ``~/.cache/collab_env_dashboard/``

**Video Bbox Overlay Viewer:**

The dashboard includes an advanced video analysis feature for viewing tracking data overlays:

**Features:**

* **Auto-Detection**: Automatically detects ``*_bboxes.csv`` files in the same directory as video files
* **Smart Activation**: "View with Overlays" button appears when tracking data is available
* **Interactive Controls**: Real-time toggle for track IDs, movement trails, opacity, and coordinate debugging
* **Multi-Format Support**: Handles both bounding box (x1,y1,x2,y2) and centroid (x,y) CSV formats
* **Persistent Server**: Single Flask server efficiently manages multiple video/CSV combinations
* **Dynamic Loading**: Add videos from dashboard, switch between them via dropdown selector

**Usage Workflow:**

1. Navigate to any video file (MP4/AVI/MOV/MKV) in the dashboard
2. If ``*_bboxes.csv`` files exist in same directory → "View with Overlays" button appears
3. Click button → Persistent server starts → Browser opens to video selector
4. Select your video from dropdown → Interactive overlay viewer loads with tracking data
5. Use real-time controls to customize visualization (trails, IDs, opacity, debug info)
6. Add more videos from dashboard → Switch between them in the same viewer interface
7. Click "Stop Server" in dashboard when finished to clean up resources

**CSV Requirements:**

* Must contain ``track_id`` and ``frame`` columns
* Bounding box format: ``x1, y1, x2, y2`` (pixel coordinates of box corners)
* Or centroid format: ``x, y`` (center point coordinates)
* File naming convention: ``*_bboxes.csv`` in same directory as video file

**3D Mesh and Track Viewer:**

The dashboard includes advanced 3D visualization capabilities for PLY meshes and tracking data:

**PLY Mesh Viewer Features:**

* **Interactive 3D Rendering**: Uses PyVista and VTK for high-quality 3D visualization
* **Point Cloud and Mesh Support**: Automatically detects and renders both point clouds and surface meshes
* **Camera Parameter Integration**: Automatically loads camera parameters from ``*_mesh-aligned.pkl`` files
* **Smart Camera Positioning**: When camera params are available, positions view to match original capture perspective

**3D Track Viewer Features:**

* **Auto-Detection**: Automatically finds ``*_3d.csv`` tracking files in the same session
* **Real-Time Playback**: Frame-by-frame playback with speed controls and timeline scrubber
* **Track Visualization**: Color-coded spheres for each track with configurable size
* **Movement Trails**: Optional trail visualization showing track paths over time
* **Camera Frustum Display**: Shows the original camera position and field of view when params are available

**3D Viewer Usage:**

1. Navigate to any PLY file in the dashboard
2. The file opens in an interactive 3D viewer with mouse controls (rotate, zoom, pan)
3. If ``*_3d.csv`` files exist → "View 3D Tracks" button appears
4. Click button → Opens 3D track viewer with mesh and animated tracks
5. Use playback controls to visualize movement patterns over time
6. Toggle display options: mesh visibility, track IDs, trails, camera frustum

**Simulation Viewer Mode:**

The dashboard includes a specialized simulation viewing mode for boid simulation outputs:

**Simulation Viewer Features:**

* **Episode Management**: Browse and playback multiple simulation episodes from parquet files
* **Multi-Mesh Visualization**: Simultaneous display of scene mesh and target submesh with configurable rendering
* **Agent Tracking**: Color-coded agent spheres with movement trails visualizing agent behavior over time
* **Configuration Display**: Shows simulation parameters from config.yaml (num_agents, num_frames, weights)
* **Path Resolution**: Automatically resolves mesh paths relative to project root from config files
* **Local Filesystem**: Works exclusively with local simulation data directories (no cloud dependency)
* **Auto-Discovery**: Automatically discovers and registers all simulations in a data directory

**Simulation Mode Usage:**

.. code:: sh

    # Start persistent server in simulation mode
    python -m collab_env.dashboard.persistent_video_server --mode simulation --data-dir simulated_data/hackathon --port 5051

    # Then open browser to http://localhost:5051/simulation
    # The server will auto-discover and register all simulations in the data directory

**Simulation Data Structure:**

Simulation folders must contain:

* ``config.yaml`` - Configuration file with mesh paths (relative to project root) and simulation parameters

  * Required fields: ``meshes.mesh_scene``, ``meshes.sub_mesh_target``, ``meshes.scene_angle``
  * Simulation params: ``simulator.num_agents``, ``simulator.num_frames``, ``simulator.num_episodes``

* ``episode-*.parquet`` - Episode data files with required columns:

  * ``id``: Agent ID
  * ``type``: Agent type (e.g., 'agent' or 'env')
  * ``time``: Frame number
  * ``x``, ``y``, ``z``: 3D position coordinates
  * ``v_x``, ``v_y``, ``v_z``: Velocity components

**Simulation Viewer Workflow:**

1. Start persistent server in simulation mode with ``--data-dir`` pointing to simulation folder
2. Server auto-discovers all simulation runs containing ``config.yaml``
3. Open browser to ``http://localhost:5051/simulation``
4. Select simulation from dropdown → Choose episode → Load data
5. Use playback controls to visualize agent behavior with mesh context
6. Toggle visibility options: scene mesh, target mesh, trails, agent IDs, grid

Usage
-----

Running a Boids Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run a boids simulation:

.. code:: sh

   python -m collab_env.sim.boids.run_boids_simulator

This will start a 3D visualization of the boids simulation using the default
configuration file collab_env/sim/boids/config.yaml. To specify a
different configuration file, add a command line argument:

.. code:: sh

    python -m collab_env.sim.boids.run_boids_simulator -cf <path_to_config_file>

There are many configurable parameters. See the example configuration file for details.
If the configuration file indicates that the visualizer should be shown, the following
key commands are available while the visualizer is running:

* Q - quits the current episode. The visualizer will be terminated, and the data for the episode and the video (if specified) will be stored.

* P - saves an image of the current frame to a file. The images saved will be numbered consecutively (image-1.png, image-2.png, etc.) in the run folder, which is also specified in the config file.

* R - resets the viewer orientation to the initial orientation.

To stop the simulation prematurely when there is no visualizer window, press ``Ctrl-C`` in the terminal.


Output of the Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The output of the simulation consists of the following files, which will appear in the run folder specified in the configuration file:

* the configuration file used for the run
* a parquet file for each episode containing a pandas dataframe
* an optional video file for each episode
* optional images saved by the user hitting P while the visualizer was running

The parquet file contains a dataframe with the following columns:

* id : the id of the object
* type : currently either 'agent' for an agent or 'env' for an environment object
* timestep : the timestep for which this data applies
* x : the x position of the object (x is horizontal)
* y : the y position of the object (y is vertical)
* z : the z position of the object (z is front and back)
* v_x : the velocity the object in the x direction
* v_y : the velocity the object in the y direction
* v_z : the velocity the object in the z direction
* distance_target_center_t : the distance of the object to the center of the t-th target.
* distance_to_target_mesh_closest_point_t : the distance of the object to the closest point on the t-th target mesh.
* target_mesh_closest_point_t : the point on the t-th target mesh that is closest to the object
* mesh_scene_distance : the distance of the object to the closest point on the mesh scene
* mesh_scene_closest_point : the point on the mesh scene that is closest to the object


Contributing
------------

This is a collaborative project where contributors merge their work via Pull Requests (PRs). The project follows a structured organization to maintain code quality and facilitate collaboration.

**Project Structure:**

* ``collab_env/`` - Main package containing subprojects:

  * ``tracking/`` - Tracking-related functionality
  * ``data/`` - Data processing utilities
  * ``gnn/`` - Graph Neural Network components
  * ``sim/`` - Simulation modules

* ``docs/`` - Documentation and notebooks organized by subproject:

  * Mirrors the structure of ``collab_env/`` with corresponding subfolders
  * Contains Python notebooks for each subproject

* ``scripts/`` - Reserved for continuous integration (CI) scripts:

  * All auxiliary code for notebooks should live in the appropriate subpackages
  * CI scripts handle testing, linting, and deployment

**Development Setup:**

To install development dependencies:

.. code:: sh

   pip install -e ".[dev]"

**Testing Requirements:**

All tests must pass before a PR can be merged. The testing suite includes:

* **Explicit Tests** (``tests/`` directory): Unit and integration tests for code functionality

  * Focus on testing code logic, not notebooks
  * More comprehensive test coverage is encouraged
  * Run with: ``make test``

* **Notebook Tests**: Jupyter notebooks are included in the test suite by default

  * Use with caution, especially for data-hungry operations
  * Notebooks are validated for execution but not for output correctness
  * Can be excluded from testing if they require heavy computational resources
  * **To exclude a notebook from testing**: Add it to the `EXCLUDED_NOTEBOOKS` list in `scripts/test_notebooks.sh`
  * **To exclude specific cells or code sections**: Use environment-based guards:

    .. code:: python

       smoke_test = "CI" in os.environ
       if not smoke_test:
           # Code that should only run locally and be excluded from CI
           expensive_computation()
           large_data_processing()

**Development Workflow:**

1. Create a feature branch from main
2. Implement changes in the appropriate subpackage
3. Add tests for new functionality in ``tests/``
4. Ensure all tests pass locally
5. Submit a Pull Request
6. CI will automatically run the full test suite
7. PR can be merged only after all tests pass AND human review is completed

**Running CI Locally:**

Before submitting a PR, you should run the CI checks locally to ensure your code meets the project standards. The CI process includes code formatting, linting, and testing.

**For Code and Tests:**

* ``make format`` - Automatically formats and fixes code style issues

  * Runs ``ruff check --fix`` to fix common code issues
  * Runs ``ruff format`` to format code according to project standards

* ``make lint`` - Checks code quality without making changes

  * Runs ``mypy`` for static type checking
  * Runs ``ruff check`` to identify code quality issues
  * Runs ``ruff format --diff`` to show formatting differences

* ``make test`` - Runs the test suite

  * Executes all tests in the ``tests/`` directory
  * Uses parallel execution for faster test runs
  * Automatically runs linting checks first

**For Notebooks:**

* ``make format-notebooks`` - Formats Jupyter notebooks

  * Runs ``nbqa isort`` to organize imports
  * Runs ``nbqa black`` to format code cells

* ``make lint-notebooks`` - Checks notebook code quality

  * Runs ``nbqa mypy`` for type checking in notebooks
  * Runs ``nbqa isort --check`` to verify import organization
  * Runs ``nbqa black --check`` to verify formatting
  * Runs ``nbqa flake8`` for additional code quality checks

* ``make test-notebooks`` - Tests notebook execution

  * Validates that notebooks can be executed without errors
  * Uses ``pytest`` with ``nbval-lax`` for notebook testing
  * Excludes resource-intensive notebooks from testing

**What is Linting?**

Linting is a static code analysis tool that checks your code for potential errors, style violations, and quality issues without executing it. It helps maintain consistent code quality across the project by:

* Identifying syntax errors and potential bugs
* Enforcing coding style and formatting standards
* Detecting unused imports and variables
* Ensuring type safety (with mypy)
* Maintaining consistent code organization

**Submitting Changes:**

For larger changes, please open an issue for discussion before submitting a pull request.

In your PR, please include:

* Changes made
* Links to related issues/PRs
* Tests
* Dependencies

For speculative changes meant for early-stage review, include ``[WIP]`` in the PR's title.

**Best Practices:**

* Keep auxiliary code in subpackages, not in ``scripts/``
* Write comprehensive tests for new functionality
* Use notebooks sparingly in the test suite for resource-intensive operations
* Follow the established folder structure for new features
* Follow PEP8 style guide and use the established coding style
* **Avoid adding temporary or large files to git**: Do not commit temporary files, large data files, or generated outputs to the repository. Check the `.gitignore` file for patterns of files that should be excluded, and add new patterns as needed for your subproject.
* **Maintain proper Python package structure**: Ensure all directories that should be Python packages contain an `__init__.py` file (can be empty). This is required for Python to recognize directories as packages and enables proper imports. When adding new subpackages, always include the `__init__.py` file.