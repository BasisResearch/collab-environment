
.. index-inclusion-marker

collab-environment
==================

`collab-environment` is an integration package across projects for representing, modeling, and simulating behavior within 3D environments.

For more details, see the paper


***I. Aitsahalia et al., “Inferring cognitive strategies from groups of animals in natural environments,” presented at the NeurIPS Workshop on Data on the Brain \& Mind Findings, 2025.***

To reproduce the figures from the paper, see `paper_figures.md <docs/paper_figures.md>`_


------------

Setup
-----

**Note:**  For alignment functionality, we strongly recommend a CUDA 11.8 compatible GPU. Using a different CUDA version will alter COLMAP results. We recommend matching the CUDA version within [collab-splats](https://github.com/BasisResearch/collab-splats).

* Using pip / uv:

  For installation with pip / uv, all dependencies are installed via a shell script:

   .. code:: sh

      # create and activate venv, e.g.
      # pip install uv
      # uv venv --python=3.10 && source .venv/bin/activate && uv pip install pip
      bash setup.sh

      # For development (includes testing and linting tools):
      bash setup.sh --dev


* Using conda

   We also provide a conda `env.yml` file that can be used to create a conda environment with the necessary dependencies. Run the following commands to create the environment:

   .. code:: sh

      conda env create -n collab-env -f env.yml
      conda activate collab-env
      bash setup.sh  # or: bash setup.sh --dev


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


* Install rclone (Required for Dashboard and GNN data access)

  rclone is used to access Google Cloud Storage buckets for the dashboard and GNN training data:

   .. code:: sh

      # For MacOS
      brew install rclone

      # For Linux (Ubuntu/Debian)
      sudo apt-get install rclone

      # For Linux (other) or manual install
      # See https://rclone.org/install/

      # Configure for GCS access (requires service account key)
      rclone config create collab-data "google cloud storage" service_account_file=/path/to/api/key.json

      # Verify installation
      rclone version


* Using gcloud

   Use of gcloud data access requires API keys stored outside this repository. Please obtain the API keys and create a ```.env``` file
   in the root directory of this repository. See below for an example:

   .. code:: sh
   
      COLLAB_DATA_KEY=/path/to/api/key.json

Documentation
-------------

Detailed documentation for specific modules:

* `Dashboard <docs/dashboard/README.md>`_ - Web-based data browser for GCS buckets
* `GNN Training <docs/gnn/GNNReadMe.md>`_ - Graph Neural Network training and rollouts
* `Simulation <docs/sim/README.md>`_ - Boids simulation and output format
* `Tracking <docs/tracking/README.md>`_ - Animal tracking and thermal video processing

For contributing guidelines, see `CONTRIBUTING.md <CONTRIBUTING.md>`_.
