
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


* Using gcloud

   Use of gcloud data access requires API keys stored outside this repository. Please obtain the API keys and create a ```.env``` file
   in the root directory of this repository. See below for an example:

   .. code:: sh
   
      COLLAB_DATA_KEY=path/to/api/key.json

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
