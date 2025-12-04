# Boids Simulation

See the main [README](../../README.rst) for environment setup instructions.

## Running a Simulation

To run a boids simulation:

```sh
python -m collab_env.sim.boids.run_boids_simulator
```

This will start a 3D visualization of the boids simulation using the default configuration file `collab_env/sim/boids/config.yaml`.

To specify a different configuration file:

```sh
python -m collab_env.sim.boids.run_boids_simulator -cf <path_to_config_file>
```

There are many configurable parameters. See the example configuration file for details.

## Visualizer Controls

If the configuration file indicates that the visualizer should be shown, the following key commands are available:

| Key | Action |
|-----|--------|
| **Q** | Quit the current episode. The visualizer will be terminated, and the data for the episode and the video (if specified) will be stored. |
| **P** | Save an image of the current frame to a file. Images are numbered consecutively (image-1.png, image-2.png, etc.) in the run folder. |
| **R** | Reset the viewer orientation to the initial orientation. |

To stop the simulation prematurely when there is no visualizer window, press `Ctrl-C` in the terminal.

## Output Files

The output of the simulation consists of the following files in the run folder specified in the configuration file:

- The configuration file used for the run
- A parquet file for each episode containing a pandas dataframe
- An optional video file for each episode
- Optional images saved by pressing P while the visualizer was running

## Parquet File Format

The parquet file contains a dataframe with the following columns:

| Column | Description |
|--------|-------------|
| `id` | The id of the object |
| `type` | Either 'agent' for an agent or 'env' for an environment object |
| `timestep` | The timestep for which this data applies |
| `x` | The x position of the object (horizontal) |
| `y` | The y position of the object (vertical) |
| `z` | The z position of the object (front and back) |
| `v_x` | The velocity in the x direction |
| `v_y` | The velocity in the y direction |
| `v_z` | The velocity in the z direction |
| `distance_target_center_t` | The distance to the center of the t-th target |
| `distance_to_target_mesh_closest_point_t` | The distance to the closest point on the t-th target mesh |
| `target_mesh_closest_point_t` | The point on the t-th target mesh that is closest to the object |
| `mesh_scene_distance` | The distance to the closest point on the mesh scene |
| `mesh_scene_closest_point` | The point on the mesh scene that is closest to the object |
