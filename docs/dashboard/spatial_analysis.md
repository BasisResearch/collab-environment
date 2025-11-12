Descriptive Data Dashboard
==========================

* General
  * all statistics either global or with moving time window
  * if boid type is available, all plots for each type and "global" plot
  * before and after t=500 (3d)
  * aggregate episodes from same config / across configs

* Self-statistics
  * heatmap of visited locations
  * heatmap of velocities (quiver plots)
  * velocity (vector distribution),  speed, acceleration (distribution), acceleration (vector)
  * Mean velocity direction over time (distribution / time series) - scalar 
  * distance to target mesh (2d - food)
  * distance to scene (2d - distance to boundary)
* pairwise statistics - both "point clouds" (vector quantities) and distrubitions of magnitude
  * relative positions (distances)
  * relative velocities
  
* correlations
  * velocities
  * distances to target
  * distances to mesh
  * mutual distances (for each pair)

* clustering statistics (TBD)

Tasks
- [] a GUI concept - Dima
- [] DB backend (?) - Dima & Jack
- [] add boid type to parquet - Tom
- [] clustering statistics - Tom