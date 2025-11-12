Descriptive Data Dashboard
==========================

**Note**: This is a legacy requirements document. For current implementation status,
see [README.md](README.md).

Original Requirements
---------------------

* General
  * all statistics either global or with moving time window
  * if boid type is available, all plots for each type and "global" plot
  * before and after t=500 (3d)
  * aggregate episodes from same config / across configs

* Self-statistics
  * ✅ heatmap of visited locations (implemented in Basic Data Viewer)
  * heatmap of velocities (quiver plots)
  * velocity (vector distribution), speed, acceleration (distribution), acceleration (vector)
  * Mean velocity direction over time (distribution / time series) - scalar
  * ✅ distance to target mesh (2d - food) (legacy Distance widget)
  * ✅ distance to scene (2d - distance to boundary) (legacy Distance widget)

* pairwise statistics - both "point clouds" (vector quantities) and distributions of magnitude
  * relative positions (distances)
  * relative velocities

* correlations
  * ✅ velocities (legacy Correlation widget)
  * distances to target
  * distances to mesh
  * mutual distances (for each pair)

* clustering statistics (TBD)

Implementation Status
---------------------

✅ Completed:

- [x] DB backend (QueryBackend with PostgreSQL/DuckDB support)
- [x] Basic Data Viewer (Phase 7.1) with integrated heatmap, animation, and time series
- [x] Legacy widgets for velocity stats, distance stats, and correlations

🚧 Planned (Phase 7.2-7.3):

- [ ] Relative Quantities Viewer (pairwise statistics)
- [ ] Enhanced Correlation Viewer (windowed/lagged modes)

📝 Future:

- [ ] Clustering statistics
- [ ] Velocity vector distributions (quiver plots)
- [ ] Acceleration analysis
