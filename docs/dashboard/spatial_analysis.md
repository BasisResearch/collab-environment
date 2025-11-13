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
  * ✅ velocity (speed distribution, mean velocity magnitude) (implemented in Velocity Stats widget)
  * acceleration (distribution), acceleration (vector)
  * ✅ Mean velocity direction over time (distribution / time series) (implemented in Velocity Stats widget)

* pairwise statistics - both "point clouds" (vector quantities) and distributions of magnitude
  * ✅ relative positions (distances) (implemented in Distance widget)
  * ✅ relative velocities (implemented in Velocity Stats widget)

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
- [x] Enhanced Velocity Stats widget with three analysis groups:
  - Individual agent speed (histogram + time series with mean ± std)
  - Mean velocity magnitude (normalized velocities, magnitude of mean vector)
  - Relative velocity magnitude (pairwise ||v_i - v_j|| analysis)
- [x] Enhanced Distance widget with pairwise relative locations:
  - Relative distances ||x_i - x_j|| between all agent pairs
  - Histogram + time series with mean ± std bands
- [x] Velocity Correlations widget (episode-level)

🚧 Planned (Phase 7.2-7.3):

- [ ] Enhanced Correlation Viewer (windowed/lagged modes)

📝 Future:

- [ ] Clustering statistics
- [ ] Velocity vector distributions (quiver plots)
- [ ] Acceleration analysis
