"""
Entry point for Spatial Analysis Dashboard.

Run with:
    panel serve collab_env/dashboard/spatial_analysis_app.py --show --port 5008 --static-dirs dashboard-static=collab_env/dashboard/static

Development mode with autoreload:
    panel serve collab_env/dashboard/spatial_analysis_app.py --dev --show --port 5008 --static-dirs dashboard-static=collab_env/dashboard/static

Note: The --static-dirs argument maps the dashboard's static files (JS/CSS)
      to the /dashboard-static/ URL route. This is required for the animation viewer.
"""

import panel as pn
import holoviews as hv
from collab_env.dashboard.spatial_analysis_gui import create_app

# Enable Panel extensions (removed "modal" - not needed for HTML-based overlay)
pn.extension("tabulator", "plotly")
hv.extension("plotly", "bokeh")  # plotly first for Scatter3D support

# Create and serve the app
app = create_app()
app.servable()
