"""
Standalone Panel app file for use with `panel serve` command.

This allows using Panel's native autoreload without the new tab issue:
    panel serve dashboard_app.py --dev --show --port 5007
"""

import panel as pn
from collab_env.dashboard.app import create_app

# Configure Panel
pn.extension("tabulator")

# Create the app
app = create_app()

# Make it servable
app.servable()
