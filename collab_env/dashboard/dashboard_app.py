"""
Standalone Panel app file for use with `panel serve` command.

This allows using Panel's native autoreload without the new tab issue:
    panel serve dashboard_app.py --dev --show --port 5007

You can also pass custom arguments:
    panel serve dashboard_app.py --dev --show --port 5007 --args --curated-bucket my_curated --processed-bucket my_processed --read-only
"""

import panel as pn
import argparse
from collab_env.dashboard.app import create_app

# Configure Panel
pn.extension("tabulator", "vtk")

# Parse command line arguments (when using panel serve ... --args)
parser = argparse.ArgumentParser(description="Dashboard configuration")
parser.add_argument(
    "--remote-name",
    default="collab-data",
    help="Name of rclone remote (default: collab-data)",
)
parser.add_argument(
    "--curated-bucket",
    default="fieldwork_curated",
    help="Name of curated data bucket (default: fieldwork_curated)",
)
parser.add_argument(
    "--processed-bucket",
    default="fieldwork_processed",
    help="Name of processed data bucket (default: fieldwork_processed)",
)
parser.add_argument(
    "--read-only",
    action="store_true",
    help="Enable read-only mode (disable upload/delete functionality)"
)

# Parse known args to handle panel serve's additional arguments
args, unknown = parser.parse_known_args()

# Create the app with the provided arguments
app = create_app(
    remote_name=args.remote_name,
    curated_bucket=args.curated_bucket,
    processed_bucket=args.processed_bucket,
    read_only=args.read_only,
)

# Make it servable
app.servable()
