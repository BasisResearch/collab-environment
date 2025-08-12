"""
Command-line interface for the data dashboard.
"""

import argparse
import sys
from loguru import logger

from .app import serve_dashboard


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GCS Data Dashboard - Browse and edit data files from GCS buckets via rclone"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5007,
        help="Port to serve the dashboard on (default: 5007)",
    )

    parser.add_argument(
        "--no-browser", action="store_true", help="Don't automatically open browser"
    )

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

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    parser.add_argument("--autoreload", action="store_true", help="Enable autoreload")

    args = parser.parse_args()

    # Configure logging
    log_level = "DEBUG" if args.debug else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    logger.info("Starting CIS Data Dashboard...")
    logger.info(f"Remote: {args.remote_name}")
    logger.info(f"Curated bucket: {args.curated_bucket}")
    logger.info(f"Processed bucket: {args.processed_bucket}")
    logger.info(f"Port: {args.port}")

    try:
        show_browser = not args.no_browser

        if args.no_browser:
            logger.info("The browser will not be opened automatically")

        if args.autoreload:
            logger.info("🔄 Autoreload enabled")
            logger.info(f"🌐 Dashboard URL: http://localhost:{args.port}")
            logger.info("📝 Edit any .py file to see automatic refresh")
            logger.warning(
                "⚠️  Note: Panel's autoreload with pn.serve() opens new browser tabs"
            )
            logger.info(
                "💡 For better autoreload experience, use: panel serve dashboard_app.py --dev --show"
            )
            # For autoreload, let Panel manage browser behavior to avoid new tabs
            serve_dashboard(port=args.port, show=show_browser, autoreload=True)
        else:
            serve_dashboard(port=args.port, show=show_browser, autoreload=False)
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
