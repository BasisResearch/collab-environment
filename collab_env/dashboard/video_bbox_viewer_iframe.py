"""
Robust iframe-based video bbox viewer using the working Flask app.
This avoids all Panel DOM timing issues by embedding the proven solution.
"""

import panel as pn
import pandas as pd
import json
import subprocess
import time
import threading
import socket
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
import tempfile
import os
import signal
import atexit

# Enable Panel extensions
pn.extension()


class FlaskVideoServer:
    """
    Manages a Flask video server for serving video + bbox overlays.
    """
    
    def __init__(self, video_path: str, bbox_df: pd.DataFrame, fps: float = 30.0, port: int = None):
        """Initialize the Flask server."""
        self.video_path = Path(video_path)
        self.bbox_df = bbox_df.copy()
        self.fps = fps
        self.port = port or self._find_free_port()
        self.process = None
        self.temp_csv = None
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
    
    def _find_free_port(self) -> int:
        """Find a free port for the Flask server."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]
    
    def _create_temp_csv(self) -> str:
        """Create a temporary CSV file with bbox data."""
        # Create temp file
        fd, temp_path = tempfile.mkstemp(suffix='.csv', prefix='bbox_data_')
        os.close(fd)  # Close file descriptor, we'll write with pandas
        
        # Save DataFrame to temp CSV
        self.bbox_df.to_csv(temp_path, index=False)
        self.temp_csv = temp_path
        
        logger.info(f"Created temp CSV: {temp_path}")
        return temp_path
    
    def set_csv_path(self, csv_path: str):
        """Use an existing CSV file instead of creating a temp one."""
        self.temp_csv = csv_path
        logger.info(f"Using existing CSV: {csv_path}")
    
    def start(self, csv_path: str = None) -> str:
        """Start the Flask server and return the URL."""
        if self.process and self.process.poll() is None:
            # Server already running
            return f"http://localhost:{self.port}"
        
        # Use provided CSV path or create temporary CSV file
        if csv_path:
            self.set_csv_path(csv_path)
        else:
            csv_path = self._create_temp_csv()
        
        # Get the Flask app script path
        flask_app_path = Path(__file__).parent.parent.parent / "video_overlay_webapp.py"
        
        if not flask_app_path.exists():
            raise FileNotFoundError(f"Flask app not found: {flask_app_path}")
        
        # Start Flask server as subprocess
        cmd = [
            "python", str(flask_app_path),
            "--video", str(self.video_path),
            "--csv", csv_path,
            "--fps", str(self.fps),
            "--port", str(self.port),
            "--host", "localhost"
        ]
        
        logger.info(f"Starting Flask server: {' '.join(cmd)}")
        
        try:
            # Start process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            url = f"http://localhost:{self.port}"
            max_attempts = 30
            
            for attempt in range(max_attempts):
                try:
                    import requests
                    response = requests.get(url, timeout=1)
                    if response.status_code == 200:
                        logger.info(f"Flask server started successfully at {url}")
                        return url
                except:
                    time.sleep(0.5)
            
            # Server didn't start
            self.cleanup()
            raise RuntimeError(f"Flask server failed to start after {max_attempts} attempts")
            
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Failed to start Flask server: {e}")
    
    def cleanup(self):
        """Clean up the Flask server and temp files."""
        # Kill process
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                try:
                    self.process.kill()
                except:
                    pass
            self.process = None
        
        # Remove temp CSV
        if self.temp_csv and os.path.exists(self.temp_csv):
            try:
                os.unlink(self.temp_csv)
                logger.info(f"Cleaned up temp CSV: {self.temp_csv}")
            except:
                pass
            self.temp_csv = None
    
    def is_running(self) -> bool:
        """Check if the server is running."""
        return self.process is not None and self.process.poll() is None


class IframeVideoBboxViewer:
    """
    Panel component that embeds the working Flask app in an iframe.
    """
    
    def __init__(self, video_path: str, bbox_df: pd.DataFrame, fps: float = 30.0):
        """Initialize the iframe viewer."""
        self.video_path = Path(video_path)
        self.bbox_df = bbox_df.copy()
        self.fps = fps
        
        # Validate video file
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Start Flask server
        self.flask_server = FlaskVideoServer(video_path, bbox_df, fps)
        self.server_url = None
        
        # Create the component
        self.component = self._create_component()
    
    def _create_component(self) -> pn.Column:
        """Create the Panel component with iframe."""
        
        # Start server
        try:
            self.server_url = self.flask_server.start()
            status_html = f"""
            <div style="background: #d4edda; color: #155724; padding: 10px; border-radius: 5px; margin: 10px 0;">
                <h4>🎥 Video Bbox Viewer (Iframe)</h4>
                <p><strong>✅ Status:</strong> Flask server running at <a href="{self.server_url}" target="_blank">{self.server_url}</a></p>
                <p><strong>Video:</strong> {self.video_path.name}</p>
                <p><strong>Tracks:</strong> {len(self.bbox_df['track_id'].unique())}</p>
                <p><strong>Annotations:</strong> {len(self.bbox_df)}</p>
                <p><strong>FPS:</strong> {self.fps}</p>
                <p><small>💡 <strong>Tip:</strong> This uses a separate Flask server for maximum compatibility and performance.</small></p>
            </div>
            """
            
            # Create iframe
            iframe_html = f"""
            <iframe 
                src="{self.server_url}" 
                width="100%" 
                height="800" 
                frameborder="0"
                style="border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            </iframe>
            """
            
        except Exception as e:
            logger.error(f"Failed to start Flask server: {e}")
            status_html = f"""
            <div style="background: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; margin: 10px 0;">
                <h4>❌ Video Bbox Viewer (Error)</h4>
                <p><strong>Error:</strong> Failed to start Flask server</p>
                <p><strong>Details:</strong> {e}</p>
                <p><strong>Video:</strong> {self.video_path.name}</p>
            </div>
            """
            iframe_html = "<p>Video viewer unavailable due to server error.</p>"
        
        # Create components
        status_pane = pn.pane.HTML(status_html)
        iframe_pane = pn.pane.HTML(iframe_html, width=950, height=820)
        
        return pn.Column(
            status_pane,
            iframe_pane,
            sizing_mode='stretch_width'
        )
    
    def panel(self) -> pn.Column:
        """Return the Panel component."""
        return self.component
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'flask_server'):
            self.flask_server.cleanup()


def create_video_bbox_viewer(video_path: str, bbox_df: pd.DataFrame, fps: float = 30.0) -> pn.Column:
    """
    Create an iframe-based video bbox viewer.
    
    Args:
        video_path: Path to the MP4 video file
        bbox_df: DataFrame with bbox/centroid data
        fps: Video frame rate
        
    Returns:
        Panel Column component with embedded Flask app
    """
    viewer = IframeVideoBboxViewer(video_path, bbox_df, fps)
    return viewer.panel()


# Test data function (same as before)
def create_test_data() -> pd.DataFrame:
    """Create sample bbox data for testing."""
    import numpy as np
    
    data = []
    for frame in range(300):
        for track_id in range(3):
            base_x = 100 + track_id * 200 + np.sin(frame * 0.1 + track_id) * 80
            base_y = 100 + np.cos(frame * 0.05 + track_id) * 60
            
            data.append({
                'track_id': track_id,
                'frame': frame,
                'x1': base_x - 30,
                'y1': base_y - 30,
                'x2': base_x + 30,
                'y2': base_y + 30
            })
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    print("Iframe Video Bbox Viewer created!")