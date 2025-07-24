import os
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from tqdm import tqdm
from ultralytics import YOLO
import yaml
import random
import shutil
import tempfile
from typing import List, Optional, Dict, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Import from the scripts folder (locally these are all on my desktop lol) 

from collab_env.tracking.thermal_export import process_directory
from collab_env.tracking.timestamps import get_creation_time, extract_timestamps_from_folder
from scripts.time_alignment import (
    extract_ffmpeg_timecode, rgb_timecode_to_ms, csq_filename_to_ms, 
    trim_video, trim_video_pair, match_rgb_to_thermal, trim_session)
from collab_env.tracking.collect_thermal_mp4s import collect_thermal_mp4s
from collab_env.tracking.frame_extraction import extract_frames_from_folder



########################################
@dataclass
class ProcessingConfig:
    """Configuration for video processing pipeline"""
    src_root_dir: Path
    target_root_dir: Path
    thermal_color: str = 'magma'
    preview: bool = True
    max_frames: int = 20000
    fps: int = 30
    target_fps: int = 15
    max_diff_ms: int = 70000

class AnimalVideoProcessor:
    """Main class for processing animal behavior videos"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.ensure_directories()
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        self.config.target_root_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Working directory: {self.config.target_root_dir}")
    
    def get_sessions(self, date_pattern: str = None) -> List[Path]:
        """Get all session directories matching the pattern"""
        if date_pattern is None:
            # Get all session directories
            pattern = "*-session_*"
        else:
            pattern = f"{date_pattern}-session_*"
        
        sessions = sorted([
            p for p in self.config.target_root_dir.iterdir() 
            if p.is_dir() and p.name.startswith(date_pattern or "") and "session_" in p.name
        ])
        
        logger.info(f"Found {len(sessions)} sessions matching pattern: {pattern}")
        return sessions
    
    def process_session_thermals(self, session_path: Path, operation: str, **kwargs) -> bool:
        """Generic method to process thermal cameras in a session"""
        success = True
        
        # Find thermal directories
        thermal_dirs = [d for d in session_path.iterdir() 
                       if d.is_dir() and d.name.startswith('thermal_')]
        
        if not thermal_dirs:
            logger.warning(f"No thermal directories found in {session_path}")
            return False
        
        for thermal_dir in thermal_dirs:
            try:
                logger.info(f"Processing {thermal_dir.name} with operation: {operation}")
                
                if operation == "csq_processing":
                    self._process_csq(thermal_dir, **kwargs)
                elif operation == "frame_extraction":
                    self._process_frame_extraction(thermal_dir, **kwargs)
                else:
                    logger.error(f"Unknown operation: {operation}")
                    success = False
                    
            except Exception as e:
                logger.error(f"Error processing {thermal_dir}: {e}")
                success = False
        
        return success
    
    def _process_csq(self, thermal_dir: Path, **kwargs):
        """Process CSQ files in thermal folder"""
        logger.info(f"Processing CSQ files in {thermal_dir}")
        
        try:
            process_directory(
                thermal_dir, 
                thermal_dir, 
                color=self.config.thermal_color,
                preview=self.config.preview,
                max_frames=self.config.max_frames,
                fps=self.config.fps
            )
            logger.info(f"CSQ processing completed for {thermal_dir}")
        except Exception as e:
            logger.error(f"Error in CSQ processing for {thermal_dir}: {e}")
            raise
    
    def _process_frame_extraction(self, thermal_dir: Path, **kwargs):
        """Extract frames from thermal videos (use time-cropped videos if available)"""
        logger.info(f"Starting frame extraction for {thermal_dir}")
        
        # Create output folder for extracted frames
        output_folder = thermal_dir / "extracted_frames"
        output_folder.mkdir(parents=True, exist_ok=True)
        
        try:
            # Use the existing thermal extraction logic
            extract_frames_from_folder(
                thermal_dir, 
                output_folder, 
                target_fps=self.config.target_fps
            )
            logger.info(f"Frame extraction completed for {thermal_dir}")
        except Exception as e:
            logger.error(f"Error in frame extraction for {thermal_dir}: {e}")
            raise
    
    def _process_frames_session(self, session_path: Path) -> bool:
        """Process frame extraction for entire session"""
        logger.info(f"Processing session-wide frame extraction for {session_path}")
        
        try:
            # Find all camera directories (thermal and RGB)
            camera_dirs = [d for d in session_path.iterdir() 
                          if d.is_dir() and (d.name.startswith('thermal_') or d.name.startswith('rgb_'))]
            
            if not camera_dirs:
                logger.warning(f"No camera directories found in {session_path}")
                return False
            
            success = True
            for camera_dir in camera_dirs:
                try:
                    if camera_dir.name.startswith('thermal_'):
                        self._process_frame_extraction(camera_dir)
                    elif camera_dir.name.startswith('rgb_'):
                        self._process_rgb_frame_extraction(camera_dir)
                except Exception as e:
                    logger.error(f"Error processing {camera_dir}: {e}")
                    success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Error in session frame processing: {e}")
            return False
    
    def _process_rgb_frame_extraction(self, rgb_dir: Path):
        """Extract frames from RGB videos"""
        logger.info(f"Processing RGB frame extraction for {rgb_dir}")
        
        # Create output folder for extracted frames
        output_folder = rgb_dir / "extracted_frames"
        output_folder.mkdir(parents=True, exist_ok=True)
        
        try:
            # Use the same frame extraction method - it should handle RGB videos too
            extract_frames_from_folder(
                rgb_dir, 
                output_folder, 
                target_fps=self.config.target_fps
            )
            logger.info(f"RGB frame extraction completed for {rgb_dir}")
        except Exception as e:
            logger.error(f"Error in RGB frame extraction for {rgb_dir}: {e}")
            raise
    
    def process_all_sessions(self, date_pattern: str = None, operations: List[str] = None):
        """Process all sessions with specified operations"""
        if operations is None:
            operations = ["csq_processing", "frame_extraction"]
        
        sessions = self.get_sessions(date_pattern)
        
        for session_path in tqdm(sessions, desc="Processing sessions"):
            self.process_single_session(session_path, operations)
    
    def process_single_session(self, session_path: Path, operations: List[str] = None) -> bool:
        """Process a single session with all operations in correct order"""
        if operations is None:
            operations = ["csq_processing", "frame_extraction"]
        
        logger.info(f"Processing session: {session_path.name}")
        
        # Validate session structure first
        issues = self.validate_session_structure(session_path)
        if issues:
            logger.warning(f"Session {session_path.name} has structural issues: {issues}")
            # Continue processing anyway - some operations might still work
        
        success = True
        for operation in operations:
            logger.info(f"Running {operation} for {session_path.name}")
            
            try:
                # Handle session-wide operations differently
                if operation == "session_frame_extraction":
                    if not self._process_frames_session(session_path):
                        logger.error(f"Failed {operation} for {session_path.name}")
                        success = False
                elif operation == "temporal_alignment":
                    if not self._process_temporal_alignment(session_path):
                        logger.error(f"Failed {operation} for {session_path.name}")
                        success = False
                else:
                    if not self.process_session_thermals(session_path, operation):
                        logger.error(f"Failed {operation} for {session_path.name}")
                        success = False
                        # Continue with other operations
                        
            except Exception as e:
                logger.error(f"Error in {operation} for {session_path.name}: {e}")
                success = False
        
        return success
    
    def _process_temporal_alignment(self, session_path: Path) -> bool:
        """Process temporal alignment for RGB and thermal videos"""
        logger.info(f"Processing temporal alignment for {session_path}")
        
        try:
            # Find RGB and thermal directories
            rgb_dirs = [d for d in session_path.iterdir() 
                       if d.is_dir() and d.name.startswith('rgb_')]
            thermal_dirs = [d for d in session_path.iterdir() 
                           if d.is_dir() and d.name.startswith('thermal_')]
            
            if not rgb_dirs or not thermal_dirs:
                logger.warning(f"Missing RGB or thermal directories in {session_path}")
                return False
            
            # Match RGB to thermal for each pair
            success = True
            for rgb_dir in rgb_dirs:
                # Find corresponding thermal directory (e.g., rgb_1 -> thermal_1)
                rgb_num = rgb_dir.name.split('_')[-1]
                thermal_dir = session_path / f"thermal_{rgb_num}"
                
                if thermal_dir.exists():
                    try:
                        if not self.match_rgb_thermal_videos(rgb_dir, thermal_dir):
                            success = False
                    except Exception as e:
                        logger.error(f"Error matching {rgb_dir.name} to {thermal_dir.name}: {e}")
                        success = False
                else:
                    logger.warning(f"No corresponding thermal directory found for {rgb_dir.name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error in temporal alignment: {e}")
            return False
    
    def match_rgb_thermal_videos(self, rgb_dir: Path, thermal_dir: Path) -> bool:
        """Match RGB and thermal videos by timestamp"""
        try:
            logger.info(f"Matching videos between {rgb_dir.name} and {thermal_dir.name}")
            match_rgb_to_thermal(
                rgb_dir=rgb_dir,
                thermal_dir=thermal_dir,
                max_diff_ms=self.config.max_diff_ms
            )
            return True
        except Exception as e:
            logger.error(f"Error matching RGB/thermal videos: {e}")
            return False
    
    def get_session_summary(self, session_path: Path) -> Dict:
        """Get summary information about a session"""
        summary = {
            'session_name': session_path.name,
            'thermal_1_exists': (session_path / 'thermal_1').exists(),
            'thermal_2_exists': (session_path / 'thermal_2').exists(),
            'rgb_1_exists': (session_path / 'rgb_1').exists(),
            'rgb_2_exists': (session_path / 'rgb_2').exists(),
        }
        
        # Add file counts
        for camera_type in ['thermal_1', 'thermal_2', 'rgb_1', 'rgb_2']:
            camera_path = session_path / camera_type
            if camera_path.exists():
                if 'thermal' in camera_type:
                    # Count CSQ files
                    csq_count = len(list(camera_path.glob('*.csq')))
                    summary[f'{camera_type}_csq_count'] = csq_count
                    
                    # Count MP4 files (processed)
                    mp4_count = len(list(camera_path.glob('*.mp4')))
                    summary[f'{camera_type}_mp4_count'] = mp4_count
                else:
                    # Count MP4 files for RGB
                    mp4_count = len(list(camera_path.glob('*.mp4')))
                    summary[f'{camera_type}_mp4_count'] = mp4_count
        
        return summary
    
    def validate_session_structure(self, session_path: Path) -> List[str]:
        """Validate that session has expected structure"""
        issues = []
        
        # Check for required thermal directories
        for thermal_dir in ['thermal_1', 'thermal_2']:
            thermal_path = session_path / thermal_dir
            if not thermal_path.exists():
                issues.append(f"Missing {thermal_dir} directory")
                continue
                
            # Check for CSQ files
            csq_files = list(thermal_path.glob('*.csq'))
            if not csq_files:
                issues.append(f"No CSQ files found in {thermal_dir}")
        
        # Check for RGB directories
        for rgb_dir in ['rgb_1', 'rgb_2']:
            rgb_path = session_path / rgb_dir
            if not rgb_path.exists():
                issues.append(f"Missing {rgb_dir} directory")
                continue
                
            # Check for MP4 files
            mp4_files = list(rgb_path.glob('*.MP4'))
            if not mp4_files:
                issues.append(f"No MP4 files found in {rgb_dir}")
        
        return issues

