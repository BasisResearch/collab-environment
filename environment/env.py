import os   
import glob
import subprocess
from pathlib import Path
from typing import Optional, TypedDict, Literal, Set, Dict, Any, Union
# from .utils import managed_process
# import cv2
# import numpy as np
# import torch 

DEFAULT_TIMEOUT = 3600

class EnvironmentConfig(TypedDict):
    """Configuration for the Environment class.
    
    Required Keys:
        file_path: Path to the input file for processing (e.g. video, images, etc.)
        dtype: Specifies if the data is 2D or 3D
        method: Processing method to use (different methods for 2D and 3D)
        
    Optional Keys:
        output_path: Path for output data
            - preproc: preprocessed images
            - model_ckpts: derived model images
            - If output_path is not specified, will default to the grandparent directory of the input file
        overwrite: If True, will rerun preprocessing even if transforms.json exists
    """
    file_path: Union[str, Path]
    dtype: Literal["2d", "3d"]
    method: str
    overwrite: bool
    output_path: Optional[Union[str, Path]]

class ValidationError(Exception):
    """Raised when environment configuration is invalid."""
    pass

class Environment:
    # Valid processing methods for each dtype
    METHODS_2D: Set[str] = {
        "clip",
    }
    
    METHODS_3D: Set[str] = {
        "splatfacto",
        "feature-splatting",
    }

    def __init__(self, config: EnvironmentConfig):
        """Initialize the environment with configuration.
        
        Args:
            config: Configuration dictionary specifying environment parameters
        """
        # Validate config before initialization
        validated_config = self.validate_config(config)
        self.config: Dict[str, Any] = dict(validated_config)

    @classmethod
    def validate_config(cls, config: EnvironmentConfig) -> EnvironmentConfig:
        """Validate the environment configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ValidationError: If the configuration is invalid
        """

        ############################################
        ######### Check fields of config ###########
        ############################################

        required_fields = {"file_path", "dtype", "method"}
        missing_fields = required_fields - set(config.keys())

        if missing_fields:
            raise ValidationError(f"Missing required fields: {missing_fields}")

        # Validate dtype
        if config["dtype"] not in ["2d", "3d"]:
            raise ValidationError("dtype must be either '2d' or '3d'")
        
        # Validate method based on dtype
        valid_methods = cls.METHODS_2D if config["dtype"] == "2d" else cls.METHODS_3D
        if config["method"] not in valid_methods:
            raise ValidationError(
                f"Invalid method '{config['method']}' for dtype '{config['dtype']}'. "
                f"Valid methods are: {sorted(valid_methods)}"
            )

        if config.get('overwrite') is None:
            config.setdefault('overwrite', False)

        ############################################
        ############ Set up file paths #############
        ############################################

        # Set the file path -> turn to a Path object for easy structuring           
        file_path = Path(config["file_path"])

        if not file_path.exists():
            raise ValidationError(f"File not found: {file_path}")

        # If so, set the file path 
        config['file_path'] = file_path
        
        # If we don't specify an output path, default to the grandparent directory of the input file
        if config.get('output_path') is None:
            default_output_path = os.path.join(file_path.parent.parent, 'environment', file_path.stem)
            config.setdefault('output_path', Path(default_output_path)) # type: ignore
            
        return config
    
    @classmethod
    def available_methods(cls, dtype: Optional[str] = None) -> None:
        """Print the available methods, either for all dtypes or a specific dtype.
        
        Args:
            dtype: Optional; if provided, prints methods for specific dtype ('2d' or '3d')
                  if None, prints all methods grouped by dtype
        """
        if dtype is not None:
            if dtype not in ["2d", "3d"]:
                raise ValueError("dtype must be either '2d' or '3d'")
            methods = cls.METHODS_2D if dtype == "2d" else cls.METHODS_3D
            print(f"Available methods for {dtype}:")
            print(f"  {sorted(methods)}")
            return

        print("Available methods:")
        print("  2D methods:", sorted(cls.METHODS_2D))
        print("  3D methods:", sorted(cls.METHODS_3D))

    def preprocess_data(self) -> None:
        """Preprocess the data in the environment.
        
        This function handles any necessary data preprocessing steps based on the
        configured dtype and method.
        """
        if self.config["dtype"] == "2d":
            self._preprocess_2d()
        else:
            self._preprocess_3d()

    def _preprocess_2d(self) -> None:
        """2D-specific preprocessing based on selected method."""
        pass

    def _preprocess_3d(self) -> None:
        """3D-specific preprocessing based on selected method."""
        file_path = self.config['file_path']
        output_path = self.config['output_path']
        
        # Determine input type based on file extension
        ext = file_path.suffix.lower()
        if ext in ['.mp4', '.mov', '.avi']:
            input_type = 'video'
        elif ext in ['.jpg', '.jpeg', '.png']:
            if '360' in str(file_path):
                input_type = 'images --camera-type equirectangular --images-per-equirect 14'
            else:
                input_type = 'images'
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        # Set the output path to same directory as input fil    
        preproc_data_path = output_path / 'preproc'
        transforms_path = preproc_data_path / "transforms.json"

        # If the transforms exists and we don't want to overwrite
        # Return and store the processed_data_path
        if transforms_path.exists() and not self.config.get('overwrite', False):
            print(f"transforms.json already exists at {transforms_path}")
            print("To rerun preprocessing, set overwrite=True in config")
            self.config['preproc_data_path'] = preproc_data_path
            return

        # TLB --> we should bump up number of frames to max 
        cmd = (
            f"ns-process-data "
            f"{input_type} "
            f"--data {file_path.as_posix()} "
            f"--output-dir {preproc_data_path.as_posix()} "
            # f"--num-frames-target {n_samples} "
        )

        subprocess.run(cmd, shell=True)
        
        # Store the preprocessed data path in the config
        self.config['preproc_data_path'] = preproc_data_path

    def extract_features(self) -> None:
        """Extract features from the preprocessed data.
        
        Feature extraction is performed according to the configured dtype and method.
        """
        if self.config["dtype"] == "2d":
            # self._extract_features_2d()
            pass
        else:
            self._extract_features_3d()

    def _extract_features_2d(self) -> None:
        """2D-specific feature extraction based on selected method."""
        pass

    def _extract_features_3d(self) -> None:
        """3D-specific feature extraction based on selected method."""

        method = self.config["method"]

        # Check that preprocessing was completed before extracting features
        if self.config['preproc_data_path'] is None:
            raise ValueError("preprocess_data() must be run before extracting features")

        # Set the model path (where it outputs results of 3d model training)
        model_path = self.config['output_path'] / method

        if os.path.exists(model_path) and not self.config['overwrite']:
            print(f"Output already exists for {method}")
            print("To rerun feature extraction, set overwrite=True in config")
            self.config['model_path'] = model_path
            return

        cmd = (
            f"ns-train "
            f"{method} "
            f"--data {self.config['preproc_data_path'].as_posix()} "
            f"--output-dir {self.config['output_path'].as_posix()} "
            f"--experiment-name ''" # This keeps our file structure as environment/BASE_NAME/method/
        )

        self.config['model_path'] = model_path
        subprocess.run(cmd, shell=True, timeout=DEFAULT_TIMEOUT)
    
    def viewer(self, websocket_port: int = 80) -> None:
        """Display or visualize the environment data.
        
        This function handles feature extraction from the preprocessed data.
        The specific feature extraction pipeline depends on config['dtype']:
            - 2D: Image-based feature extraction
            - 3D: Volume-based feature extraction
        """

        # Find all runs with config.yml files
        output_dir = Path(str(self.config['output_path']), self.config['method'])

        # Grab all directories with a config.yml file --> convert to paths
        run_dirs = glob.glob(os.path.join(output_dir, "**/config.yml"))
        run_dirs = [Path(run_dir).parent for run_dir in run_dirs]
            
        if not run_dirs:
            raise ValueError(f"No runs with config.yml found in {output_dir}")
            
        # Sort runs by directory name (which contains timestamp)
        sorted_runs = sorted(run_dirs)
        
        # Print available runs
        print("\nAvailable runs:")
        for i, run in enumerate(sorted_runs):
            print(f"[{i}] {run.name}")
            
        # Prompt user to select a run
        while True:
            try:
                selection = input("\nSelect run number (or press Enter for most recent): ").strip()
                if selection == "":
                    selected_run = sorted_runs[-1]
                    break
                idx = int(selection)
                if 0 <= idx < len(sorted_runs):
                    selected_run = sorted_runs[idx]
                    break
                print(f"Please enter a number between 0 and {len(sorted_runs)-1}")
            except ValueError:
                print("Please enter a valid number")

        self.config['model_path'] = selected_run.as_posix()

        cmd = (
            f"ns-viewer "
            f"--load-config {self.config['model_path']}/config.yml "
            f"--viewer.websocket-port {websocket_port}"
        )
        
        subprocess.run(cmd, shell=True, timeout=DEFAULT_TIMEOUT)