import os
from pathlib import Path
from typing import Optional, Union


def expand_path(path: Union[str, Path], base_path: Optional[Path] = None) -> Path:
    """
    Expand a path that might be relative, contain user dir (~), or environment variables.

    Args:
        path: Path to expand
        base_path: Optional base path for relative paths (if None, uses current working directory)

    Returns:
        Absolute Path object
    """
    # Convert to Path if string
    path = Path(path)

    # Expand user directory and environment variables
    path = Path(os.path.expanduser(str(path)))
    path = Path(os.path.expandvars(str(path)))

    # If path is absolute, just return it
    if path.is_absolute():
        return path

    # For relative paths, use base_path or current directory
    base = base_path or Path.cwd()
    return (base / path).resolve()


def get_project_root(
    markers: list[str] = ["setup.cfg", ".git", "pyproject.toml"],
) -> Path:
    """Find project root by checking multiple possible marker files"""

    current = Path().absolute()
    while current != current.parent:
        for marker in markers:
            if (current / marker).exists():
                return current
        current = current.parent

    raise FileNotFoundError(
        f"Could not find project root (no markers found: {markers})"
    )
