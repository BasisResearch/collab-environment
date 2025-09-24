"""
Test suite for the dashboard components.
"""

import subprocess
import pytest


def test_rclone_availability():
    """Test if rclone is available."""
    try:
        result = subprocess.run(
            ["rclone", "version"], capture_output=True, text=True, check=True
        )
        assert result.returncode == 0
        assert "rclone" in result.stdout.lower()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("rclone not available - install from https://rclone.org/")


def test_rclone_remotes():
    """Test if required rclone remote is configured."""
    pytest.importorskip("subprocess")

    try:
        result = subprocess.run(
            ["rclone", "listremotes"], capture_output=True, text=True, check=True
        )
        remotes = [
            line.strip().rstrip(":")
            for line in result.stdout.strip().split("\n")
            if line.strip()
        ]

        if "collab-data" not in remotes:
            pytest.skip(
                "collab-data remote not configured - run: rclone config create collab-data"
            )

        assert "collab-data" in remotes

    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("rclone not available or remotes not accessible")


def test_dashboard_imports():
    """Test if dashboard modules can be imported."""
    # Test core components
    from collab_env.dashboard.rclone_client import RcloneClient
    from collab_env.dashboard.session_manager import SessionManager
    from collab_env.dashboard.file_viewers import FileContentManager

    # Test UI dependencies (will skip if not available)
    pytest.importorskip("panel")
    pytest.importorskip("param")

    # Test main app
    from collab_env.dashboard.app import DataDashboard, create_app

    # Basic instantiation test
    assert RcloneClient is not None
    assert SessionManager is not None
    assert FileContentManager is not None
    assert DataDashboard is not None
    assert create_app is not None


def test_rclone_client_integration():
    """Test RcloneClient functionality with real rclone."""
    # Skip if rclone not available
    try:
        subprocess.run(["rclone", "version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("rclone not available")

    # Skip if remote not configured
    try:
        result = subprocess.run(
            ["rclone", "listremotes"], capture_output=True, text=True, check=True
        )
        remotes = [
            line.strip().rstrip(":")
            for line in result.stdout.strip().split("\n")
            if line.strip()
        ]
        if "collab-data" not in remotes:
            pytest.skip("collab-data remote not configured")
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("rclone remotes not accessible")

    from collab_env.dashboard.rclone_client import RcloneClient

    # Test client initialization
    client = RcloneClient()
    assert client.remote_name == "collab-data"

    # Test bucket listing (should not fail even if no buckets)
    buckets = client.list_buckets()
    assert isinstance(buckets, list)


@pytest.mark.skip(reason="YML viewer removed from dashboard")
def test_file_viewer_yml():
    """Test YML file viewer functionality."""
    import tempfile
    import os
    from collab_env.dashboard.file_viewers import (
        FileViewerRegistry,
        TextViewer,
    )

    # Test registry creation
    registry = FileViewerRegistry()
    assert registry is not None

    # Test text viewer assignment for YML
    text_viewer = registry.get_viewer("test.yml")
    assert text_viewer is not None
    assert isinstance(text_viewer, TextViewer)

    # Test text rendering with actual file
    test_content = "test: value\nother: 123"

    # Create temporary file for testing
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False
    ) as temp_file:
        temp_file.write(test_content)
        temp_file_path = temp_file.name

    try:
        render_info = text_viewer.render_view(temp_file_path, "test.yml")
        assert render_info["type"] == "text"
        assert "content" in render_info
        assert render_info["content"] == test_content
        assert render_info["language"] == "yaml"
    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)

    # Test editing capabilities
    assert text_viewer.can_edit()


def test_file_viewer_csv():
    """Test CSV file viewer functionality."""
    import tempfile
    import os
    from collab_env.dashboard.file_viewers import (
        FileViewerRegistry,
        TableViewer,
    )

    # Test registry creation
    registry = FileViewerRegistry()
    assert registry is not None

    # Test table viewer assignment for CSV
    table_viewer = registry.get_viewer("test.csv")
    assert table_viewer is not None
    assert isinstance(table_viewer, TableViewer)

    # Test CSV rendering with actual file
    test_content = "header1,header2\nvalue1,value2\nvalue3,value4"

    # Create temporary file for testing
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as temp_file:
        temp_file.write(test_content)
        temp_file_path = temp_file.name

    try:
        render_info = table_viewer.render_view(temp_file_path, "test.csv")
        assert render_info["type"] == "table"
        assert "html" in render_info
        assert "stats" in render_info
        # Check that it parsed the CSV correctly
        assert render_info["stats"]["rows"] == 2
        assert render_info["stats"]["columns"] == 2
        assert render_info["stats"]["column_names"] == ["header1", "header2"]
        # The HTML should contain the table data
        assert "value1" in render_info["html"]
        assert "value2" in render_info["html"]
    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)

    # Test editing capabilities
    assert not table_viewer.can_edit()


def test_session_manager_unit():
    """Test SessionManager unit functionality (without rclone dependency)."""
    from collab_env.dashboard.session_manager import SessionInfo

    # Test SessionInfo dataclass
    session = SessionInfo(name="test_session")
    assert session.name == "test_session"
    assert not session.has_curated
    assert not session.has_processed
    assert not session.is_complete

    # Test with paths
    session_with_data = SessionInfo(
        name="complete_session",
        curated_path="session_001",
        processed_path="session_001",
    )
    assert session_with_data.has_curated
    assert session_with_data.has_processed
    assert session_with_data.is_complete
