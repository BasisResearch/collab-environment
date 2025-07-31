from collab_env.data.gcs_utils import GCSClient
import uuid
import pytest
from collab_env.data.file_utils import expand_path, get_project_root


# fixture
@pytest.fixture
def gcs_client():
    return GCSClient()


TEST_BUCKET_NAME = f"test-bucket-collab-data_{uuid.uuid4()}"


@pytest.fixture
def test_bucket_name():
    return TEST_BUCKET_NAME


@pytest.fixture(scope="session")
def setup_and_teardown_module():
    # Setup code

    client = GCSClient()
    client.create_bucket(TEST_BUCKET_NAME)

    yield

    # Teardown code
    client.delete_bucket(TEST_BUCKET_NAME)


def test_create_folder(gcs_client, test_bucket_name, setup_and_teardown_module):
    # generate a random folder name
    test_folder_name = f"test_folder_{uuid.uuid4()}"
    gcs_client.create_folder(f"{test_bucket_name}/{test_folder_name}")

    # check if the new folder exists via glob
    assert len(gcs_client.glob(f"{test_bucket_name}/{test_folder_name}/*")) == 1

    # remove the folder
    gcs_client.remove_folder(f"{test_bucket_name}/{test_folder_name}/")
    assert len(gcs_client.glob(f"{test_bucket_name}/{test_folder_name}/*")) == 0


def test_upload_file(gcs_client, test_bucket_name, setup_and_teardown_module):
    test_folder_name = f"test_folder_{uuid.uuid4()}"
    gcs_client.create_folder(f"{test_bucket_name}/{test_folder_name}")
    gcs_client.upload_file(
        expand_path("Makefile", get_project_root()),
        f"{test_bucket_name}/{test_folder_name}/",
    )

    # check if the file exists via glob
    assert len(gcs_client.glob(f"{test_bucket_name}/{test_folder_name}/Makefile")) == 1

    # remove the file
    gcs_client.delete_path(f"{test_bucket_name}/{test_folder_name}/Makefile")
    assert len(gcs_client.glob(f"{test_bucket_name}/{test_folder_name}/Makefile")) == 0

    # remove the folder
    gcs_client.remove_folder(f"{test_bucket_name}/{test_folder_name}")
    assert len(gcs_client.glob(f"{test_bucket_name}/{test_folder_name}/*")) == 0


def test_is_folder(gcs_client, test_bucket_name, setup_and_teardown_module):
    """Test the is_folder method"""
    test_folder_name = f"test_folder_{uuid.uuid4()}"

    # Test 1: Check if folder exists after creation
    gcs_client.create_folder(f"{test_bucket_name}/{test_folder_name}")
    assert gcs_client.is_folder(f"{test_bucket_name}/{test_folder_name}")

    # Test 2: Check if folder exists with trailing slash
    assert gcs_client.is_folder(f"{test_bucket_name}/{test_folder_name}/")

    # Test 3: Check non-existent folder
    assert not gcs_client.is_folder(f"{test_bucket_name}/non_existent_folder")

    # Test 4: Check if folder still exists after adding a file
    gcs_client.upload_file(
        expand_path("Makefile", get_project_root()),
        f"{test_bucket_name}/{test_folder_name}/",
    )
    assert gcs_client.is_folder(f"{test_bucket_name}/{test_folder_name}")

    # Test 5: Check folder contents
    contents = gcs_client.list_folder_contents(f"{test_bucket_name}/{test_folder_name}")
    assert len(contents) >= 2  # Should have Makefile and .folder_marker

    # Cleanup
    gcs_client.delete_path(f"{test_bucket_name}/{test_folder_name}/Makefile")
    gcs_client.remove_folder(f"{test_bucket_name}/{test_folder_name}")
    assert not gcs_client.is_folder(f"{test_bucket_name}/{test_folder_name}")
