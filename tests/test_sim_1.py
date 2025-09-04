import os

from tests.sim_test_util import sim_check_files


def test_sim_files():
    """

    Returns:

    """

    """
    -- 090225 10:56PM 
    Only do this test if it is not a remote test since storing the video seems to be problematic
    on github. This is an issue that needs to be investigated.  
    """
    remote_test = "CI" in os.environ
    if not remote_test:
        sim_check_files()
