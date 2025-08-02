import os
import shutil
from glob import glob

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

from collab_env.data.file_utils import get_project_root, expand_path

# clear the sim-runs folder from previous tests -- a little dicey
sim_runs_path = expand_path(
            "tests/sim-runs", get_project_root()
        )
print(f'removing path {sim_runs_path}')
shutil.rmtree(sim_runs_path)
print(f'making directory {sim_runs_path}')
os.mkdir(f'{sim_runs_path}')

program_path = expand_path(
            "collab_env/sim/boids/run_boids_simulator.py", get_project_root()
        )

# Test to see that the run_boids_simulator runs successfully with test config file
result = os.system(f'python {program_path} -cf tests/sim_test_config.yaml')
assert(result == 0)

# Test to see that output folder was created
folder_list = glob(f'{sim_runs_path}/*')
assert(len(folder_list) > 0)

# Test to see the proper number of episodes were recorded. Should be 3 episodes.
result_file_list = glob(f'{folder_list[0]}/*.parquet')
assert(len(result_file_list) == 3)

# Test to see the proper number of frames were recorded for the first episode. Should be 5 frames.
df = pq.read_pandas(result_file_list[0], columns=['time', 'type']).to_pandas()
assert(df.max(axis=0)['time'] == 5)

# Test to see the proper number of agents were recorded for the first episode.
# Should be 20 agents over 5 agents, so agent should appear 50 times. 
assert(df['type'].value_counts().get('agent', 0) == 5 * 20)



