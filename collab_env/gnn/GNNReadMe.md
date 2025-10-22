# GNN Training and Rollouts

## Files

- `train.py` - Main training script
- `gnn.py` - Core GNN training functions
- `gnn_definition.py` - Model definitions (now using **GATv2Conv**)
- `utility.py` - Utility functions
- `graphgym/` - **NEW: GraphGym integration for architecture search**
  - See `graphgym/README.md` for details

## Quick Start
### Quick test
```
python collab_env/gnn/train.py --test
```

### Full training on single dataset
```
python collab_env/gnn/train.py \
    --dataset boid_single_species_basic \
    --workers-per-gpu 3 \
    --seeds 5
```

## Key Features
- **Automatic GPU distribution** - Jobs automatically assigned to least loaded GPU
- **Multiple jobs per GPU** - Run 3-4 training jobs per GPU to maximize utilization
- **Simple and minimal** - No complex frameworks, just concurrent.futures
- **Single dataset focus** - Hyperparameter search on one dataset at a time

## Parameters

Training explores these hyperparameters:
- **Noise levels**: 0, 0.005
- **Heads**: 1, 2, 3
- **Visual ranges**: 0.1, 0.5

## Details: run the GNNs

All GNN scripts are in the folder:

```
collab_env/gnn
```

### Setup

* Activate conda

   We also provide a conda `env.yml` file that can be used to create a conda environment with the necessary dependencies. Run the following command to create the environment:

   ```bash
   conda env create -n collab-environment -f env.yml
   conda activate collab-environment
   ```

* Copy data from Google Cloud to local folder. For example to copy the boid with food dataset to the local folder simulated_data/runpod/:

   
   #### Install rclone (see https://rclone.org/install/)

   #### Configure for GCS access
   ```bash
   rclone config create collab-data "google cloud storage" service_account_file=/path/to/api/key.json
   ```

   #### Copy data
   ```
   rclone collab_simulated_data:boid_food_basic.pt simulated_data/runpod/ --gcs-bucket-policy-only

   rclone collab_simulated_data:boid_food_basic_config.pt simulated_data/runpod/ --gcs-bucket-policy-only
   ```

* Train GNN

  The following code will spawn 10 models with different seeds for each combination of all parameters.

   - **Noise levels**: 0, 0.005
   - **Heads**: 1, 2, 3
   - **Visual ranges**: 0.1, 0.5

   ```bash
   python collab_env/gnn/train.py --dataset boid_food_basic --batch-size 50 --train-size 0.7 --seeds 10
   ```

* Increase GPU utilization (4 jobs per GPU)
   python collab_env/gnn/train.py --workers-per-gpu 4

The training code will save 3 files for each model under the folder trained_models. Each file will start with the save name

- `<dataset>_<model>_n<noise>_h<heads>_vr<visual_range>_s<seed>.pt`

  (1) Trained model weights.

  (2) Training specification, which includes parameters visual_range, training noise level, and training epoch number.

  (3) Model specification, which includes the number of attention heads. 

* Rollout a trained GNN

   If we would like to rollout starting from the 5th frame and do a total of 100 frames of roll-out:

   ```
   collab_env/gnn/train.py --dataset boid_food_basic --batch-size 50 --rollout 5 --total_rollout 100
   ```

## Available Datasets

- `boid_single_species_basic`
- `boid_single_species_independent`
- `boid_food_basic_alignment`
- `boid_food_basic_independent`
- `boid_food_strong`


## GraphGym Architecture Search (NEW!)

We now support systematic GNN architecture search using GraphGym. This allows you to:
- Compare multiple GNN types (GCN, **GATv2**, GIN, SAGE, etc.)
- Search over architecture hyperparameters systematically
- Find the best GNN for your specific task

### Quick Start

**Test installation:**
```bash
python collab_env/gnn/run_graphgym_experiments.py --test
```

**Run architecture search:**
```bash
python collab_env/gnn/run_graphgym_experiments.py \
    --config configs/graphgym/base/boids_trajectory.yaml \
    --grid configs/graphgym/grids/quick_test.txt \
    --max-workers 4
```

**See full documentation:**
```bash
cat collab_env/gnn/graphgym/README.md
```

### Important: GATv2 vs GAT

The existing custom GNN has been **updated to use GATv2Conv** instead of GATConv.

**Why GATv2?**
- GAT v1: Attention is essentially static
- GAT v2: Attention is truly dynamic and more expressive
- GATv2 consistently outperforms GAT v1

There's no reason to use GAT v1 anymore!

## Kill running processes spawned
```
ps -Af | grep multiproc |  grep '[p]ython -c from multiprocessing.spawn' | awk '{print $2}' | xargs -r kill
```
