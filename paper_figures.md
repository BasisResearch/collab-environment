# Figures from the paper

## Figure 2

The video alignment was done manually using the scripts in [collab_env/tracking](collab_env/tracking). Detailed instructions appear in [docs/tracking/README.md](docs/tracking/README.md) and an example notebook is available at [docs/tracking/full_pipeline.ipynb](docs/tracking/full_pipeline.ipynb).

## Figure 4

Requires running [docs/alignment/align.ipynb](docs/alignment/align.ipynb) and [docs/alignment/reprojection.ipynb](docs/alignment/reprojection.ipynb) sequentially.

* Default parameters for camera alignment
* The following parameters for reprojection:
    - n_agents = 20
    - n_min_frames = 150

## Figure 5

Instructions on running the boids simulation appear in [docs/sim/README.md](docs/sim/README.md).

## Figure 6

Instructions for training and analyzing GNNs appear in [docs/gnn/GNNReadMe.md](docs/gnn/GNNReadMe.md).

* For Figure 6(B) see [figures/gnn/A-rollout.ipynb](figures/gnn/A-rollout.ipynb)
* For Figure 6(C) see [figures/gnn/B-attention_weights_food.ipynb](figures/gnn/B-attention_weights_food.ipynb)

## Figure 7

See [figures/gnn/appendix-rollout_perturb.ipynb](figures/gnn/appendix-rollout_perturb.ipynb)

## Figure 8

See [figures/gnn/Z-tracked-data.ipynb](figures/gnn/Z-tracked-data.ipynb)
