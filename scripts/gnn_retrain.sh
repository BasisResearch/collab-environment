#!/bin/bash
 
python -m collab_env.gnn.train --dataset boid_food_basic --self_loops --epochs 50 --seeds 5 --batch-size 50 --early-stopping-patience 5 
python -m collab_env.gnn.train --dataset boid_food_basic --self_loops  --seeds 3 --rollout 5
python -m collab_env.gnn.train --dataset boid_food_independent --self_loops --epochs 50 --seeds 5 --batch-size 50 --early-stopping-patience 5
python -m collab_env.gnn.train --dataset boid_food_independent --self_loops  --seeds 3 --rollout 5
python -m collab_env.gnn.train --dataset boid_food_strong --self_loops --epochs 50 --seeds 5 --batch-size 50 --early-stopping-patience 5
python -m collab_env.gnn.train --dataset boid_food_strong --self_loops  --seeds 3 --rollout 5