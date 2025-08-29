#! /bin/bash

MAX_FRAMES=10
N_STEPS=10
METHOD=integrated_gradients
RUNCMD="python collab_env/gnn/explain_gnn_integrated_gradients.py"
OPTS="--method $METHOD --max-frames $MAX_FRAMES --n-steps $N_STEPS --save-data --file-id -1"

$RUNCMD $OPTS \
    --model-path trained_models/boid_single_species_basic/trained_models/boid_single_species_basic_vpluspplus_a_n0.005_h1_vr0.5_s0 \
    --rollout-path=trained_models/boid_single_species_basic/rollouts/ \
    --data-name=boid_single_species_basic

$RUNCMD $OPTS \
    --model-path trained_models/boid_single_species_independent/trained_models/boid_single_species_independent_vpluspplus_a_n0.005_h1_vr0.5_s0 \
    --rollout-path=trained_models/boid_single_species_independent/rollouts/ \
    --data-name=boid_single_species_independent

$RUNCMD $OPTS \
    --model-path trained_models/boid_single_species_weakalignment_large/trained_models/boid_single_species_weakalignment_large_vpluspplus_a_n0.005_h1_vr0.5_s0 \
    --rollout-path=trained_models/boid_single_species_weakalignment_large/rollouts/ \
    --data-name=boid_single_species_weakalignment_large

$RUNCMD $OPTS \
    --model-path trained_models/boid_food_basic/trained_models/boid_food_basic_vpluspplus_a_n0.005_h1_vr0.5_s3 \
    --rollout-path=trained_models/boid_food_basic/rollouts/ \
    --data-name=boid_food_basic

$RUNCMD $OPTS \
    --model-path trained_models/boid_food_independent/trained_models/boid_food_independent_vpluspplus_a_n0.005_h1_vr0.5_s3 \
    --rollout-path=trained_models/boid_food_independent/rollouts/ \
    --data-name=boid_food_independent
