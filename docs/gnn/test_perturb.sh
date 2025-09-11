#! /bin/bash

NAME=vpluspplus_a
DATA=boid_perturbed
HEADS=1
BATCHSIZE=50
WORKERS_PER_GPU=4

RUNCMD="python /workspace/collab-environment/collab_env/gnn/train.py"
OPTS1="--model-name $NAME --heads $HEADS --noise 0.005 --seeds 0 1 2 3"
OPTS2="--batch-size $BATCHSIZE"
OPTS3="--workers-per-gpu $WORKERS_PER_GPU"

HEADS=3

RUNCMD="python /workspace/collab-environment/collab_env/gnn/train.py"
OPTS1="--model-name $NAME --heads $HEADS --noise 0.005 --seeds 0 1 2 3"
OPTS2="--batch-size $BATCHSIZE"
OPTS3="--workers-per-gpu $WORKERS_PER_GPU"


$RUNCMD $OPTS1 $OPTS2 $OPTS3 --dataset=$DATA

NAME=vpluspplus_a
DATA=boid_food_perturbed
HEADS=3
BATCHSIZE=50
WORKERS_PER_GPU=4

RUNCMD="python /workspace/collab-environment/collab_env/gnn/train.py"
OPTS1="--model-name $NAME --heads $HEADS --noise 0.005 --seeds 0 1 2 3"
OPTS2="--batch-size $BATCHSIZE"
OPTS3="--workers-per-gpu $WORKERS_PER_GPU"


$RUNCMD $OPTS1 $OPTS2 $OPTS3 --dataset=$DATA