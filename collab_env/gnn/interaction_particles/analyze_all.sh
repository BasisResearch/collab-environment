#!/bin/bash

python collab_env/gnn/interaction_particles/analyze_dataset.py simulated_data/runpod/boid_single_species_basic.pt --save-dir analysis/runpod_basic

python collab_env/gnn/interaction_particles/analyze_dataset.py simulated_data/runpod/boid_single_species_weakalignment_large.pt --save-dir analysis/runpod_weakalignment

python collab_env/gnn/interaction_particles/analyze_dataset.py simulated_data/runpod/boid_single_species_independent.pt --save-dir analysis/runpod_independent