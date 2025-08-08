# Referencing https://github.com/beneater/boids
#   and https://vergenet.net/~conrad/boids/pseudocode.html
# boid sims in raw units
# the main function is update_boids()

import random
import numpy as np


def update_boids(boids, width, height, species_configs):
    for b in boids:
        cfg = species_configs[b["species"]]
        fly_towards_center(b, boids, cfg)  # Rule1
        avoid_others(b, boids, cfg)  # Rule2
        match_velocity(b, boids, cfg)  # Rule3
        keep_within_bounds(b, width, height, cfg)  # optional
        limit_speed(b, cfg)  # optional, has to be applied last.
    for b in boids:  # update all birds at the same time
        b["x"] += b["dx"]
        b["y"] += b["dy"]


def init_multi_species_boids(
    species_configs, species_counts, width, height, velocity_scale=10, seed=2025
):
    np.random.seed(seed)
    boids = []
    for species, count in species_counts.items():
        for _ in range(count):
            boids.append(
                {
                    "x": random.random() * width,
                    "y": random.random() * height,
                    "dx": random.random() * velocity_scale,
                    "dy": random.random() * velocity_scale,
                    "species": species,
                }
            )
    return boids


def distance(b1, b2):
    return np.hypot(b1["x"] - b2["x"], b1["y"] - b2["y"])


def same_species(b1, b2):
    "boolean, returns if b1 and b2 are the same spieces."
    return b1["species"] == b2["species"]


def fly_towards_center(boid, boids, cfg):
    """
    Rule 1: Boids try to fly towards the centre of mass of neighbouring boids.
    Reference: https://vergenet.net/~conrad/boids/pseudocode.html

    boid: the boid in reference
    boids: all the boids.
    cfg: parameters of the Boid model
    """
    cx = cy = cnt = 0
    for o in boids:
        if o is not boid and same_species(boid, o):
            if distance(boid, o) < cfg["visual_range"]:
                cx += o["x"]
                cy += o["y"]
                cnt += 1
    if cnt:
        boid["dx"] += ((cx / cnt) - boid["x"]) * cfg["centering_factor"]
        boid["dy"] += ((cy / cnt) - boid["y"]) * cfg["centering_factor"]


def avoid_others(boid, boids, cfg):
    """
    Rule 2: Boids try to keep a small distance away from other objects (including other boids).
    Reference: https://vergenet.net/~conrad/boids/pseudocode.html

    boid: the boid in reference
    boids: all the boids.
    cfg: parameters of the Boid model
    """
    mx = my = 0
    for o in boids:
        if o is not boid and same_species(boid, o):
            if distance(boid, o) < cfg["min_distance"]:
                mx += boid["x"] - o["x"]
                my += boid["y"] - o["y"]
    boid["dx"] += mx * cfg["avoid_factor"]
    boid["dy"] += my * cfg["avoid_factor"]


def match_velocity(boid, boids, cfg):
    """
    Rule 3: Boids try to match velocity with near boids.
    Reference: https://vergenet.net/~conrad/boids/pseudocode.html

    boid: the boid in reference
    boids: all the boids.
    cfg: parameters of the Boid model
    """
    avx = avy = cnt = 0
    for o in boids:
        if o is not boid and same_species(boid, o):
            if distance(boid, o) < cfg["visual_range"]:
                avx += o["dx"]
                avy += o["dy"]
                cnt += 1
    if cnt:
        boid["dx"] += ((avx / cnt) - boid["dx"]) * cfg["matching_factor"]
        boid["dy"] += ((avy / cnt) - boid["dy"]) * cfg["matching_factor"]


def keep_within_bounds(boid, width, height, cfg):
    m, t = cfg["margin"], cfg["turn_factor"]
    if boid["x"] < m:
        boid["dx"] += t
    if boid["x"] > width - m:
        boid["dx"] -= t
    if boid["y"] < m:
        boid["dy"] += t
    if boid["y"] > height - m:
        boid["dy"] -= t


def limit_speed(boid, cfg):
    """
    It is an optional rule.
    Note that this procedure operates directly on b.velocity, rather than returning an offset vector. It is not used like the other rules; rather, this procedure is called after all the other rules have been applied and before calculating the new position"""
    sp = np.hypot(boid["dx"], boid["dy"])
    if sp > cfg["speed_limit"]:
        f = cfg["speed_limit"] / sp
        boid["dx"] *= f
        boid["dy"] *= f
