# Referencing https://github.com/beneater/boids
#   and https://vergenet.net/~conrad/boids/pseudocode.html
# boid sims in raw units 
# the main function is update_boids()

import random
import numpy as np

def update_boids(boids, width, height, species_configs):
    for b in boids:
        cfg = species_configs[b['species']]
        fly_towards_center(b,boids,cfg) # Rule1
        avoid_others(b,boids,cfg) # Rule2
        match_velocity(b,boids,cfg) # Rule3
        keep_within_bounds(b,width,height,cfg) #optional
        limit_speed(b,cfg) #optional, has to be applied last.
    for b in boids: #update all birds at the same time
        b['x'] += b['dx']; b['y'] += b['dy']

def update_boids_with_food(boids, width, height, species_configs):
    for b in boids:
        cfg = species_configs[b['species']]
        if b['perching']:
            b['perching_time'] -= 1
        else:
            fly_towards_center(b,boids,cfg) # Rule1
            avoid_others(b,boids,cfg) # Rule2
            match_velocity(b,boids,cfg) # Rule3
            approach_food(b,boids,cfg)
            keep_within_bounds(b,width,height,cfg) #optional
            limit_speed(b,cfg) #optional, has to be applied last.
            
        perch(b,boids,cfg) #food is on the ground
        

    for b in boids: #update all birds at the same time
        if not b['perching']:
            b['x'] += b['dx']; b['y'] += b['dy']
        else:
            b['y'] = 0
            b['dy'] = 0


def init_multi_species_boids(species_configs, species_counts, width, height,
                             velocity_scale = 10,
                             seed = 2025):
    np.random.seed(seed)
    boids = []
    for species, count in species_counts.items():
        if species == 'food':
            continue
        for _ in range(count):
            boids.append({
                'x': random.random() * width,
                'y': random.random() * height,
                'dx': random.random() * velocity_scale,
                'dy': random.random() * velocity_scale,
                'species': species,
                'perching': np.random.binomial(n=1, p=0.2),
                'perching_time': 0
            })
    return boids

def distance(b1, b2):
    return np.hypot(b1['x']-b2['x'], b1['y']-b2['y'])

def same_species(b1, b2):
    "boolean, returns if b1 and b2 are the same spieces."
    return b1['species']==b2['species']

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
        if o is not boid and same_species(boid,o):
            if distance(boid, o) < cfg['visual_range']:
                cx += o['x']; cy += o['y']; cnt+=1
    if cnt:
        boid['dx'] += ((cx/cnt)-boid['x'])*cfg['centering_factor']
        boid['dy'] += ((cy/cnt)-boid['y'])*cfg['centering_factor']

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
        if o is not boid and same_species(boid,o):
            if distance(boid,o) < cfg['min_distance']:
                mx += boid['x']-o['x']
                my += boid['y']-o['y']
    boid['dx'] += mx * cfg['avoid_factor']
    boid['dy'] += my * cfg['avoid_factor']

def match_velocity(boid, boids, cfg):
    """
    Rule 3: Boids try to match velocity with near boids.
    Reference: https://vergenet.net/~conrad/boids/pseudocode.html

    boid: the boid in reference
    boids: all the boids.
    cfg: parameters of the Boid model
    """
    avx=avy=cnt=0
    for o in boids:
        if o is not boid and same_species(boid,o):
            if distance(boid,o) < cfg['visual_range']:
                avx+=o['dx']; avy+=o['dy']; cnt+=1
    if cnt:
        boid['dx'] += ((avx/cnt)-boid['dx'])*cfg['matching_factor']
        boid['dy'] += ((avy/cnt)-boid['dy'])*cfg['matching_factor']

def perch(boid, boids, cfg):
    if boid['y'] < 0 and not boid['perching']:
        boid['y'] = 0
        boid['dy'] = 0
        boid['perching'] = 1
        boid['perching_time'] = random.random() * cfg['perching_time']
    elif boid['perching'] and boid['perching_time'] < 0:
        boid['perching'] = 0
        boid['dx'] += 20
        boid['dy'] += 20
    elif boid['perching'] and boid['perching_time'] > 0:
        pass
    else
        boid['perching'] = 0
        

def approach_food(boid, boids, cfg):
    x_food = cfg['food']['x'] #scalar for now
    y_food = cfg['food']['y'] #scalar for now
    gamma = cfg['food_factor']
    dx = x_food - boid['x']
    dy = y_food - boid['y']
    d = np.sqrt(dx ** 2 + dy ** 2)
    if d <= cfg["visual_range"] * 2:
        boid['dx'] += dx * gamma
        boid['dy'] += dy * gamma

def keep_within_bounds(boid, width, height, cfg):
    m, t = cfg['margin'], cfg['turn_factor']
    if boid['x']<m:           boid['dx'] += t
    if boid['x']>width-m:     boid['dx'] -= t
    if boid['y']<m and not boid['perching']:
        boid['dy'] += t
    if boid['y']>height-m:    boid['dy'] -= t

def limit_speed(boid, cfg):
    """
    It is an optional rule.
    Note that this procedure operates directly on b.velocity, rather than returning an offset vector. It is not used like the other rules; rather, this procedure is called after all the other rules have been applied and before calculating the new position"""
    sp = np.hypot(boid['dx'],boid['dy'])
    if sp>cfg['speed_limit']:
        f = cfg['speed_limit']/sp
        boid['dx']*=f; boid['dy']*=f

