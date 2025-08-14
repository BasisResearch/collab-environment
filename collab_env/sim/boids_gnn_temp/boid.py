# Referencing https://github.com/beneater/boids
#   and https://vergenet.net/~conrad/boids/pseudocode.html
# boid sims in raw units 
# the main function is update_boids()

import random
import numpy as np

def update_boids(boids, width, height, species_configs):
    for b in boids:
        if "food" in b['species']:
            b['dx'], b['dy'] = 0, 0
            continue
        cfg = species_configs[b['species']]
        fly_towards_center(b,boids,cfg) # Rule1
        avoid_others(b,boids,cfg) # Rule2
        match_velocity(b,boids,cfg) # Rule3
        keep_within_bounds(b,width,height,cfg) #optional
        limit_speed(b,cfg) #optional, has to be applied last.

    for b in boids: #update all birds at the same time
        b['x'] += b['dx']
        b['y'] += b['dy']

def update_boids_with_food(boids, width, height, species_configs):
    """
    Each boid has state variables:
    - satiated, during which period,
        it does not approach food, and only follows the Boid rule outside food zone.
        Initiated food_time < 0
        Terminated by food_time < -threshold
    - eating, during which period,
        it does not move.
        Initiated by near_food == True and not satiated
        Terminated by food_time < 0
    - aux variables:
        near_food
        food_time
    """
    for b in boids:
        if "food" in b['species']:
            b['dx'], b['dy'] = 0, 0
            continue

        cfg = species_configs[b['species']]

        b['food_time'] -= 1

        if b['eating']:
            b['dx'], b['dy'] = 0, 0
        
        if b['satiated']:
            avoid_others(b,boids,cfg) # Rule2
            keep_within_bounds_with_food(b,width,height,cfg) #optional
            limit_speed(b,cfg) #optional, has to be applied last.
        else:
            # Boid, searching for food
            fly_towards_center(b,boids,cfg) # Rule1
            avoid_others(b,boids,cfg) # Rule2
            match_velocity(b,boids,cfg) # Rule3
            approach_food(b,boids,cfg)
            keep_within_bounds_with_food(b,width,height,cfg) #optional
            limit_speed(b,cfg) #optional, has to be applied last.
        
        # food state variable section
        update_state_variables(b,cfg)


    for b in boids: #update all birds at the same time
        b['x'] += b['dx']
        b['y'] += b['dy']

        


def init_multi_species_boids(species_configs, species_counts, width, height,
                             velocity_scale = 10,
                             seed = 2025):
    
    boids = []
    for species, count in species_counts.items():
        if 'food' in species:
            boids.append({
                'species': species,
                'x': species_configs[species]["x"],
                'y': species_configs[species]["y"],
                'dx':0,
                'dy':0})
            continue
        for _ in range(count):
            np.random.seed(seed + _)
            random.seed(seed + _) 
            boids.append({
                'x': random.random() * width,
                'y': random.random() * height,
                'dx': (random.random()-0.5) * velocity_scale,
                'dy': (random.random()-0.5) * velocity_scale,
                'species': species,
                'perching': np.random.binomial(n=1, p=0.2),
                'perching_time': 0,
                'satiated': 0,
                'eating': 0,
                'food_time': 0
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
    if cfg["independent"]:
        return None

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
    if cfg["independent"]:
        return None
        
    avx=avy=cnt=0
    for o in boids:
        if o is not boid and same_species(boid,o):
            if distance(boid,o) < cfg['visual_range']:
                avx+=o['dx']; avy+=o['dy']; cnt+=1
    if cnt:
        boid['dx'] += ((avx/cnt)-boid['dx'])*cfg['matching_factor']
        boid['dy'] += ((avy/cnt)-boid['dy'])*cfg['matching_factor']

def perch(boid, boids, cfg):
    if boid['y'] <= 0 and not boid['perching']:
        start_perching(boid, cfg)
    elif boid['perching'] and boid['perching_time'] < 0:
        end_perching(boid, boids, cfg)
    elif boid['perching'] and boid['perching_time'] > 0:
        pass
    else:
        boid['perching'] = 0

def start_perching(boid, cfg):
    boid['y'] = 0
    boid['dy'] = 0
    boid['dx'] = 0
    boid['perching'] = 1
    boid['perching_time'] = random.random() * cfg['perching_time']

def end_perching(boid, boids, cfg):
    boid['perching'] = 0
    fly_towards_center(boid,boids,cfg) # Rule1
    boid['dx'] = random.random() * cfg["food_visual_range"] * 0.05
    boid['dy'] = random.random() * cfg["food_visual_range"] * 0.05
    avoid_others(boid,boids,cfg) # Rule2
    

def near_food(boid, cfg):
    threshold = cfg["food_eating_range"]
    d = distance(cfg['food'], boid)
    return d <= threshold

def update_state_variables(boid,cfg):
    if boid['eating']:
        if boid['food_time'] < 0 and boid['food_time'] > cfg['hunger_threshold']:
            boid['satiated'] = 1
            boid['eating'] = 0
            initiate_random_direction(boid, cfg)
    elif boid['satiated'] == 1:
        if boid['food_time'] < cfg['hunger_threshold']:
            boid['satiated'] = 0
    elif (not boid['satiated'] and not boid['eating']) and near_food(boid, cfg):
        boid['eating'] = 1
        start_food_time(boid, cfg)

def initiate_random_direction(boid, cfg):
    random.seed()
    theta = np.deg2rad(random.random() * 180)
    #print("theta", theta)
    boid['dx'] = np.cos(theta) * cfg['turn_factor'] * 0.05
    boid['dy'] = np.sin(theta) * cfg['turn_factor'] * 0.05

    
def start_food_time(boid, cfg):
    boid['food_time'] = random.random() * cfg['food_time']

def approach_food(boid, boids, cfg):
    x_food = cfg['food']['x'] #scalar for now
    y_food = cfg['food']['y'] #scalar for now
    gamma = cfg['food_factor']
    dx = x_food - boid['x']
    dy = y_food - boid['y']
    d = distance(cfg['food'], boid)
    if d <= cfg["food_visual_range"]:
        boid['dx'] += dx * gamma
        boid['dy'] += dy * gamma

def keep_within_bounds(boid, width, height, cfg):
    m, t = cfg['margin'], cfg['turn_factor']
    if boid['x'] < m:
        boid['dx'] += t
    if boid['x'] > width-m:
        boid['dx'] -= t
    if boid['y'] < m:
        boid['dy'] += t
    if boid['y']>height-m:
        boid['dy'] -= t
    return None

def keep_within_bounds_with_food(boid, width, height, cfg):
    m, t = cfg['margin'], cfg['turn_factor']
    if boid['x'] < m:
        boid['dx'] += t #* (boid['x'] - 0)
    if boid['x'] > width - m:
        boid['dx'] -= t #* np.abs(width - boid['x'])
    if boid['y'] > height - m:
        boid['dy'] -= t #* np.abs(height - boid['y'])
    if boid['eating']:
        return None
    if boid['y'] < m:
        boid['dy'] += t #* np.abs(height - boid['y'])

    return None

def limit_speed(boid, cfg):
    """
    It is an optional rule.
    Note that this procedure operates directly on b.velocity, rather than returning an offset vector. It is not used like the other rules; rather, this procedure is called after all the other rules have been applied and before calculating the new position"""
    sp = np.hypot(boid['dx'],boid['dy'])
    if sp>cfg['speed_limit']:
        f = cfg['speed_limit']/sp
        boid['dx']*=f; boid['dy']*=f

