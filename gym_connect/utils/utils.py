import math
import random
import numpy as np

'''
Used for trajectory randomization
'''
def flip_shift(path_xy,random_rotate=False):
    
    # flip
    if random.random() < 0.5:
        path_xy = np.flip(path_xy, axis=0)
    shift_amount = random.randint(0, len(path_xy) - 1)
    
    # randomize starting pos
    path_xy = np.roll(path_xy, shift_amount, axis=0)
    
    # rotate if ellipse
    if random_rotate:
        ang = random.random()*2*math.pi
        s,c = math.sin(ang), math.cos(ang)
        path_xy = path_xy @ np.array([[c,-s],[s,c]])

    return path_xy

# TODO: Add description (can't remember rn)
def create_fiedler_vector(v):
    v_matrix = np.tile(v, (v.size, 1))
    diff_matrix = v_matrix - v_matrix.T
    x = np.sum(diff_matrix**2, axis=1).reshape((-1, 1))
    return x

def relu(a):
    return a * (a>0)

def angle_between_vectors(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))