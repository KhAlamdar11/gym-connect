import numpy as np
import matplotlib.pyplot as plt
import math
from math import sin, cos
from itertools import combinations
from scipy.optimize import linear_sum_assignment
import random


def assign_and_reorder_positions(current_positions, desired_positions):
    n = current_positions.shape[0]  # Number of robots/positions
    
    # Initialize the cost matrix with zeros
    cost_matrix = np.zeros((n, n))


    
    # Calculate the cost (Euclidean distance) for each robot to each position
    for i in range(n):
        for j in range(n):
            cost_matrix[i, j] = np.linalg.norm(current_positions[i] - desired_positions[j])
    
    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Reorder desired_positions based on the assignment
    reordered_desired_positions = desired_positions[col_ind]
    
    # Return the reordered desired_positions matrix
    return reordered_desired_positions

# Helper function to calculate distance between two points
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def reorder_by_dist(arr,p):
    squared_diff = (arr - p) ** 2
    distances = np.sqrt(np.sum(squared_diff, axis=1))
    sorted_indices = np.argsort(distances)
    return arr[sorted_indices]

def find_updown(pts,start,end):
    up_pts, down_pts = [], []
    m = (end[1] - start[1]) / (end[0] - start[0])
    c = start[1] - m * start[0]
    
    for pt in pts:
        y_line = m * pt[0] + c
        if pt[1] >= y_line:
            up_pts.append(pt)
        else:
            down_pts.append(pt)
    return np.array(up_pts), np.array(down_pts)

def svstack(arrays):
    """
    Stacks arrays vertically, handling empty arrays by ensuring they have the correct number of columns.

    Parameters:
    - arrays: A list of numpy arrays to stack.

    Returns:
    - A vertically stacked numpy array.
    """
    # Filter out completely empty arrays and get the shape of the first non-empty array
    arrays = [np.array(arr) for arr in arrays]
    non_empty_arrays = [arr for arr in arrays if arr.size > 0]
    if not non_empty_arrays:  # if all arrays are empty, return an empty array
        return np.array([])
    
    # Assume all non-empty arrays have the same number of columns
    num_columns = non_empty_arrays[0].shape[1] if len(non_empty_arrays[0].shape) > 1 else 0

    # Reshape empty arrays to have the correct number of columns
    arrays = [arr if arr.size > 0 else np.empty((0, num_columns)) for arr in arrays]

    # Stack the arrays vertically
    return np.vstack(arrays)

# Helper function to interpolate points
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return np.array([x, y])

# Helper function to interpolate points
def interpolate(p1, p2, r):
    d = distance(p1, p2)
    if d < r:
        return []

    # Number of intermediate points needed
    num_points = int(np.floor(d / r))

    # Unit vector in the direction from p1 to p2
    vec = ((p2[0] - p1[0]) / d, (p2[1] - p1[1]) / d)

    r = d / (num_points + 1) 
    
    pts = [(p1[0] + i * vec[0] * r, p1[1] + i * vec[1] * r) for i in range(1, num_points + 1)]

    pts = np.vstack((np.atleast_2d(p1),pts,np.atleast_2d(p2)))

    # Generate points
    return pts

def gen_latt_odd(black_nodes, r, n_layers, config = ['up', 'down']):
    """
    Generates intermediate blue dot positions to ensure that each segment
    between black nodes does not exceed the length r, and plots the result.

    Parameters:
    - black_nodes: List of tuples representing the positions of the black nodes.
    - r: The maximum allowed edge length.

    Returns:
    - List of tuples representing the positions of the blue dots.
    """

    if n_layers == 0:
        return np.array([])

    blue_dots = np.array([])

    up_pts = []
    down_pts = []

    # Iterate over pairs of black nodes
    for i in range(len(black_nodes) - 1):
        p1 = black_nodes[i]
        p2 = black_nodes[i + 1]    
        
        # Calculate the angle
        angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])

        # Generate pts
        # for equilateral triangle
        angle_up = angle + math.radians(60)
        angle_down = angle - math.radians(60)

        up_pts.append(pol2cart(r, angle_up) + p1)
        down_pts.append(pol2cart(r, angle_down) + p1)
    
    up_pts = np.array(up_pts)
    down_pts = np.array(down_pts)

    if 'up' in config:
        blue_dots = svstack([blue_dots, up_pts])
        up_pts = gen_latt_odd(up_pts,r,n_layers-1,['up'])
        blue_dots = svstack([blue_dots, up_pts])
    
    if 'down' in config:
        blue_dots = svstack([blue_dots, down_pts])
        down_pts = gen_latt_odd(down_pts,r,n_layers-1,['down'])
        blue_dots = svstack([blue_dots, down_pts])        

    return blue_dots

def gen_latt_even(black_nodes, r, n_layers, config = ['up', 'down']):
    """
    Generates intermediate blue dot positions to ensure that each segment
    between black nodes does not exceed the length r, and plots the result.

    Parameters:
    - black_nodes: List of tuples representing the positions of the black nodes.
    - r: The maximum allowed edge length.

    Returns:
    - List of tuples representing the positions of the blue dots.
    """

    if n_layers <= 0:
        return np.array([])

    blue_dots = np.array([])

    up_pts = []
    down_pts = []

    # create init 2 even layers
    start = black_nodes[0]
    end = black_nodes[-1]  

    dir_line = np.array(end) - np.array(start)
    dir_orth_line = np.array([-dir_line[1], dir_line[0]])
    dir_orth_line = dir_orth_line / np.linalg.norm(dir_orth_line)
 
    for p in black_nodes[1:-1]:
        up_pts.append(p + dir_orth_line * r/2) 
        down_pts.append(p - dir_orth_line * r/2)  


    if 'up' in config:
        blue_dots = svstack([blue_dots, up_pts+0.1])
        up_pts = gen_latt_odd(up_pts,r,n_layers-1,['up'])
        blue_dots = svstack([blue_dots, up_pts])
    
    if 'down' in config:
        blue_dots = svstack([blue_dots, down_pts-0.1])
        down_pts = gen_latt_odd(down_pts,r,n_layers-1,['down'])
        blue_dots = svstack([blue_dots, down_pts])        


    return blue_dots

def calc_n_layers(n_nodes, n_init_nodes):
    '''
    given number of available nodes, and the straight line nodes, finds 
    possible number of layers
    also finds if the config should be even or odd
    '''

    allowed_nodes = n_nodes 

    n_odd = n_init_nodes - 2
    n_layers_odd = 0
    n_nodes_in_layer = n_init_nodes - 1
    # check how many nodes will be used if we use odd topology
    while True:
        new_nodes = 2 * n_nodes_in_layer
        possible_nodes = n_odd + new_nodes
        # print(f"n_nodes_in_layer: {n_nodes_in_layer}, new_nodes: {new_nodes}")
        # print(f"Posible: {possible_nodes}, Allowed: {allowed_nodes}")
        # print(f"n_init: {n_init_nodes}")
        if possible_nodes <= allowed_nodes:
            n_odd = possible_nodes
            n_layers_odd += 1
            n_nodes_in_layer -= 1
        else:
            break
            
    # check how many nodes will be used if we use even topology
    n_even = 0
    n_layers_even = -1
    n_nodes_in_layer = n_init_nodes - 2 
    # check how many nodes will be used if we use odd topology
    while n_nodes_in_layer>0:
        new_nodes = 2 * n_nodes_in_layer
        possible_nodes = n_even + new_nodes
        # print(f"n_nodes_in_layer: {n_nodes_in_layer}, new_nodes: {new_nodes}")
        # print(f"Posible: {possible_nodes}, Allowed: {allowed_nodes}")
        # print(f"n_init: {n_init_nodes}")
        if possible_nodes <= allowed_nodes:
            n_even = possible_nodes
            n_layers_even += 1
            n_nodes_in_layer -= 1
        else:
            break
        
    # print(f'Odd uses: {n_odd} with {n_layers_odd} layers')
    # print(f'Even uses: {n_even} with {n_layers_even} layers')
    # print('-------------------')

    # if n_even>n_odd and n_layers_even <= 1 and n_layers_even > 0:
    #     print(f'Even chosen.. Used nodes: {n_even}, layers: {n_layers_even}')
    #     return 'even', n_layers_even
    # else:
    #     print(f'Odd chosen.. Used nodes: {n_odd}, layers: {n_layers_odd}')
    #     return 'odd', n_layers_odd
    return 'odd', n_layers_odd


def gen_lattice(n_nodes, r, start, end):
    '''
    returns a lattice in form [[x,y],[x,y],...]
    '''


    # try to create a straight line connection
    black_nodes = interpolate(start,end,r)

    # if straight line not possible, return an error of config not possible

    # calculate how many layers are needed and what mode (even/odd)
    n_init_nodes = black_nodes.shape[0] 
    mode, n_layers = calc_n_layers(n_nodes, n_init_nodes)

    # print(f'mode: {mode}, n_layers: {n_layers}')

    # create the lattice
    # if mode == 'odd':
    #     blue_dot_positions = gen_latt_odd(black_nodes, r, n_layers = n_layers)
    #     if blue_dot_positions.shape[0] <= n_nodes + 4:
    #         blue_dot_positions = gen_latt_odd(black_nodes, r, n_layers = n_layers+1)
    #     nodes = svstack([black_nodes, blue_dot_positions])
    # elif mode == 'even':
    #     nodes = gen_latt_even(black_nodes, r, n_layers = n_layers)

    min_nodes = gen_latt_odd(black_nodes, r, n_layers = n_layers)
    max_nodes = gen_latt_odd(black_nodes, r, n_layers = n_layers + 1)

    # work with last layers
    set1 = set(map(tuple, min_nodes)) 
    set2 = set(map(tuple, max_nodes))
    last_layer_nodes = np.array(list(set2 - set1))
    last_layer_nodes = reorder_by_dist(last_layer_nodes,start)
    # chose closest available nodes to start
    n_last_layer_nodes = n_nodes - min_nodes.shape[0] - black_nodes.shape[0] + 2
    
    if n_last_layer_nodes % 2 == 0:
        last_layer_nodes = reorder_by_dist(last_layer_nodes,start)
        # chose closest available nodes to start
        last_layer_nodes = last_layer_nodes[:n_last_layer_nodes]
    else:
        n_down = n_last_layer_nodes//2
        n_up = n_down + 1
        # print(f"last_layer_nodes: {last_layer_nodes}")
        # print(f"start: {start}")
        # print(f"end: {end}")
        # print(f"last_layer_nodes: {last_layer_nodes}")
        up, down = find_updown(last_layer_nodes,start,end)

        # print('_________________________')
        # print(n_last_layer_nodes)
        # print(down.shape[0],n_down)
        # print(up.shape[0],n_up)
        # print('_________________________')

        up = reorder_by_dist(up,start)
        down = reorder_by_dist(down,start)

        up = up[:n_up]
        down = down[:n_down]

        last_layer_nodes = svstack([up,down])


    # print('-----------')
    # print('black_nodes ', black_nodes.shape[0])
    # print('min_nodes ', min_nodes.shape[0])
    # print('max_nodes ', max_nodes.shape[0])
    # print('n_last_layer_nodes ', n_last_layer_nodes)

    black_nodes = black_nodes[1:-1,:]
    nodes = svstack([black_nodes,min_nodes])
    nodes = svstack([nodes,last_layer_nodes])

    return nodes