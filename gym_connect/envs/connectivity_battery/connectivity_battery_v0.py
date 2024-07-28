import gym
from gym import spaces, error, utils
from gym.utils import seeding
import numpy as np
import configparser
from os import path
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca
import json

from gym_connect.utils.lattice import *
from gym_connect.utils.k_connectivity_controller import KConnectivityController
from gym_connect.utils.vis import *
from gym_connect.utils.utils import *
from gym_connect.utils.config_manager import load_config_params

font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}


class ConnectivityBatteryV0(gym.Env):

    def __init__(self):

        self.mean_pooling = True  # normalize the adjacency matrix by the number of neighbors or not

        # number states per agent``
        self.nx_system = 4
        # number of observations per agent
        self.n_features = 8
        # number of actions per agent
        self.nu = 2

        # default problem parameters
        self.n_agents = 10  # int(config['network_size'])
        self.comm_radius = 2.5  # float(config['comm_radius'])
        self.dt = 0.01  # #float(config['system_dt'])
        self.v_max = 3.0  #  float(config['max_vel_init'])

        # pin specific params
        self.robot_speed = 1
        self.pin_speed_vec = [0.0,0.0]
        self.robot_init_dist = 5.0
        self.robot_speed = 1

        # UAV params
        self.uav_speed = 0.85

        self.fig = None

        # pinned node positions
        self.start_node = np.array([0.1,0.1])
        self.t = 0

        # controller params
        self.fiedler_value_list = []
        self.speed_gain_list = []

        self.path_xy = None

        self.state_values = None

        self.seed()


    def params_from_cfg(self, args):
        load_config_params(self,args)
        

    def reset(self,i=0):

        self.mode_change = True

        self.t = 0

        x = np.zeros((self.n_agents, self.nx_system))
        
        if self.mode == 'random' or 'random' in self.modes:
            self.modes.append('random')
            # self.mode = random.choice(['circle','ellipse'])
            self.mode = 'ellipse'
            self.modes.append(self.mode)
            self.mode_change = True
        elif self.mode == 'keyboard':
            self.mode = 'keyboard'
            self.end_node = self.end_node_init

        if self.mode != 'keyboard':
            _, _, self.end_node = self.get_curve()
        
        vector_se = self.end_node - self.start_node
        magnitude_se = np.linalg.norm(vector_se)
        self.pin_speed_vec = self.robot_speed * vector_se / magnitude_se

        # keep good initialization
        self.mean_vel = np.mean(x[:, 2:4], axis=0)
        self.init_vel = x[:, 2:4]
        self.x = x

        self.x[0,:2] = self.start_node
        self.x[1,:2] = self.end_node

        # create controller
        self.connectivity_controller = KConnectivityController(self.controller_params)

        # return None
        # set initial topo
        p = gen_lattice(self.n_agents-2, self.comm_radius*0.8, self.start_node, self.end_node)
        for i, (x, y) in enumerate(p):
            self.x[i+2, 0] = x
            self.x[i+2, 1] = y

        self.controller()

        self.compute_helpers()

        return (self.state_values, self.state_network)


    def step(self, v):

        '''
        takes desired velocity command of each node and adjusts
        '''

        assert v.shape == (self.n_agents, self.nu)
       
        v =  self.uav_speed * v
        
        # Update velocities
        self.x[:, 2:4] = v[:, :2] 

        # Update positions with in-place addition
        self.x[:, 0:2] += self.x[:, 2:4] * self.dt
        
        # Set start and end nodes
        self.x[0, :2], self.x[1, :2] = self.start_node, self.end_node

        self.compute_helpers()

        # print(self.t, len(self.path_xy))
        # update fiedler value
        self.fiedler_value, _ = self.connectivity_controller.get_fiedler()

        if self.path_xy is not None:
            done = True if self.fiedler_value<=0.1 or self.t > len(self.path_xy) - 3 else False
        else:
            done = True if self.fiedler_value<=0.1 else False

        # update battery levels
        self.decrease_battery()

        # update robot
        if self.mode != 'keyboard':
            self.set_position(self.path_xy[self.t],1)
        self.t+=1

        return (self.state_values, self.state_network), self.instant_cost(), done, {}


    def compute_helpers(self):
        """
        Creates observation vector for IL with DA-GNNs
        """
        self.diff = self.x.reshape((self.n_agents, 1, self.nx_system)) - self.x.reshape((1, self.n_agents, self.nx_system))
        self.r2 =  np.multiply(self.diff[:, :, 0], self.diff[:, :, 0]) + np.multiply(self.diff[:, :, 1], self.diff[:, :, 1])
        np.fill_diagonal(self.r2, np.Inf)

        self.adj_mat = (self.r2 < self.comm_radius).astype(float)

        # Normalize the adjacency matrix by the number of neighbors - results in mean pooling, instead of sum pooling
        n_neighbors = np.reshape(np.sum(self.adj_mat, axis=1), (self.n_agents,1)) # correct - checked this
        n_neighbors[n_neighbors == 0] = 1
        self.adj_mat_mean = self.adj_mat / n_neighbors 

        v_dummy = np.zeros_like(self.diff[:, :, 0])

        self.x_features = np.dstack((v_dummy,
                                    self.diff[:, :, 0],
                                    np.divide(self.diff[:, :, 0], self.r2), 
                                    
                                    v_dummy,
                                    self.diff[:, :, 1],
                                    np.divide(self.diff[:, :, 1], self.r2), 
                                    
                                    v_dummy,
                                    v_dummy))


        self.state_values = np.sum(self.x_features * self.adj_mat.reshape(self.n_agents, self.n_agents, 1), axis=1)
        
      
        self.state_values[:,0] = self.x[:,0]
        self.state_values[:,3] = self.x[:,1]
        
        self.fiedler_value, self.fiedler_vector = self.connectivity_controller.get_fiedler()

        self.state_values[:,6] = self.fiedler_value
        self.state_values[:,7] = create_fiedler_vector(self.fiedler_vector)[:,0]
        
        self.state_values = self.state_values.reshape((self.n_agents, self.n_features))

        if self.mean_pooling:
            self.state_network = self.adj_mat_mean
        else:
            self.state_network = self.adj_mat

        # self.diff = self.x.reshape((self.n_agents, 1, self.nx_system)) - self.x.reshape((1, self.n_agents, self.nx_system))
        # self.r2 =  np.multiply(self.diff[:, :, 0], self.diff[:, :, 0]) + np.multiply(self.diff[:, :, 1], self.diff[:, :, 1])
        # np.fill_diagonal(self.r2, np.Inf)

        # self.adj_mat = (self.r2 < self.comm_radius).astype(float)

        # # Normalize the adjacency matrix by the number of neighbors - results in mean pooling, instead of sum pooling
        # n_neighbors = np.reshape(np.sum(self.adj_mat, axis=1), (self.n_agents,1)) # correct - checked this
        # n_neighbors[n_neighbors == 0] = 1
        # self.adj_mat_mean = self.adj_mat / n_neighbors 

        # self.x_features = np.dstack((self.diff[:, :, 0],
        #                             np.divide(self.diff[:, :, 0], self.r2), 
        #                             self.diff[:, :, 0],
        #                             self.diff[:, :, 0],

        #                             self.diff[:, :, 1],
        #                             np.divide(self.diff[:, :, 1], self.r2), 
        #                             self.diff[:, :, 0],
        #                             self.diff[:, :, 0]))


        # self.state_values = np.sum(self.x_features * self.adj_mat.reshape(self.n_agents, self.n_agents, 1), axis=1)
        
        # d_start = self.start_node - self.x[:,0:2]
        # d_end = self.end_node - self.x[:,0:2]

        # self.state_values[:,0] = self.x[:,0]
        
        # self.state_values[:,2] = d_start[:,0]
        # self.state_values[:,3] = d_end[:,0]
        
        # self.state_values[:,4] = self.x[:,1]
        
        # self.state_values[:,6] = d_start[:,1]
        # self.state_values[:,7] = d_end[:,1]
        
        # self.state_values = self.state_values.reshape((self.n_agents, self.n_features))

        # if self.mean_pooling:
        #     self.state_network = self.adj_mat_mean
        # else:
        #     self.state_network = self.adj_mat

        # Add fiddler value as last input
        # self.state_values = np.hstack((self.state_values, np.zeros((self.state_values.shape[0], 1))))
        # self.state_values[:, 8], _ = self.connectivity_controller.get_fiedler()
        

        # if self.fiedler_value is not None:
        #     self.fiedler_value_list.append(self.fiedler_value[0])
        # np.save('fielder.npy', np.array(self.fiedler_value_list))

        # if self.fiedler_value is not None:
        #     self.speed_gain_list.append(self.speed_gain())
        # np.save('speed.npy', np.array(self.speed_gain_list))

    
    #____________________  Controller  ________________________

    def controller(self):
        u_c = self.connectivity_controller(self.get_positions(),self.battery)
        u_s = self.connectivity_controller.calculate_repulsion_forces(self.get_positions())
        return u_c + u_s

    #____________________  Utils  ________________________

    def update_robot(self, u=[0,0]):
        if self.mode != 'keyboard':
            self.end_node += np.array(u) * self.robot_speed
        else:
            print(self.start_node,self.end_node)
            ang = angle_between_vectors(self.end_node-self.start_node,u)
            if ang<90:
                self.end_node += self.speed_gain() * np.array(u) * self.robot_speed
            else:
                self.end_node += np.array(u) * self.robot_speed


    def update_base(self, u=[0,0]):
        if self.mode != 'keyboard':
            self.end_node += np.array(u) * self.robot_speed
        else:
            ang = angle_between_vectors(self.start_node-self.end_node,u)
            if ang<90:
                self.start_node += self.speed_gain() * np.array(u) * self.robot_speed
            else:
                self.start_node += np.array(u) * self.robot_speed


    def speed_gain(self):
        T2 = 15
        lambda_0 = 0.4 #criticallambda value
        return 1 / (1 + np.exp(-T2 * (self.fiedler_value - lambda_0)))[0]

    def get_curve(self):

        R = self.robot_init_dist
        r = 2*self.robot_init_dist

        dist_per_step = self.robot_speed * self.dt

        if self.mode == 'circle':
            path_len = 2*math.pi*R
            t = np.linspace(0, 2 * np.pi, int(path_len/dist_per_step))
            x, y = R*np.cos(t), R*np.sin(t)

        elif self.mode == 'ellipse':
            # there is no closed form solution for an elipse so estimated (assump: major = minor * 2)
            path_len = 9.68*R
            t = np.linspace(0, 2 * np.pi, int(path_len/dist_per_step))
            x, y = R*np.cos(t), r * np.sin(t)
            
        self.path_xy = np.array([[x, y] for x, y in zip(x, y)])

        # randomization of start and direction
        if self.mode == 'circle':
            self.path_xy = flip_shift(self.path_xy)
        elif 'random' in self.modes and self.mode == 'ellipse':
            self.path_xy = flip_shift(self.path_xy,random_rotate=True)

        return x, y, self.path_xy[0]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    #____________________  Features  ________________________

    def decrease_battery(self):
        
        if 1000 not in self.battery_decay_select:
            self.battery[self.battery_decay_select] -= self.battery_decay_rate
        else:
            self.battery -= self.battery_decay_rate
            # print(self.battery)

        self.battery = relu(self.battery)
        
        agents_to_add = np.nonzero(self.battery < self.critical_battery_level)[0]
        for i in agents_to_add:
            if self.n_agents-3 < int(self.add_uav_limit[0]) or self.fiedler_value <= self.add_uav_limit[1]:
                if not(self.in_motion[i]): 
                    if self.add_uav_criterion == 'nearest_neighbor':
                        self.add_agent(i)
                    elif self.add_uav_criterion == 'near_base':
                        self.add_agent_base(i)
                    else:
                        print("Invalid add_uav_criterion! Agent will not be added.")
                    self.in_motion[i] = True

        agents_to_remove = np.nonzero(self.battery < self.dead_battery_level)[0]

        for i in agents_to_remove:
            self.kill_node_i(i)


    def kill_node_i(self,i):
        '''
        kills node i
        TODO: Modify for allowing for multiple pinned nodes
        '''
    
        self.n_agents -= 1
        if 1000 not in self.battery_decay_select:
            self.battery_decay_select = self.battery_decay_select[self.battery_decay_select != i] - 1
    
        self.x = np.delete(self.x, i+2, axis=0)
        self.battery = np.delete(self.battery, i, axis=0)
        self.in_motion = np.delete(self.in_motion, i, axis=0)


    def kill_node_random(self):
        '''
        kills a random node besides the 2 pinned notes
        TODO: Modify for allowing for multiple pinned nodes
        '''
        i = random.randint(2, self.n_agents-1)
        self.n_agents -= 1
        self.x = np.delete(self.x, i, axis=0)


    def instant_cost(self):  # sum of differences in velocities
        return self.connectivity_controller.get_fiedler()[0][0]


    def get_stats(self):
        stats = {}
        stats['vel_diffs'] = np.sqrt(np.sum(np.power(self.x[:, 2:4] - np.mean(self.x[:, 2:4], axis=0), 2), axis=1))
        stats['min_dists'] = np.min(np.sqrt(self.r2), axis=0)
        return stats


    #____________________  Add agents  _____________________

    def add_agent_i(self,agent,n):
        t = np.linspace(0, 2 * np.pi, 9)   
        # print(t) 
        a, b = 0.4*self.comm_radius*np.cos(t), 0.4*self.comm_radius*np.sin(t)
        a += self.x[n,0]
        b += self.x[n,1]
        arc = np.array([[a, b] for a, b in zip(a, b)])

        # shortlist pts
        possible_pts = []
        possible_connections = []
        for j in range(arc.shape[0]):
            allow = True
            cons = 0
            for i in range(self.n_agents):
                if i != n and distance(arc[j],self.x[i]) < self.comm_radius*0.3:
                    allow = False
                    break
                elif i != n and i!=agent and distance(arc[j],self.x[i]) < self.comm_radius :
                    cons += 1
            if allow:
                possible_pts.append(arc[j])
                possible_connections.append(cons)

        return np.array(possible_pts), np.array(possible_connections)


    def add_agent(self,agent):
        # find its neighbors
        neighbors = np.nonzero(self.state_network[agent, :])[0]
        arc = np.array([])
        poss = np.array([])
        for neigh in neighbors:
            a, pos = self.add_agent_i(agent,neigh)
            arc = svstack([arc, a])
            poss = np.concatenate([poss,pos])
        
        if poss.shape[0]!=0:
            m = np.argmax(poss)
            
            new_agent = np.array([arc[m,0],arc[m,1],0,0])

            self.x = np.vstack([self.x,new_agent])
            self.n_agents += 1
            self.battery = np.append(self.battery, 1.0)
            self.in_motion = np.append(self.in_motion, False)

    def add_agent_base(self,agent):
        t = np.linspace(0, 2 * np.pi, 15)   
        a, b = 0.7*self.comm_radius*np.cos(t), 0.7*self.comm_radius*np.sin(t)
        a += self.x[0,0]
        b += self.x[0,1]
        arc = np.array([[a, b] for a, b in zip(a, b)])

        # shortlist pts
        possible_pts = []
        possible_connections = []
        for j in range(arc.shape[0]):
            allow = True
            cons = 0
            for i in range(self.x.shape[0]):
                if i != 0 and distance(arc[j],self.x[i]) < self.comm_radius*0.7:
                    allow = False
                    break
                elif i != 0 and i!=agent and distance(arc[j],self.x[i]) < self.comm_radius:
                    cons += 1
            if allow:
                possible_pts.append(arc[j])
                possible_connections.append(cons)

        arc, poss = np.array(possible_pts), np.array(possible_connections)
        m = np.argmax(poss)
        
        new_agent = np.array([arc[m,0],arc[m,1],0,0])

        self.x = np.vstack([self.x,new_agent])
        self.n_agents += 1
        self.battery = np.append(self.battery, 1.0)
        self.in_motion = np.append(self.in_motion, False)


    def compute_network_stats(self):
        n = self.n_agents
        average_distances = []

        for i in range(n):
            x_i, y_i = self.x[i, 0], self.x[i, 1]
            distances = np.sqrt((self.x[:, 0] - x_i) ** 2 + (self.x[:, 1] - y_i) ** 2)
            within_radius = distances[(distances <= self.comm_radius) & (distances > 0)]
            if within_radius.size > 0:
                average_distances.append(np.mean(within_radius))

        if len(average_distances) == 0:
            return 0, 0, (0, 0)  # If no distances are within the radius, return zeros

        average_distances = np.array(average_distances)
        mean_distance = np.mean(average_distances)
        std_distance = np.std(average_distances)
        min_distance = np.min(average_distances)
        max_distance = np.max(average_distances)

        return mean_distance, std_distance, min_distance, max_distance


    #____________________  Getters  ________________________

    def get_n_agents(self):
        return self.n_agents
    
    def get_positions(self):
        return self.x[:,:2]
    
    def get_adj_mat(self):
        return self.state_network

    def get_params(self):
        return self.n_agents, self.comm_radius, self.start_node, self.end_node

    def get_fiedler_list(self):
        return self.fiedler_value_list
    
    def get_fiedler(self):
        return self.fiedler_value
    
    #____________________  Setters  ________________________

    def set_battery(self,i,batt_level):
        '''
        sets battery of the ith node to a certain value
        '''
        self.battery[i] = batt_level

    def set_all_batteries(self,arr):
        '''
        sets the battery levels of all uavs
        be careful that battery of pinned nodes is kept const 1
        '''
        arr = np.array(arr)
        assert arr.shape == (self.n_agents-2,) 
        self.battery = arr

        # print(self.battery)

    def set_batteries_random(self,lower_bound,n_lower_bound):
        '''
        sets the battery levels of all uavs
        be careful that battery of pinned nodes is kept const 1
        '''
        pass

    def set_positions(self,p):
        for pos in range(p.shape[0]):
            self.x[pos,0] = p[pos,0]
            self.x[pos,1] = p[pos,1]

        self.start_node = p[0]
        self.end_node = p[1]

        vector_se = self.end_node - self.start_node
        magnitude_se = np.linalg.norm(vector_se)
        self.pin_speed_vec = self.robot_speed * vector_se / magnitude_se

    def set_pin_speed(self,speed):
        self.robot_speed *= speed

    def set_speed(self,speed):
        self.uav_speed = speed

    def set_position(self,pos,idx):        
        self.x[idx,:2] = np.array(pos)   
        # if we are updating start and end nodes
        if idx == 1:        
            self.end_node = self.x[1,:2]
        if idx == 0:        
            self.start_node = self.x[0,:2]

    #____________________  Viz  ________________________

    def render(self,mode='human'):
        """
        Render the environment with agents as points in 2D space
        """
        if self.render_method == 'thesis':
            return render_thesis(self)
        else:
            render_sim(self)
        

    def close(self):
        pass

    





