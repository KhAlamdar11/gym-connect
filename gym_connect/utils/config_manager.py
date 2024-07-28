import json
import math
import numpy as np


def load_config_params(self, args):

    self.n_agents = args.getint('n_agents') 

    self.comm_radius = args.getfloat('comm_radius')

    self.mode = args.get('mode')
    self.modes = [] # to keep track of random mode

    #________________________  UAV control params  ________________________
    self.uav_speed = args.getfloat('uav_speed')
    self.uav_v_max = args.getfloat('uav_v_max')

    #________________________  controller  ________________________
    delta = args.getfloat('delta')
    sigma = math.sqrt(-self.comm_radius/(2*math.log(delta)))
    self.controller_params = {'battery_aware': args.getint('battery_aware'),
                                'sigma':sigma,
                                'range':self.comm_radius,
                                'normalized': args.getint('normalized'),
                                'epsilon': args.getfloat('epsilon'),
                                'gainConnectivity': args.getfloat('gainConnectivity'),
                                'gainRepel': args.getfloat('gainRepel'),
                                'repelThreshold': self.comm_radius*args.getfloat('repelThreshold'),
                                'unweighted': args.getint('unweighted'),
                                'v_max': args.getfloat('uav_v_max'),
                                'critical_battery_level': args.getfloat('critical_battery_level'),
                                'tau': args.getfloat('tau')}

    #________________________  states  ________________________
    # number of observations per agent
    self.n_features = args.getint('n_states')
    # number of actions per agent
    self.nu = args.getint('n_actions')
    self.dt = args.getfloat('dt')


    #________________________  mobile robot params!  ________________________
    self.robot_speed = args.getfloat('robot_speed')
    self.robot_init_dist = args.getfloat('robot_init_dist')
    self.end_node_init = np.array(json.loads(args.get('end_node_init')), dtype=float)
    self.start_node = np.array(json.loads(args.get('start_node_init')), dtype=float)


    #________________________  UAV battery params  ________________________
    initial_batteries_str = args.get('initial_batteries')
    if initial_batteries_str:
        try:
            self.battery = np.array(json.loads(initial_batteries_str), dtype=float)
            if self.battery.shape[0] != self.n_agents - 2:
                print(f"Error: initial_batteries must be an array of size {self.n_agents - 2}. Using default values between 0.5 and 1.0")
                self.battery = np.linspace(0.5, 1.0, self.n_agents-2)
        except json.JSONDecodeError:
            print(f"Invalid value for 'initial_batteries'. Using default values between 0.5 and 1.0")
            self.battery = np.linspace(0.2, 1.0, self.n_agents-2)
    else:
        self.battery = np.linspace(0.5, 1.0, self.n_agents-2)
        print("No initial batteries provided, using default:", self.battery)
    self.battery_decay_rate = args.getfloat('battery_decay_rate')
    self.battery_decay_select = np.array(json.loads(args.get('battery_decay_select')), dtype=int)
    self.critical_battery_level = args.getfloat('critical_battery_level')
    self.dead_battery_level = args.getfloat('dead_battery_level')


    #________________________  UAV addition  ________________________
    self.add_uav_limit = np.array(json.loads(args.get('add_uav_limit')), dtype=float)
    self.add_uav_criterion = args.get('add_uav_criterion')


    #________________________  viz params   ________________________
    self.plot_lim = np.array(json.loads(args.get('plot_lim')), dtype=float)
    self.in_motion = np.zeros(self.n_agents-2, dtype=bool)

    self.render_method = args.get('render_method')