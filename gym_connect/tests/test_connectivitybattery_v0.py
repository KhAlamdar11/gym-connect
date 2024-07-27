# for connectivity_classic 0, 1 and 2

from scipy.io import loadmat
import numpy as np
from os import path
import numpy as np
import gym
import gym_connect
import sys
import math
import configparser
import random
import curses
from utils.keyboard_controller import KeyboardController
import imageio
import signal


def run(stdscr,args):
    
    env_name = args.get('env')
    env = gym.make(env_name)
    env.env.params_from_cfg(args)
    
    # Use seed
    seed = args.getint('seed')
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    mode = args.get('mode')

    # Environment resets
    obs = env.reset()
    
    # data variables
    fiedler_list = []
    n_agents_list = []
    stats_list = []

    done = False

    # Configure curses environment
    if mode == 'keyboard':
        stdscr.clear()
        stdscr.nodelay(True)  # Set the window to non-blocking mode
        keyboard_controller = KeyboardController()

    # Set up video writer
    writer = imageio.get_writer('simulation.gif', fps=20)

    # Define the signal handler
    def signal_handler(sig, frame):
        print("Ctrl+C detected! Saving data and exiting...")
        np.save('fiedler_list.npy', np.array(fiedler_list))  # Saves array1 to array1.npy
        np.save('n_agents_list.npy', np.array(n_agents_list))  # Saves array2 to array2.npy
        writer.close()  # Close the writer and finish the video
        sys.exit(0)

    while True:
        if not(done):
            action = env.env.controller()
            _, _, done, _ = env.step(action)
        
        # display and save video
        env.render(mode='human')
        frame = env.render(mode='rgb_array')
        writer.append_data(frame)  # Append frame to video

        # keep track of data
        # fiedler value
        fiedler_list.append(env.env.get_fiedler())
        # number of current agenys
        n_agents_list.append(env.env.get_n_agents())
        # collect distance metrics
        stats_list.append(env.env.compute_network_stats())

        if mode == 'keyboard':
            key = stdscr.getch()
            if key != -1:
                if key == ord('p'):
                    break
                speed, is_robot = keyboard_controller.get_speed_for_key(key)
                if speed:
                    if is_robot:
                        env.env.update_robot(speed)
                    else:
                        env.env.update_base(speed)

            stdscr.refresh()

        if done:
            # env.reset()
            done = False
            np.save('fiedler_list_1.npy', np.array(fiedler_list))  
            np.save('n_agents_list_1.npy', np.array(n_agents_list))  
            np.save('stats_list_1.npy', np.array(stats_list)) 
        
        # Register the signal handler for SIGINT
        signal.signal(signal.SIGINT, signal_handler)


    if mode == 'keyboard':
        stdscr.nodelay(False)


def main():
    fname = sys.argv[1]
    config_file = path.join(path.dirname(__file__), fname)
    print(config_file)
    config = configparser.ConfigParser()
    config.read(config_file) 
    
    # Initialize curses and run the main logic within curses wrapper
    # run(config[config.default_section])
    curses.wrapper(run, config[config.default_section])

if __name__ == "__main__":
    main()