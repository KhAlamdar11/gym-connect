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
    
    
    done = False

    # Configure curses environment
    if mode == 'keyboard':
        stdscr.clear()
        stdscr.nodelay(True)  # Set the window to non-blocking mode
        keyboard_controller = KeyboardController(dim=3)

    while not(done):
        action = env.env.controller()
        _, _, done, _ = env.step(action)
        env.render()

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
            env.reset()
            done = False
    
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
