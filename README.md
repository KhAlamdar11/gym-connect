# Gym Connect

## Bug report

- add tau to all the cfgs

## Versions


### random

run random tests by modifying n_agents, comm_radius and tau

### Connectivity Battery v0

env-name: ConnectivityBattery-v0

This environment is designed for testing out controllers without inclusion of immitation learning related modules.

Main Features:
- Allows selection of keyboard control, and predefined trajectories (circle and ellipse (randomized))
- Allows selection of classic (Sabattini) controller and battery aware controller
- Full battery management related variables (see cfg)
- Dynamic removal and addition of agents

```bash
python3 test_connectivitybattery_v0.py cfg/connectivity_battery_v0.cfg
```

### Connectivity Battery v1

env-name: ConnectivityBattery-v1

Decoupled controller with same features as above. Here, the controller only outsputs connectivity forces, and the seperation force is computed in the step function. This is done to allow learning only part of the control law instead of all end to end!

```bash
python3 test_connectivitybattery_v0.py cfg/cfg_test.cfg
```

### Connectivity3D-v0

```bash
python3 test_connectivity3d_v0.py cfg/connectivity_3d_v0.cfg
```

## How to run

Test folder contains scripts for running environments.

```bash
cd gym-connect2/gym_connect/tests/
python3 connect_ros4.py cfg/cros_v4.cfg
```


## Setup

1) `cd gym-connect`
2) `pip3 install -e . `

## Dependencies

- `pip install -r /path/to/requirements.txt`



## To use

Include the following code in your Python script:
~~~~
import gym  
import gym_connect
env = gym.make("Connectivity-v0")` 
~~~~
and then use the `env.reset()` and `env.step()` for interfacing with the environment as you would with other OpenAI Gym environments. 
These implementations also include a `env.controller()` function that gives the best current set of actions to be used for imitation learning.

## Acknowledgements

This codebase is structured similarly to and inspired by the work of katetolstaya (https://github.com/katetolstaya/gym-flock).



