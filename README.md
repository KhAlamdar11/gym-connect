# Gym Connect

[![Python](https://img.shields.io/badge/Python-3.7%20or%20later-blue.svg)](https://www.python.org/downloads/)
[![OpenAI Gym](https://img.shields.io/badge/OpenAI%20Gym-v0.11.0-blue.svg)](https://github.com/openai/gym)


This repository contains part of material developed for Master's thesis in [<a href="#ref1">1</a>].

The package contains gym environments for battery aware connectivity maintainance algorithms. It allows for initialization of multi-agent formations, keyboard or pre-selected control of stubborn agents, connectivity maintainance algorihtm from [<a href="#ref2">2</a>] and battery aware connectivity maintainance of multi agent systems from [<a href="#ref2">1</a>]. 

<div align="center">
    <img src="/media/classic.gif" alt="testing" height="240">
</div>

<div align="center">
    <img src="/media/near.gif" alt="testing" height="150"><img src="/media/base.gif" alt="testing" height="150">
</div>

## Installation

1. **Dependencies**: 

Install dependencies via the requirements file.

```bash
pip install -r /path/to/requirements.txt
```

In case of global installation, care must be taken as this repository employs an older version of OpenAI Gym to avoid any version conflicts.

2. **Setup the package:**

```bash
cd gym-connect
pip install -e . 
```

## Available Environments

### Connectivity Battery v0

env-name: ConnectivityBattery-v0

This environment is designed for testing out controllers without inclusion of immitation learning related modules.

Main Features:
- Allows selection of keyboard control, and predefined trajectories (circle and ellipse (randomized))
- Allows selection of classic [<a href="#ref2">2</a>] controller and battery aware controller [<a href="#ref1">1</a>]
- Full battery management related variables (see config files)
- Dynamic removal and addition of agents

To run the test script that simulates a stubborn agent moving in random ellipses or circles around the other stubborn agent, use the following commands:

```bash
cd gym_connect/tests/
python3 test_connectivitybattery_v0.py cfg/cfg_test_random.cfg
```

If the stubborn agents are to be controlled manually using the keyboard instead of following a pre-defined trajectory, the same script can be run with a different configuration file:

```bash
cd gym_connect/tests/
python3 test_connectivitybattery_v0.py cfg/cfg_test_keyboard.cfg
```

The following keys can be used to control the two pinned agents:

<div align="center">
    <img src="/media/keyboard.png" alt="testing" height="130">
</div>

**Additional Information**

If `Ctrl+C` is pressed where the script is running, a simulation video will be saved along with some data files. These data files include numpy files for tracking:

- Fiedler value
- Distance measurements
- Number of agents 

Refer to the configuration files for more details on the simulation parameters and settings.

### Connectivity Battery v1

env-name: ConnectivityBattery-v1

Decoupled controller with same features as above. Here, the controller only outsputs connectivity forces, and the seperation force is computed in the step function. This is done to allow learning only part of the control law instead of all end to end.

To run the test script:

```bash
cd gym_connect/tests/
python3 test_connectivitybattery_v1.py cfg/cfg_test_random.cfg
```

### Connectivity3D-v0

env-name: Connectivity3D-v0

Extension to 3D.

To run the test script:

```bash
cd gym_connect/tests/
python3 test_connectivity3d_v0.py cfg/cfg_test_3D.cfg
```


## To use

Include the following code in your Python script:

~~~~
import gym  
import gym_connect
env = gym.make("Connectivity-v0")` 
~~~~

and then use the `env.reset()` and `env.step()` for interfacing with the environment as you would with other OpenAI Gym environments.  These implementations also include a `env.controller()` function that gives the best current set of actions to be used for imitation learning.

For more examples, consult the test scripts under the ```gym_connect/tests/``` dir.

## Acknowledgements

This codebase is structured similarly to and inspired by the work of katetolstaya (https://github.com/katetolstaya/gym-flock).


## References

<a id="ref1"></a>
[1]: Alamdar, K.G., 2022, June. Connectivity Maintainence for ad-hoc UAV Networks for Multi-robot Missions. University of Zagreb.

<a id="ref2"></a>
[2]: L. Sabattini, N. Chopra, and C. Secchi, “Decentralized connectivity maintenance for cooperative control of mobile  robotic systems,” The International Journal of Robotics Research, vol. 32, no. 12, pp. 1411–1423, 2013. https://doi.org/10.1177/0278364913499085


