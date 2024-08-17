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

    ⚠️ In case of global installation, care must be taken as this repository employs an older version of OpenAI Gym to avoid any version conflicts.

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

The following keys can be used to control the two stubborn agents:

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

Extension of the connectivity controller to 3D.

To run the test script:

```bash
cd gym_connect/tests/
python3 test_connectivity3d_v0.py cfg/cfg_test_3d.cfg
```

⚠️ Some funcionalities, such as spawning of new agents are not well defined in 3D space.

⚠️ Agents in 3D space are more unstable and gains require to be tuned extensively. 

## Custon Usage

To use the environment in your own script:

~~~~
import gym  
import gym_connect
env = gym.make("Connectivity-v0") 
~~~~

`env.reset()` and `env.step()` can be used to interface with the environment, just like with other OpenAI Gym environments. Additionally, the `env.controller()` function computes the connectivity controller output for the agents.

For more examples, consult the test scripts under the ```gym_connect/tests/``` directory.

## Acknowledgements

This codebase is structured similarly to and inspired by the work of katetolstaya (https://github.com/katetolstaya/gym-flock).

## TODOs
- [ ] Update all variables to be object-agnostic (e.g., use "agents" instead of "UAV/robot/base/node") for consistency.
- [ ] Upgrade the lattice generation algorithm to fix the recursion bug (see [Issue](https://github.com/KhAlamdar11/gym-connect/issues/1)).
- [ ] Create a new environment that allows dynamic allotment of stubborn agents, instead of presetting them to 2.
- [ ] Upgrade to support trajectories for multiple stubborn agents simultaneously (currently, only one agent can move in a circle).
- [ ] Upgrade 3D environment to include agent addition strategies in 3D.
- [ ] Upgrade the entire codebase to use Gymnasium.



## References

<a id="ref1"></a>
[1]: K.G. Alamdar, “Connectivity Maintainence for ad-hoc UAV Networks for Multi-robot Missions,'' University of Zagreb, 2024, June.

<a id="ref2"></a>
[2]: L. Sabattini, N. Chopra, and C. Secchi, “Decentralized connectivity maintenance for cooperative control of mobile  robotic systems,” The International Journal of Robotics Research, vol. 32, no. 12, pp. 1411–1423, 2013. https://doi.org/10.1177/0278364913499085


