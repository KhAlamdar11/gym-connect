from gym.envs.registration import register

register(
    id='ConnectivityBattery-v0',
    entry_point='gym_connect.envs.connectivity_battery:ConnectivityBatteryV0',
    max_episode_steps=1000000000,
)

register(
    id='ConnectivityBattery-v1',
    entry_point='gym_connect.envs.connectivity_battery:ConnectivityBatteryV1',
    max_episode_steps=1000000000,
)


register(
    id='Connectivity3D-v0',
    entry_point='gym_connect.envs.connectivity_3d:Connectivity3DV0',
    max_episode_steps=1000000000,
)



