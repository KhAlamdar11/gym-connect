class KeyboardController:
    def __init__(self,dim=2):

        if dim==2:
            self.key_speed_robot = {
                '6': [1, 0], '4': [-1, 0], '8': [0, 1], '2': [0, -1],
                '9': [0.7, 0.7], '3': [0.7, -0.7], '1': [-0.7, -0.7], '7': [-0.7, 0.7]
            }
            self.key_speed_base = {
                'd': [1, 0], 'a': [-1, 0], 'w': [0, 1], 'x': [0, -1],
                'e': [0.7, 0.7], 'c': [0.7, -0.7], 'z': [-0.7, -0.7], 'q': [-0.7, 0.7]
            }
        else:
            self.key_speed_robot = {
                '6': [1, 0, 0], '4': [-1, 0, 0], '8': [0, 1, 0], '2': [0, -1, 0],
                '9': [0.7, 0.7, 0], '3': [0.7, -0.7, 0], '1': [-0.7, -0.7, 0], '7': [-0.7, 0.7, 0],
                'o': [0.0, 0.0, 1.0], 'l': [0.0, 0.0, -1.0]
            }
            self.key_speed_base = {
                'd': [1, 0, 0], 'a': [-1, 0, 0], 'w': [0, 1, 0], 'x': [0, -1, 0],
                'e': [0.7, 0.7, 0], 'c': [0.7, -0.7, 0], 'z': [-0.7, -0.7, 0], 'q': [-0.7, 0.7, 0],
                'i': [0.0, 0.0, 1.0], 'k': [0.0, 0.0, -1.0]
            }

    def get_robot_keys(self):
        return [ord(k) for k in self.key_speed_robot.keys()]

    def get_base_keys(self):
        return [ord(k) for k in self.key_speed_base.keys()]

    def get_speed_for_key(self, key):
        char = chr(key)
        if char in self.key_speed_robot:
            return self.key_speed_robot[char], True
        elif char in self.key_speed_base:
            return self.key_speed_base[char], False
        return None, None
