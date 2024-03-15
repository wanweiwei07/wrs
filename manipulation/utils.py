class ManipulationData(object):
    def __init__(self, sgl_arm_robot):
        self.robot = sgl_arm_robot
        self._jaw_width_list = []
        self._conf_list = []
        self._hold_list = []
        self._release_list = []
        self._anime_list = []

    @property
    def sgl_arm_robot(self):
        return self.robot

    @property
    def conf_list(self):
        return self._conf_list

    @property
    def jaw_width_list(self):
        return self._jaw_width_list

    @property
    def hold_list(self):
        return self._hold_list

    @property
    def release_list(self):
        return self._release_list

    def extend(self, conf_list):
        self._conf_list += conf_list
        self._jaw_width_list += [None] * len(conf_list)
        self._hold_list += [None] * len(conf_list)
        self._release_list += [None] * len(conf_list)

    def __add__(self, other):
        if self.robot is other.robot:
            self._jaw_width_list += other.jaw_width_list
            self._conf_list += other.conf_list
            self._hold_list += other.hold_list
            self._release_list += other.release_list
            return self
        else:
            raise ValueError("Manipulation data for different robots cannot be concatenated.")

    def __len__(self):
        return len(self.conf_list)

    def __str__(self):
        return f"""Configuration List:\n-----------------------\n{self._conf_list}\n
                Jaw Width List:\n-----------------------\n{self._jaw_width_list}\n
                Hold List:\n-----------------------\n{self._hold_list}\n
                Release List:\n-----------------------\n{self._release_list}
                """

    def is_available(self):
        return (len(self.conf_list) != 0)
