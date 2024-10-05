import wrs.robot_sim.robots.robot_interface as ri
import wrs.motion.motion_data as motd


class ManipulationData(motd.MotionData):
    def __init__(self, initor):
        if isinstance(initor, ri.RobotInterface):
            super().__init__(robot=initor)
            self._jaw_width_list = []
        elif isinstance(initor, motd.MotionData):
            super().__init__(robot=initor.robot)
            self._jaw_width_list = []
            self.__add__(initor)

    @property
    def sgl_arm_robot(self):
        return self.robot

    @property
    def conf_list(self):
        return self._conf_list

    @property
    def jaw_width_list(self):
        return self._jaw_width_list

    def extend(self, conf_list, mesh_list=None):
        super().extend(conf_list=conf_list, mesh_list=mesh_list)
        self._jaw_width_list += [None] * len(conf_list)

    def update_jaw_width(self, idx, jaw_width):
        if idx > len(self._jaw_width_list) or idx <= -len(self._jaw_width_list):
            raise ValueError("Index out of range for update_jaw_width!")
        else:
            self.jaw_width_list[idx] = jaw_width
            self.robot.backup_state()
            self.robot.goto_given_conf(jnt_values=self.conf_list[idx])
            self.robot.change_jaw_width(jaw_width=jaw_width)
            self._mesh_list[idx] = self.robot.gen_meshmodel()
            self.robot.restore_state()

    def __add__(self, other):
        super().__add__(other=other)
        if isinstance(other, motd.MotionData):
            self._jaw_width_list += [None] * len(other.conf_list)
        else:
            self._jaw_width_list += other.jaw_width_list
        return self

    def __str__(self):
        out_str = super().__str__()
        out_str += ("\n" + "Jaw Width List:\n-----------------------")
        for jaw_width in self._jaw_width_list:
            out_str += "\n" + f"\njaw_width={jaw_width}"
        return out_str
