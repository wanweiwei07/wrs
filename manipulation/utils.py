import motion.utils as m_util


class ManipulationData(m_util.MotionData):
    def __init__(self, robot):
        super().__init__(robot=robot)
        self._jaw_width_list = []

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

    def __add__(self, other):
        super().__add__(other=other)
        self._jaw_width_list += other.jaw_width_list
        return self

    def __str__(self):
        out_str = super().__str__()
        out_str += ("\n" + "Jaw Width List:\n-----------------------")
        for jaw_width in self._jaw_width_list:
            out_str += "\n" + f"\njaw_width={jaw_width}"
        return out_str
