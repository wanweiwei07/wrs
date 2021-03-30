import os
import copy
import robotcon.robotiq.program_builder as pb


class RobotiqHE(object):
    """
    Specialized for Robotiq HE
    author: weiwei
    date: 20191211, 20191212
    """

    def __init__(self):
        pblder = pb.ProgramBuilder()
        _this_dir, _ = os.path.split(__file__)
        filpath = os.path.join(_this_dir, "../urscript_eseries", "robotiqhe.script")
        pblder.load_prog(filpath)
        self.original_program = pblder.return_program_to_run()

    def _open_gripper(self, speedpercentage=90, forcepercentage=90):
        """
        open gripper with given speed and force percentage
        :param speedpercentange: 0~100 percent
        :param forcepercentage: 0~100 percent
        :param fingerdistance: 0~50.0
        :return:
        author: weiwei
        date: 20181110
        """
        complete_program = copy.deepcopy(self.original_program)
        complete_program = complete_program.replace("program_replace_speed",
                                                    "rq_set_force_norm(" + str(forcepercentage) + ")")
        complete_program = complete_program.replace("program_replace_force",
                                                    "rq_set_speed_norm(" + str(speedpercentage) + ")")
        complete_program = complete_program.replace("program_replace_command", "rq_open_and_wait()")
        return complete_program

    def _close_gripper(self, speedpercentage=90, forcepercentage=90):
        """
        :param speedpercentange: 0~100 percent
        :param forcepercentage: 0~100 percent
        :param fingerdistance: 0~85 mm
        :return:
        author: weiwei
        date: 20181110
        """
        complete_program = copy.deepcopy(self.original_program)
        complete_program = complete_program.replace("program_replace_speed",
                                                    "rq_set_force_norm(" + str(forcepercentage) + ")")
        complete_program = complete_program.replace("program_replace_force",
                                                    "rq_set_speed_norm(" + str(speedpercentage) + ")")
        complete_program = complete_program.replace("program_replace_command", "rq_close_and_wait()")
        return complete_program

    def return_program_to_run(self, mode="open", speedpercentage=90, forcepercentage=90):
        if mode is "open":
            return self._open_gripper(speedpercentage=speedpercentage, forcepercentage=forcepercentage)
        elif mode is "close":
            return self._close_gripper(speedpercentage=speedpercentage, forcepercentage=forcepercentage)
        else:
            raise ValueError("Mode must be open or close!")
