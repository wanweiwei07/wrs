import os
import copy
import numpy as np
import robot_con.ur.program_builder as pb


class RobotiqETwoFinger(object):
    """
    Specialized for Robotiq HE
    author: weiwei
    date: 20191211, 20191212
    """

    def __init__(self, type='hande'):
        """
        :param type: 'hande' or '2f85' or '2f140'
        author: weiwei
        date: 20210401
        """
        pblder = pb.ProgramBuilder()
        _this_dir, _ = os.path.split(__file__)
        script = "rtq_eseries_hand.script"
        filpath = os.path.join(_this_dir, "../urscripts_eseries", script)
        pblder.load_prog(filpath)
        self.original_program = pblder.get_program_to_run()
        if type is 'hande':
            self.open_limit = 50.0
            self.original_program = self.original_program.replace("program_replace_open_limit", self.open_limit)
        elif type is '2f85':
            self.open_limit = 85.0
            self.original_program = self.original_program.replace("program_replace_open_limit", self.open_limit)
        elif type is '2f140':
            self.open_limit = 140.0
            self.original_program = self.original_program.replace("program_replace_open_limit", self.open_limit)
        else:
            raise NotImplementedError

    def return_program_to_run(self, speedpercentage=90, forcepercentage=90, fingerdistance=0.0):
        """
        return a program that changes the jawwidth of the gripper with
        given speed percentage, force percentage, and fingerdistance
        :param speedpercentange: 0~100 percent
        :param forcepercentage: 0~100 percent
        :Param fingerdistance: 0.0~self.open_limit
        :return:
        author: weiwei
        date: 20181110, 20210401
        """
        fingerdistance = np.clip(fingerdistance, 0, self.open_limit)
        complete_program = copy.deepcopy(self.original_program)
        complete_program = complete_program.replace("program_replace_speed",
                                                    "rq_set_force_norm(" + str(forcepercentage) + ")")
        complete_program = complete_program.replace("program_replace_force",
                                                    "rq_set_speed_norm(" + str(speedpercentage) + ")")
        complete_program = complete_program.replace("program_replace_command", f'rq_move_mm({fingerdistance})')
        return complete_program
