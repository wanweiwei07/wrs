import os
import copy
import numpy as np
import wrs.robot_con.ur.program_builder as pb


class RobotiqCBTwoFinger(object):
    """
    Specialized for Robotiq HE
    author: weiwei
    date: 20191211, 20191212
    """

    def __init__(self, type='rtq85'):
        """
        :param type: 'hande' or '2f85' or '2f140'
        author: weiwei
        date: 20210401
        """
        pblder = pb.ProgramBuilder()
        _this_dir, _ = os.path.split(__file__)
        script = "rtq_cbseries_hand.script"
        filpath = os.path.join(_this_dir, "../urscripts_cbseries", script)
        pblder.load_prog(filpath)
        self.original_program = pblder.get_program_to_run()
        if type is 'rtq85':
            self.open_limit = 85.0
            self.original_program = self.original_program.replace("program_replace_open_limit", str(self.open_limit))
        elif type is 'rtq140':
            self.open_limit = 140.0
            self.original_program = self.original_program.replace("program_replace_open_limit", str(self.open_limit))
        else:
            raise NotImplementedError

    def get_actuation_program(self, speed_percentage=90, force_percentage=90, finger_distance=0.0):
        """
        return a program that changes the ee_values of the grippers with
        given speed percentage, force percentage, and finger_distance
        :param speed_percentange: 0~100 percent
        :param force_percentage: 0~100 percent
        :Param finger_distance: 0.0~self.open_limit
        :return:
        author: weiwei
        date: 20181110, 20210401
        """
        finger_distance = np.clip(finger_distance, 0, self.open_limit)
        complete_program = copy.deepcopy(self.original_program)
        complete_program = complete_program.replace("program_replace",
                                                    f"rq_set_force_norm({force_percentage})\n"
                                                    f"rq_set_speed_norm({speed_percentage})\n"
                                                    f"rq_move_and_wait_mm({finger_distance})")
        return complete_program

    def get_jaw_width_program(self, pc_server_socket):
        """
        get jaw_width
        """
        socket_name = pc_server_socket.getsockname()
        complete_program = copy.deepcopy(self.original_program)
        complete_program = complete_program.replace("program_replace",
                                                    f"textmsg(\"open connection\")\n"
                                                    f"socket_open(\"{socket_name[0]}\", {socket_name[1]})\n"
                                                    f"socket_send_line(rq_current_pos_mm())")
        return complete_program
