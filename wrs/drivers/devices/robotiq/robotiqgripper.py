import logging


class Robotiq_Two_Finger_Gripper(object):
    complete_program = ""
    header = "def myProg():" + "\n"
    end = "\n" + "end_type"
    logger = False

    def __init__(self, type=85):
        self.logger = logging.getLogger("urx")
        self.__reset()
        self.__type = type

    def __reset(self, speed=100, force=50):
        # defining the grippers
        self.complete_program = ""
        self.add_line_to_program("set_analog_inputrange(0, 0)")
        self.add_line_to_program("set_analog_inputrange(1, 0)")
        self.add_line_to_program("set_analog_inputrange(2, 0)")
        self.add_line_to_program("set_analog_inputrange(3, 0)")
        self.add_line_to_program("set_analog_outputdomain(0, 0)")
        self.add_line_to_program("set_analog_outputdomain(1, 0)")
        self.add_line_to_program("set_tool_voltage(0)")
        self.add_line_to_program("set_runstate_outputs([])")
        # self.add_line_to_program("set_payload(1.28)") #1.28 is the weight of the grippers plus the TF sensor in KG
        self.add_line_to_program("socket_close(\"gripper_socket\")")
        # self.add_line_to_program("sleep(1)") #in Robotiq's example they do a wait here... I haven't found it nec
        self.add_line_to_program("socket_open(\"127.0.0.1\",63352,\"gripper_socket\")")
        # self.add_line_to_program("sleep(1)")
        self.add_line_to_program(
            "socket_set_var(\"SPE\"," + str(speed) + ",\"gripper_socket\")")  # Speed 0-255 is valid
        self.add_line_to_program("sync()")
        self.add_line_to_program(
            "socket_set_var(\"FOR\"," + str(force) + ",\"gripper_socket\")")  # Force 0-255 is valid
        self.add_line_to_program("sync()")
        self.add_line_to_program("socket_set_var(\"ACT\",1,\"gripper_socket\")")  # Activate robot_s
        self.add_line_to_program("sync()")
        self.add_line_to_program("socket_set_var(\"GTO\",1,\"gripper_socket\")")
        self.add_line_to_program("sync()")

    def open_gripper(self, speedpercentange=100, forcepercentage=100, fingerdistance=85):
        """
        open grippers with given speed and force percentage

        :param speedpercentange: 0~100 percent
        :param forcepercentage: 0~100 percent
        :return:

        author: weiwei
        date: 20181110
        """

        if fingerdistance > self.__type:
            print("The given opening linear_distance is larger than maximum linear_distance: " + str(self.__type) + ".")
            raise ValueError()

        if speedpercentange > 100 or forcepercentage > 100:
            raise Exception
        speed = round(speedpercentange / 100.0 * 255.0)
        force = round(forcepercentage / 100.0 * 255.0)
        self.complete_program = ""
        self.add_line_to_program("set_analog_inputrange(0, 0)")
        self.add_line_to_program("set_analog_inputrange(1, 0)")
        self.add_line_to_program("set_analog_inputrange(2, 0)")
        self.add_line_to_program("set_analog_inputrange(3, 0)")
        self.add_line_to_program("set_analog_outputdomain(0, 0)")
        self.add_line_to_program("set_analog_outputdomain(1, 0)")
        self.add_line_to_program("set_tool_voltage(0)")
        self.add_line_to_program("set_runstate_outputs([])")
        # self.add_line_to_program("set_payload(1.28)") #1.28 is the weight of the grippers plus the TF sensor in KG
        self.add_line_to_program("socket_close(\"gripper_socket\")")
        # self.add_line_to_program("sleep(1)") #in Robotiq's example they do a wait here... I haven't found it nec
        self.add_line_to_program("socket_open(\"127.0.0.1\",63352,\"gripper_socket\")")
        # self.add_line_to_program("sleep(1)")
        self.add_line_to_program(
            "socket_set_var(\"SPE\"," + str(speed) + ",\"gripper_socket\")")  # Speed 0-255 is valid
        self.add_line_to_program("sync()")
        self.add_line_to_program(
            "socket_set_var(\"FOR\"," + str(force) + ",\"gripper_socket\")")  # Force 0-255 is valid
        self.add_line_to_program("sync()")
        position = round((self.__type - fingerdistance) / self.__type * 255.0)
        self.add_line_to_program(
            "socket_set_var(\"POS\"," + str(position) + ",\"gripper_socket\")")  # 0 is open; range is 0-255
        self.add_line_to_program("sync()")
        self.add_line_to_program("sleep(1)")

    def close_gripper(self, speedpercentange=255, forcepercentage=50):
        """

        :param speedpercentange: 0~100 percent
        :param forcepercentage: 0~100 percent
        :param fingerdistance: 0~85 mm
        :return:

        author: weiwei
        date: 20181110
        """

        if speedpercentange > 100 or forcepercentage > 100:
            raise Exception
        speed = round(speedpercentange / 100.0 * 255.0)
        force = round(forcepercentage / 100.0 * 255.0)
        self.complete_program = ""
        self.add_line_to_program("set_analog_inputrange(0, 0)")
        self.add_line_to_program("set_analog_inputrange(1, 0)")
        self.add_line_to_program("set_analog_inputrange(2, 0)")
        self.add_line_to_program("set_analog_inputrange(3, 0)")
        self.add_line_to_program("set_analog_outputdomain(0, 0)")
        self.add_line_to_program("set_analog_outputdomain(1, 0)")
        self.add_line_to_program("set_tool_voltage(0)")
        self.add_line_to_program("set_runstate_outputs([])")
        # self.add_line_to_program("set_payload(1.28)") #1.28 is the weight of the grippers plus the TF sensor in KG
        self.add_line_to_program("socket_close(\"gripper_socket\")")
        # self.add_line_to_program("sleep(1)") #in Robotiq's example they do a wait here... I haven't found it nec
        self.add_line_to_program("socket_open(\"127.0.0.1\",63352,\"gripper_socket\")")
        # self.add_line_to_program("sleep(1)")
        self.add_line_to_program(
            "socket_set_var(\"SPE\"," + str(speed) + ",\"gripper_socket\")")  # Speed 0-255 is valid
        self.add_line_to_program("sync()")
        self.add_line_to_program(
            "socket_set_var(\"FOR\"," + str(force) + ",\"gripper_socket\")")  # Force 0-255 is valid
        self.add_line_to_program("sync()")
        position = 255
        self.add_line_to_program(
            "socket_set_var(\"POS\"," + str(position) + ",\"gripper_socket\")")  # 255 is closed; range is 0-255
        self.add_line_to_program("sync()")
        self.add_line_to_program("sleep(1)")

    def add_line_to_program(self, new_line):
        if (self.complete_program != ""):
            self.complete_program += "\n"
        self.complete_program += new_line

    def ret_program_to_run(self):
        if (self.complete_program == ""):
            self.logger.debug("robotiq_two_finger_gripper's program is empty")
            return ""

        prog = self.header
        prog += self.complete_program
        prog += self.end
        return prog
