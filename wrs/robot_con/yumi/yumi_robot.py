'''
Abstraction for the YuMi Robot
Authors: Jacky Liang
'''
from time import sleep
from wrs.robot_con.yumi.yumi_arm import YuMiArm, YuMiArmFactory
from wrs.robot_con.yumi.yumi_constants import YuMiConstants as YMC

class YuMiRobot:
    """ Interface to both arms of an ABB YuMi robot_s.
    Communicates with the robot_s over Ethernet.
    """

    def __init__(self, ip=YMC.IP, port_l=YMC.PORTS["left"]["server"], port_r=YMC.PORTS["right"]["server"], tcp=YMC.TCP_ABB_GRASP_GRIPPER,
                    include_left=True, include_right=True, debug=YMC.DEBUG,
                    log_pose_histories=False, log_state_histories=False,
                    arm_type='local', ros_namespace = None):
        """Initializes a YuMiRobot

        Parameters
        ----------
            ip : string formatted IP Address, optional
                    IP address of YuMi.
                    Defaults to YuMiConstants.IP
            port_l : int, optional
                    Port of left arm server.
                    Defaults to YuMiConstants.PORT_L
            port_r : int, optional
                    Port of right arm server.
                    Defaults to YuMiConstants.PORT_R
            tcp : RigidTransform, optional
                    Tool Center Point Offset of the endeffectors.
                    Defaults to YuMiConstants.TCP_DEFAULT_GRIPPER
            include_left : bool, optional
                    If True, the left arm is included and instantiated.
                    Defaults to True
            include_right : bool, optional
                    If True, the right arm is included and instantiated.
                    Defaults to True
            debug : bool, optional
                    If True, enables debug mode, so no ethernet socket will be created, and all response messages are faked.
                    Defaults to YuMiConstants.DEBUG
            log_pose_histories : bool, optional
                    If True, uses yumi_history_logger to log pose histories. Enables usage of flush_pose_histories.
                    Defaults to False
            log_state_histories : bool, optional
                    If True, uses yumi_history_logger to log state histories. Enables usage of flush_state_histories.
                    Defaults to False
                    
            ros_namespace : string
                ROS namespace of arm. Used by remote YuMiArm only. If None, namespace is same as current ROS namespace
            arm_type : string
                Type of arm. One of {'local', 'remote'}
            
                'local'  creates local YuMiArm objects that communicates over ethernet. This ignores ros_namespace
            
                'remote' creates YuMiArm objects that communicates over ROS with a server. This ignores ip, port_l, and port_r

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        """
        if not include_left and not include_right:
            raise Exception("Must include one of the arms for YuMiRobot!")

        self.tcp = tcp
        self._arms = []

        if include_left:
            if arm_type == 'local':
                self.left = YuMiArm('left', ip=ip, port=port_l, debug=debug, log_pose_histories=log_pose_histories,
                                    log_state_histories=log_state_histories)
            elif arm_type == 'remote':
                self.left = YuMiArmFactory.YuMiArm('remote', 'left', ros_namespace)
            else:
                raise RuntimeError("arm_type {0} for YuMiArm is not a valid arm end_type".format(arm_type))
            self._arms.append(self.left)
        if include_right:
            if arm_type =='local':
                self.right = YuMiArm('right', ip=ip, port=port_r, debug=debug, log_pose_histories=log_pose_histories,
                                     log_state_histories=log_state_histories)
            elif arm_type == 'remote':
                self.right = YuMiArmFactory.YuMiArm('remote', 'right', ros_namespace)
            else:
                raise RuntimeError("arm_type {0} for YuMiArm is not a valid arm end_type".format(arm_type))
            self._arms.append(self.right)

        self.set_tool(self.tcp)
        self.set_z('fine')

    def reset(self):
        '''Calls the reset function for each instantiated arm object.
        '''
        for arm in self._arms:
            arm.reset()
            
    def start(self):
        '''Calls the start function for each instantiated arm object.
        '''
        for arm in self._arms:
            arm.start()

    def stop(self):
        '''Calls the stop function for each instantiated arm object.
        '''
        for arm in self._arms:
            arm.stop()
        sleep(1)

    def open_grippers(self):
        ''' Calls open_gripper function for each instantiated arm object.
        '''
        if len(self._arms) == 2:
            self.left.open_gripper(wait_for_res=False)
            self.right.open_gripper(wait_for_res=True)
        else:
            self._arms[0].open_gripper()

    def goto_state_sync(self, left_state, right_state):
        '''Commands both arms to go to assigned states in sync. Sync means both
        motions will end_type at the same time.

        Parameters
        ----------
            left_state : YuMiState
                    Target state for left arm
            right_state : YuMiState
                    Target state for right arm

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        YuMiControlException
            If commanded pose triggers any motion errors that are catchable by RAPID sever.
        '''
        if len(self._arms) != 2:
            raise Exception("Cannot goto state sync when not both arms are included!")

        self.left._goto_state_sync(left_state)
        self.right._goto_state_sync(right_state)

    def goto_pose_sync(self, left_pose, right_pose):
        '''Commands both arms to go to assigned poses in sync. Sync means both
        motions will end_type at the same time.

        Parameters
        ----------
            left_pose : RigidTransform
                    Target pose for left arm
            right_pose : RigidTransform
                    Target pose for right arm

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        YuMiControlException
            If commanded pose triggers any motion errors that are catchable by RAPID sever.
        '''
        if len(self._arms) != 2:
            raise Exception("Cannot goto pose sync when not both arms are included!")

        self.left._goto_pose_sync(left_pose)
        self.right._goto_pose_sync(right_pose)

    def set_v(self, n):
        '''Sets speed for both arms using n as the speed number.

        Parameters
        ----------
            n: int
                speed number. If n = 100, then speed will be set to the corresponding v100
                specified in RAPID. Loosely, n is translational speed in milimeters per second

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        '''
        speed_data = YuMiRobot.get_v(n)
        for arm in self._arms:
            arm.set_speed(speed_data)

    def set_z(self, name):
        '''Sets zoning settings for both arms according to name.

        Parameters
        ----------
            name : str
                Name of zone setting. ie: "z10", "z200", "fine"

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        '''
        zone_data = YuMiRobot.get_z(name)
        for arm in self._arms:
            arm.set_zone(zone_data)

    def set_tool(self, pose):
        '''Sets TCP (Tool Center Point) for both arms using given pose as offset

        Parameters
        ----------
            pose : RigidTransform
                Pose of new TCP as offset from the default TCP

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        '''
        for arm in self._arms:
            arm.set_tool(pose)

    def reset_home(self):
        '''Moves both arms to home position

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        YuMiControlException
            If commanded pose triggers any motion errors that are catchable by RAPID sever.
        '''
        if hasattr(self, 'left'):
            self.left.goto_state(YMC.L_HOME_STATE, wait_for_res=True)
        if hasattr(self, 'right'):
            self.right.goto_state(YMC.R_HOME_STATE, wait_for_res=True)

    def calibrate_grippers(self):
        '''Calibrates grippers for instantiated arms.

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        '''
        for arm in self._arms:
            arm.calibrate_gripper()

    @staticmethod
    def construct_speed_data(tra, rot):
        '''Constructs a speed data tuple that's in the same format as ones used in RAPID.

        Parameters
        ----------
            tra : float
                    translational speed (milimeters per second)
            rot : float
                    rotational speed (degrees per second)

        Returns:
            A tuple of correctly formatted speed data: (tra, rotmat, tra, rotmat)
        '''
        return (tra, rot, tra, rot)

    @staticmethod
    def get_v(n):
        '''Gets the corresponding speed data for n as the speed number.

        Parameters
        ----------
            n : int
                    speed number. If n = 100, will return the same speed data as v100 in RAPID

        Returns
        -------
            Corresponding speed data tuple using n as speed number
        '''
        return YuMiRobot.construct_speed_data(n, 500)

    @staticmethod
    def get_z(name):
        '''Gets the corresponding speed data for n as the speed number.

        Parameters
        ----------
            name : str
                    Name of zone setting. ie: "z10", "z200", "fine"

        Returns
        -------
            Corresponding zone data dict to be used in set_z
        '''
        values = YuMiRobot.ZONE_VALUES[name]
        point_motion = 1 if name == 'fine' else 0
        return {
            'point_motion': point_motion,
            'values': values
        }

    @staticmethod
    def construct_zone_data(pzone_tcp, pzone_ori, zone_ori):
        '''Constructs tuple for zone data

        Parameters
        ----------
            pzone_tcp : float
                    path zone size for TCP
            pzone_ori : float
                    path zone size for orientation
            zone_ori : float
                    zone size for orientation

        Returns:
            A tuple of correctly formatted zone data: (pzone_tcp, pzone_ori, zone_ori)
        '''
        return (pzone_tcp, pzone_ori, zone_ori)

    ZONE_VALUES = {
        'fine' : (0,0,0),#these values actually don't matter for fine
        'z0'  : (.3,.3,.03),
        'z1'  : (1,1,.1),
        'z5'  : (5,8,.8),
        'z10' : (10,15,1.5),
        'z15' : (15,23,2.3),
        'z20' : (20,30,3),
        'z30' : (30,45,4.5),
        'z50' : (50,75,7.5),
        'z100': (100,150,15),
        'z200': (200,300,30)
    }

if __name__ == '__main__':
    # logging.getLogger().setLevel(YMC.LOGGING_LEVEL)
    robot = YuMiRobot()
    print(robot.right.get_state())
    print(robot.left.get_state())
    robot.reset_home()