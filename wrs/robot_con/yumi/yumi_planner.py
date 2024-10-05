"""
Motion planner for the YuMi
Author: Jeff Mahler
"""
import IPython
import logging
import numpy as np
from robotconn.yumirapid.autolab_core import RigidTransform
from robotconn.yumirapid.yumi_state import YuMiState
from robotconn.yumirapid.yumi_trajectory import YuMiTrajectory
from robotconn.yumirapid.yumi_constants import YuMiConstants as ymc


try:
    import geometry_msgs
    import moveit_commander
    import moveit_msgs
    import rospy
except ImportError:
    logging.error("Unable to load ROS! Will not be able to use client-side motion planner!")

class YuMiMotionPlanner:
    """ Client-side motion planner for the ABB YuMi, based on MoveIt!
    """
    
    def __init__(self, arm='left', planner='RRTstar',
                 goal_pos_tol=0.001,
                 planning_time=0.1,
                 eef_delta=0.01,
                 jump_thresh=0.0):
        self._arm = 'right_arm'
        if arm.lower() == 'left':
            self._arm = 'left_arm'
        self._goal_pos_tol = goal_pos_tol
        self._planning_time = planning_time

        self._init_planning_interface()
        self._update_planning_params()
        self.set_planner(planner)

    def _init_planning_interface(self):
        """ Initializes the MoveIn motion interface """
        self._robot_comm = moveit_commander.RobotCommander()
        self._planning_group = moveit_commander.MoveGroupCommander(self._arm)
        self._planning_group.set_pose_reference_frame(ymc.MOVEIT_PLANNING_REFERENCE_FRAME)

    def _update_planning_params(self):
        """ Updates the parameters of the planner """
        self._planning_group.set_goal_position_tolerance(self._goal_pos_tol)
        self._planning_group.set_planning_time(self._planning_time)

    def set_planner(self, planner):
        """ Sets the motion planner for the YuMi """
        if planner not in ymc.MOVEIT_PLANNER_IDS.keys():
            raise ValueError('Planner %s not supported' %(planner))
        self._planner_id = ymc.MOVEIT_PLANNER_IDS[planner]
        self._planning_group.set_planner_id(self._planner_id)
        
    @property
    def goal_position_tolerance(self):
        return self._goal_pos_tol

    @property
    def planning_time(self):
        return self._planning_time

    @property
    def planner(self):
        return self._planner

    @property
    def planner(self):
        return self._planner

    @goal_position_tolerance.setter
    def goal_position_tolerance(self, tol):
        self._goal_pos_tol = tol
        self._update_planning_params()

    @planning_time.setter
    def planning_time(self, planning_time):
        self._planning_time = planning_time
        self._update_planning_params()

    @planner.setter
    def planner(self, planner):
        self.set_planner(planner)

    def plan_linear_path(self, start_state, start_pose, goal_pose,
                         traj_len=10, eef_delta=0.01, jump_thresh=0.0):
        """
        Plans a linear trajectory between the start and goal pose from the initial joint configuration. The poses should be specified in the frame of reference of the end_type effector tip. Start state must correspond to the start pose - right now this is up to the user.
        
        Parameters
        ----------
        start_state : YuMiState
            initial state of the arm
        start_pose : RigidTransform
            initial pose of end_type effector tip
        goal_pose : RigidTransform
            goal pose of end_type effector tip
        traj_len : int
            number of waypoints to use in motion
        eef_delta : float
            linear_distance for interpolation of the final planner path in cartesian space
        jump_thresh : float
            the threshold for jumping between poses 

        Returns
        -------
        traj : YuMiTrajectory
            the planned trajectory to execute
        """
        # check valid state types
        if not isinstance(start_state, YuMiState):
            raise ValueError('Start state must be specified')

        # check valid pose types
        if not isinstance(start_pose, RigidTransform) or not isinstance(goal_pose, RigidTransform):
            raise ValueError('Start and goal poses must be specified as RigidTransformations')
        # check valid frames
        if start_pose.from_frame != 'grippers' or goal_pose.from_frame != 'grippers' or (start_pose.to_frame != 'world' and start_pose.to_frame != 'base') or (goal_pose.to_frame != 'world' and goal_pose.to_frame != 'base'):
            raise ValueError('Start and goal poses must be from frame \'grippers\' to frame \{\'world\', \'base\'\}')
            
        # set start state of planner
        start_state_msg = moveit_msgs.msg.RobotState()
        start_state_msg.joint_state.name = ['yumi_joint_%d_r' %i for i in range(1,8)]
        if self._arm == 'left_arm':
            start_state_msg.joint_state.name = ['yumi_joint_%d_l' %i for i in range(1,8)]
        start_state_msg.joint_state.position = start_state.in_radians
        self._planning_group.set_start_state(start_state_msg)
     
        # convert start and end_type pose to the planner's reference frame
        start_pose_hand = start_pose * ymc.T_GRIPPER_HAND
        goal_pose_hand = goal_pose * ymc.T_GRIPPER_HAND

        # get waypoints
        pose_traj = start_pose_hand.linear_trajectory_to(goal_pose_hand, traj_len)
        waypoints = [t.pose_msg for t in pose_traj]

        # plan plath
        plan, fraction = self._planning_group.compute_cartesian_path(waypoints, eef_delta, jump_thresh)
        if fraction >= 0.0 and fraction < 1.0:
            logging.warning('Failed to plan full path.')
            return None
        if fraction < 0.0:
            logging.warning('Error while motion path.')
            return None

        # convert from moveit joint traj in radians
        joint_names = plan.joint_trajectory.joint_names
        joint_traj = []
        for t in plan.joint_trajectory.points:
            joints_and_names = zip(joint_names, t.positions)
            joints_and_names.sort(key = lambda x: x[0])
            joint_traj.append(YuMiState([np.rad2deg(x[1]) for x in joints_and_names]))

        return YuMiTrajectory(joint_traj)

    def plan_shortest_path(self, start_state, start_pose, goal_pose, timeout=0.1):
        """
        Plans the shortest path in joint space between the start and goal pose from the initial joint configuration. The poses should be specified in the frame of reference of the end_type effector tip. Start state must correspond to the start pose - right now this is up to the user.
        
        Parameters
        ----------
        start_state : YuMiState
            initial state of the arm
        start_pose : RigidTransform
            initial pose of end_type effector tip
        goal_pose : RigidTransform
            goal pose of end_type effector tip
        timeout : float
            planner timeout (shorthand for setting new motion time then motion)

        Returns
        -------
        traj : YuMiTrajectory
            the planned trajectory to execute
        """
        # check valid state types
        if not isinstance(start_state, YuMiState):
            raise ValueError('Start state must be specified')

        # check valid pose types
        if not isinstance(start_pose, RigidTransform) or not isinstance(goal_pose, RigidTransform):
            raise ValueError('Start and goal poses must be specified as RigidTransformations')

        # check valid frames
        if start_pose.from_frame != 'grippers' or goal_pose.from_frame != 'grippers' or (start_pose.to_frame != 'world' and start_pose.to_frame != 'base') or (goal_pose.to_frame != 'world' and goal_pose.to_frame != 'base'):
            raise ValueError('Start and goal poses must be from frame \'grippers\' to frame \{\'world\', \'base\'\}')
            
        # set motion time
        self.planning_time = timeout

        # set start state of planner
        start_state_msg = moveit_msgs.msg.RobotState()
        start_state_msg.joint_state.name = ['yumi_joint_%d_r' %i for i in range(1,8)]
        if self._arm == 'left_arm':
            start_state_msg.joint_state.name = ['yumi_joint_%d_l' %i for i in range(1,8)]
        start_state_msg.joint_state.position = start_state.in_radians
        self._planning_group.set_start_state(start_state_msg)
     
        # convert start and end_type pose to the planner's reference frame
        start_pose_hand = start_pose * ymc.T_GRIPPER_HAND
        goal_pose_hand = goal_pose * ymc.T_GRIPPER_HAND

        # plan plath
        self._planning_group.set_pose_target(goal_pose_hand.pose_msg)
        plan = self._planning_group.plan()
        if len(plan.joint_trajectory.points) == 0:
            logging.warning('Failed to plan path.')
            return None

        # convert from moveit joint traj in radians
        joint_names = plan.joint_trajectory.joint_names
        joint_traj = []
        for t in plan.joint_trajectory.points:
            joints_and_names = zip(joint_names, t.positions)
            joints_and_names.sort(key = lambda x: x[0])
            joint_traj.append(YuMiState([np.rad2deg(x[1]) for x in joints_and_names]))

        return YuMiTrajectory(joint_traj)
