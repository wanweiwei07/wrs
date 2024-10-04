'''
Please run this program as superuser (sudo)
author: Chenhao
'''

import os
import sys
#################ENVIRONMENT SETUP####################
#
ros_bridge_path = "/home/hlabwrs/Desktop/ros_bridge"
#
######################################################
# ADD dist-packages to python
sys.path.append(ros_bridge_path + "/devel/lib/python2.7/dist-packages")
sys.path.append("/opt/ros/indigo/lib/python2.7/dist-packages")
# Settup the Environment variables
os.environ["ROS_ROOT"] = "/opt/ros/indigo/share/ros"
os.environ["ROS_PACKAGE_PATH"] = ros_bridge_path + "/src:/opt/ros/indigo/share:/opt/ros/indigo/stacks"
os.environ["ROS_MASTER_URI"] = "http://011402P0015.local:11311"
os.environ["ROSLISP_PACKAGE_DIRECTORIES"]= ros_bridge_path+"/devel/share/common-lisp"
os.environ["ROS_DISTRO"]="indigo"
os.environ["ROS_IP"]="10.0.1.3"
os.environ["ROS_ETC_DIR"]= "/opt/ros/indigo/etc/ros"

import rospy
from baxter_interface import Gripper
from baxter_interface import Limb
from baxter_interface import RobotEnable
from baxter_interface import CameraController
import baxter_interface
from copy import copy
import sys
import actionlib
import threading
from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
)
from trajectory_msgs.msg import (
    JointTrajectoryPoint,
)
import cv2 as cv
import cv_bridge

from sensor_msgs.msg import (
    Image,
)

from dynamic_reconfigure.server import Server

from baxter_interface.cfg import (
    PositionJointTrajectoryActionServerConfig,
    VelocityJointTrajectoryActionServerConfig,
    PositionFFJointTrajectoryActionServerConfig,
)
from joint_trajectory_action.joint_trajectory_action import (
    JointTrajectoryActionServer,
)

class Baxter:
    def __init__(self, usegripper = True):
        rospy.init_node("Baxter")
        threading.Thread(target=self.__jointtraj_server).start()
        self.trajrgt = Trajectory("right")
        self.trajlft = Trajectory("left")
        if usegripper:
            self.__right_hand = Gripper("right")
            self.__left_hand = Gripper("left")
            self.__right_hand.calibrate()
            self.__left_hand.calibrate()

        self.__right_limb = Limb("right")
        self.__left_limb = Limb("left")
        self.baxter = RobotEnable()
        self.__enabled = self.baxter.state().enabled
        self.enable_baxter()

        self.img = None

        for camname in ['head_camera', 'left_hand_camera', 'right_hand_camera']:
            try:
                cam = CameraController(camname)
                # cam.resolution = (1280, 800)
                cam.resolution = (320, 200)
            except:
                pass

        print("baxter opened")

    @property
    def enabled(self):
        return self.__enabled

    def __jointtraj_server(self):
        limb = "both"
        rate = 100.0
        mode = "position"
        if mode == 'velocity':
            dyn_cfg_srv = Server(VelocityJointTrajectoryActionServerConfig,
                                 lambda config, level: config)
        elif mode == 'position':
            dyn_cfg_srv = Server(PositionJointTrajectoryActionServerConfig,
                                 lambda config, level: config)
        else:
            dyn_cfg_srv = Server(PositionFFJointTrajectoryActionServerConfig,
                                 lambda config, level: config)
        jtas = []
        if limb == 'both':
            jtas.append(JointTrajectoryActionServer('right', dyn_cfg_srv,
                                                    rate, mode))
            jtas.append(JointTrajectoryActionServer('left', dyn_cfg_srv,
                                                    rate, mode))
        else:
            jtas.append(JointTrajectoryActionServer(limb, dyn_cfg_srv, rate, mode))

        def cleanup():
            for j in jtas:
                j.clean_shutdown()

        rospy.on_shutdown(cleanup)
        print("Running. Ctrl-c to quit")
        rospy.spin()

    def enable_baxter(self):
        print("Opening the baxter")
        if not self.__enabled: self.baxter.enable()
        self.__enabled = self.baxter.state().enabled

    def disable_baxter(self):
        if self.__enabled: self.baxter.disable()
        self.__enabled = self.baxter.state().enabled

    def opengripper(self, armname="rgt"):
        if not self.__enabled: return
        arm = self.__right_hand if armname == "rgt" else self.__left_hand
        if self._is_grippable(arm):
            arm.open()
        else:
            rospy.logwarn(armname+ " have not been calibrated")

    def closegripper(self, armname="rgt"):
        if not self.__enabled: return
        arm = self.__right_hand if armname == "rgt" else self.__left_hand
        if self._is_grippable(arm):
            arm.close()
        else:
            rospy.logwarn(armname + " have not been calibrated")

    def commandgripper(self,pos , armname="rgt"):
        if not self.__enabled: return
        arm = self.__right_hand if armname == "rgt" else self.__left_hand
        if self._is_grippable(arm):
            arm.command_position(position = pos)
        else:
            rospy.logwarn(armname + " have not been calibrated")

    def _is_grippable(self,gripper):
        return (gripper.calibrated() and gripper.ready())

    def currentposgripper(self,armname ="rgt"):
        arm = self.__right_hand if armname == "rgt" else self.__left_hand
        return arm.position()

    def getjnts(self,armname="rgt"):
        if not self.__enabled: return
        limb = self.__right_limb if armname == "rgt" else self.__left_limb
        angles = limb.joint_angles()
        return angles

    def movejnts(self,jnts_dict, speed =.6 , armname="rgt"):
        if not self.__enabled: return
        limb = self.__right_limb if armname == "rgt" else self.__left_limb
        traj = self.trajrgt if armname is "rgt" else self.trajlft
        # limb.set_joint_position_speed(speed)
        currentjnts = [limb.joint_angle(joint) for joint in limb.joint_names()]
        traj.add_point(currentjnts, 0)
        if isinstance(jnts_dict, dict):
            path_angle = [jnts_dict[joint_name] for joint_name in limb.joint_names()]
        else:
            path_angle = jnts_dict
        # print(path_angle)
        traj.add_point(path_angle, speed)
        # limb.move_to_joint_positions(jnts_dict,threshold= baxter_interface.settings.JOINT_ANGLE_TOLERANCE)
        limb.set_joint_position_speed(.3)

    def movejnts_cont(self, jnts_dict_list, speed=.1, armname="rgt"):
        traj = self.trajrgt if armname is "rgt" else self.trajlft
        if not self.__enabled: return
        limb = self.__right_limb if armname == "rgt" else self.__left_limb
        # limb.set_joint_position_speed(.6)
        currentjnts = [limb.joint_angle(joint) for joint in limb.joint_names()]
        traj.add_point(currentjnts, 0)
        rospy.on_shutdown(traj.stop)
        t = speed
        for jnts_dict in jnts_dict_list:
            # limb.move_to_joint_positions(jnts_dict, threshold= baxter_interface.settings.JOINT_ANGLE_TOLERANCE)

            if isinstance(jnts_dict,dict):
                path_angle = [jnts_dict[joint_name] for joint_name in limb.joint_names()]
            else:
                path_angle = jnts_dict
            # print(path_angle)
            traj.add_point(path_angle,t)
            t += speed
        traj.start()
        traj.wait(-1)
        limb.set_joint_position_speed(.3)
        traj.clear()


    def getforce(self,armname="rgt"):
        if not self.__enabled: return
        limb = self.__right_limb if armname == "rgt" else self.__left_limb
        ft = limb.endpoint_effort()
        force = ft['force']
        torque = ft['torque']
        print ft
        return list(force+torque)

    def setscreenimage(self,path):
        # it seems not working
        """
        Send the image located at the specified path to the head
        display on Baxter.

        @param path: path to the image file to load and send
        """
        # rospy.init_node('rsdk_xdisplay_image', anonymous=True)
        print("Changing the photo")
        img = cv.imread(path)
        msg = cv_bridge.CvBridge().cv2_to_imgmsg(img, encoding="bgr8")
        pub = rospy.Publisher('/robot_s/xdisplay', Image, latch=True, queue_size=1)
        pub.publish(msg)
        # Sleep to allow for image to be published.
        rospy.sleep(1)

    def getimage(self,cameraname="left_hand_camera"):
        # ['head_camera', 'left_hand_camera', 'right_hand_camera']
        # if cameraname != self.camera_on:
        #     cam = CameraController(self.camera_on)
        #     cam.close()
        #     cam = CameraController(cameraname)
        #     cam.resolution = (1280, 800)
        #     cam.open()
        #     self.camera_on = cameraname
        print("Getting imag...")
        "rosrun image_view image_view image:=/cameras/left_hand_camera/image"
        "rosrun image_view image_view image:=/cameras/head_camera/image"
        # Instantiate CvBridge
        bridge = cv_bridge.CvBridge()
        def image_callback(msg):
            print("Received an image!")
            try:
                # Convert your ROS Image message to OpenCV2
                cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
            except cv_bridge.CvBridgeError, e:
                print(e)
            else:
                # Save your OpenCV2 image as a jpeg
                self.img = cv_img
                # return cv2_img
        # Define your image topic
        image_topic = "/cameras/"+cameraname+"/image"
        # Set up your subscriber and define its callback
        imgsub = rospy.Subscriber(image_topic, Image, image_callback)
        while self.img is None:
            pass

        imgsub.unregister()
        img = self.img.astype("float")
        self.img = None

        print (img)
        return img

    def __del__(self):
        self.baxter.disable()
        for camname in ['head_camera', 'left_hand_camera', 'right_hand_camera']:
            try:
                cam = CameraController(camname)
                cam.close()
            except:
                pass

class Trajectory(object):
    def __init__(self, limb):
        ns = 'robot_s/limb/' + limb + '/'
        self._client = actionlib.SimpleActionClient(ns + "follow_joint_trajectory", FollowJointTrajectoryAction)
        self._goal = FollowJointTrajectoryGoal()
        self._goal_time_tolerance = rospy.Time(0.1)
        self._goal.goal_time_tolerance = self._goal_time_tolerance
        server_up = self._client.wait_for_server(timeout=rospy.Duration(3.0))
        if not server_up:
            rospy.logerr("Timed out waiting for Joint Trajectory"
                         " Action Server to connect. Start the action server"
                         " before running example.")
            rospy.signal_shutdown("Timed out waiting for Action Server")
            sys.exit(1)
        self.clear(limb)

    def add_point(self, positions, time):
        point = JointTrajectoryPoint()
        point.positions = copy(positions)
        point.time_from_start = rospy.Duration(time)
        self._goal.trajectory.points.append(point)

    def start(self):
        self._goal.trajectory.header.stamp = rospy.Time.now()
        self._client.send_goal(self._goal)

    def stop(self):
        self._client.cancel_goal()

    def wait(self, timeout=15.0):
        self._client.wait_for_result(timeout=rospy.Duration(timeout))

    def result(self):
        return self._client.get_result()

    def clear(self, limb):
        self._goal = FollowJointTrajectoryGoal()
        self._goal.goal_time_tolerance = self._goal_time_tolerance
        self._goal.trajectory.joint_names = [limb + '_' + joint for joint in \
            ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']]

    def stopserver(self):
        print "ddddd"
        self.serverpid.kill()
        print self.serverpid.poll()
        print "server stop"

if __name__ == "__main__":
    # b = Baxter()
    rospy.init_node("afds")
    def server():
        limb = "both"
        rate = 100.0
        mode = "position"
        if mode == 'velocity':
            dyn_cfg_srv = Server(VelocityJointTrajectoryActionServerConfig,
                                 lambda config, level: config)
        elif mode == 'position':
            dyn_cfg_srv = Server(PositionJointTrajectoryActionServerConfig,
                                 lambda config, level: config)
        else:
            dyn_cfg_srv = Server(PositionFFJointTrajectoryActionServerConfig,
                                 lambda config, level: config)
        jtas = []
        if limb == 'both':
            jtas.append(JointTrajectoryActionServer('right', dyn_cfg_srv,
                                                    rate, mode))
            jtas.append(JointTrajectoryActionServer('left', dyn_cfg_srv,
                                                    rate, mode))
        else:
            jtas.append(JointTrajectoryActionServer(limb, dyn_cfg_srv, rate, mode))

        def cleanup():
            for j in jtas:
                j.clean_shutdown()

        rospy.on_shutdown(cleanup)
        print("Running. Ctrl-c to quit")
        rospy.spin()


    server()
    # angles = b.getjnts(armname="left")
    # print (angles)
    # b.movejnts(angles,speed=1,armname="left")
    # a.close()
