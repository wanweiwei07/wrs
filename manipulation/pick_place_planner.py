import os
import math
import copy
import pickle
import numpy as np
import basis.data_adapter as da
import modeling.collisionmodel as cm
import motion.optimization_based.incremental_nik as inik
import motion.probabilistic.rrt_connect as rrtc
import manipulation.approach_depart_planner as adp


class PickPlacePlanner(adp.ADPlanner):

    def __init__(self, robot):
        """
        :param object:
        :param robot_helper:
        author: weiwei, hao
        date: 20191122, 20210113
        """
        super().__init__(robot)

    def gen_object_motion(self, component_name, conf_list, obj_pos, obj_rotmat, type='absolute'):
        """
        :param conf_list:
        :param obj_pos:
        :param obj_rotmat:
        :param type: 'absolute' or 'relative'
        :return:
        author: weiwei
        date: 20210125
        """
        objpose_list = []
        if type == 'absolute':
            for _ in conf_list:
                objpose_list.append(rm.homomat_from_posrot(obj_pos, obj_rotmat))
        elif type == 'relative':
            jnt_values_bk = self.rbt.get_jnt_values(component_name)
            for conf in conf_list:
                self.rbt.fk(component_name, conf)
                gl_obj_pos, gl_obj_rotmat = self.rbt.cvt_loc_intcp_to_gl(component_name, obj_pos, obj_rotmat)
                objpose_list.append(rm.homomat_from_posrot(gl_obj_pos, gl_obj_rotmat))
            self.rbt.fk(component_name, jnt_values_bk)
        else:
            raise ValueError('Type must be absolute or relative!')
        return objpose_list

    def find_common_graspids(self,
                             component_name,
                             hnd_name,
                             grasp_info_list,
                             goal_homomat_list,
                             obstacle_list=[],
                             toggle_debug=False):
        """
        find the common collision free and IK feasible graspids
        :param component_name:
        :param hnd_name: a component may have multiple hands
        :param grasp_info_list:
        :param goal_homomat_list: [homomat, ...]
        :param obstacle_list
        :return: [final_available_graspids, intermediate_available_graspids]
        author: weiwei
        date: 20210113, 20210125
        """
        hnd_instance = self.rbt.hnd_dict[hnd_name]
        # start reasoning
        previously_available_graspids = range(len(grasp_info_list))
        intermediate_available_graspids = []
        hndcollided_grasps_num = 0
        ikfailed_grasps_num = 0
        rbtcollided_grasps_num = 0
        jnt_values_bk = self.rbt.get_jnt_values(component_name)
        for goalid, goal_homomat in enumerate(goal_homomat_list):
            goal_pos = goal_homomat[:3, 3]
            goal_rotmat = goal_homomat[:3, :3]
            graspid_and_graspinfo_list = zip(previously_available_graspids,  # need .copy()?
                                             [grasp_info_list[i] for i in previously_available_graspids])
            previously_available_graspids = []
            for graspid, grasp_info in graspid_and_graspinfo_list:
                jaw_width, _, loc_hnd_pos, loc_hnd_rotmat = grasp_info
                gl_hnd_pos = goal_rotmat.dot(loc_hnd_pos) + goal_pos
                gl_hnd_rotmat = goal_rotmat.dot(loc_hnd_rotmat)
                hnd_instance.fix_to(gl_hnd_pos, gl_hnd_rotmat)
                hnd_instance.jaw_to(jaw_width)  # TODO detect a range?
                if not hnd_instance.is_mesh_collided(obstacle_list):  # common graspid without considering robots
                    jnt_values = self.rbt.ik(component_name, gl_hnd_pos, gl_hnd_rotmat)
                    if jnt_values is not None:  # common graspid consdiering robot ik
                        if toggle_debug:
                            hnd_tmp = hnd_instance.copy()
                            hnd_tmp.gen_meshmodel(rgba=[0, 1, 0, .2]).attach_to(base)
                        self.rbt.fk(component_name, jnt_values)
                        is_rbt_collided = self.rbt.is_collided(obstacle_list)  # common graspid consdiering robot cd
                        # TODO is_obj_collided
                        is_obj_collided = False  # common graspid consdiering obj cd
                        if (not is_rbt_collided) and (
                                not is_obj_collided):  # hnd cdfree, robot ikfeasible, robot cdfree
                            if toggle_debug:
                                self.rbt.gen_meshmodel(rgba=[0, 1, 0, .5]).attach_to(base)
                            previously_available_graspids.append(graspid)
                        elif (not is_obj_collided):  # hnd cdfree, robot ikfeasible, robot collided
                            rbtcollided_grasps_num += 1
                            if toggle_debug:
                                self.rbt.gen_meshmodel(rgba=[1, 0, 1, .5]).attach_to(base)
                    else:  # hnd cdfree, robot ik infeasible
                        ikfailed_grasps_num += 1
                        if toggle_debug:
                            hnd_tmp = hnd_instance.copy()
                            hnd_tmp.gen_meshmodel(rgba=[1, .6, 0, .2]).attach_to(base)
                else:  # hnd collided
                    hndcollided_grasps_num += 1
                    if toggle_debug:
                        hnd_tmp = hnd_instance.copy()
                        hnd_tmp.gen_meshmodel(rgba=[1, 0, 1, .2]).attach_to(base)
            intermediate_available_graspids.append(previously_available_graspids.copy())
            print('-----start-----')
            print('Number of collided grasps at goal-' + str(goalid) + ': ', hndcollided_grasps_num)
            print('Number of failed IK at goal-' + str(goalid) + ': ', ikfailed_grasps_num)
            print('Number of collided robots at goal-' + str(goalid) + ': ', rbtcollided_grasps_num)
            print('------end------')
        final_available_graspids = previously_available_graspids
        self.rbt.fk(component_name, jnt_values_bk)
        return final_available_graspids, intermediate_available_graspids

    def gen_moveto_linear(self,
                          component_name,
                          hnd_name,
                          objcm,
                          grasp_info,
                          start_obj_pos,
                          start_obj_rotmat,
                          goal_obj_pos,
                          goal_obj_rotmat,
                          granularity=.03,
                          seed_jnt_values=None):
        objcm = objcm.copy()
        objcm.set_pos(start_obj_pos)
        objcm.set_rotmat(start_obj_rotmat)
        jaw_width, loc_tcp_pos, loc_hnd_pos, loc_hnd_rotmat = grasp_info
        start_hnd_pos = start_obj_rotmat.dot(loc_tcp_pos) + start_obj_pos
        start_hnd_rotmat = start_obj_rotmat.dot(loc_hnd_rotmat)
        goal_hnd_pos = goal_obj_rotmat.dot(loc_tcp_pos) + goal_obj_pos
        goal_hnd_rotmat = goal_obj_rotmat.dot(loc_hnd_rotmat)
        jnt_values_bk = self.rbt.get_jnt_values(component_name)
        self.rbt.fk(self.rbt.ik(component_name, start_hnd_pos, start_hnd_rotmat, seed_conf=seed_jnt_values))
        rel_obj_pos, rel_obj_rotmat = self.rbt.hold(objcm, jaw_width, hnd_name)
        self.rbt.fk(jnt_values_bk)
        conf_list = self.inik_slvr.gen_linear_motion(component_name,
                                                     start_hnd_pos,
                                                     start_hnd_rotmat,
                                                     goal_hnd_pos,
                                                     goal_hnd_rotmat,
                                                     obstacle_list=[],
                                                     granularity=granularity,
                                                     seed_jnt_values=seed_jnt_values)
        jawwidth_list = self.gen_jawwidth_motion(conf_list, jaw_width)
        objpose_list = self.gen_object_motion(component_name, conf_list, rel_obj_pos, rel_obj_rotmat, type='relative')
        self.rbt.release(objcm, jaw_width, hnd_name)
        return conf_list, jawwidth_list, objpose_list

    def gen_moveto_motion(self,
                          component_name,
                          hnd_name,
                          objcm,
                          grasp_info,
                          start_obj_pos,
                          start_obj_rotmat,
                          goal_obj_pos,
                          goal_obj_rotmat,
                          obstacle_list=[],
                          seed_jnt_values=None):
        objcm = objcm.copy()
        objcm.set_pos(start_obj_pos)
        objcm.set_rotmat(start_obj_rotmat)
        jaw_width, loc_tcp_pos, loc_hnd_pos, loc_hnd_rotmat = grasp_info
        start_hnd_pos = start_obj_rotmat.dot(loc_tcp_pos) + start_obj_pos
        start_hnd_rotmat = start_obj_rotmat.dot(loc_hnd_rotmat)
        goal_hnd_pos = goal_obj_rotmat.dot(loc_tcp_pos) + goal_obj_pos
        goal_hnd_rotmat = goal_obj_rotmat.dot(loc_hnd_rotmat)
        jnt_values_bk = self.rbt.get_jnt_values(component_name)
        start_conf = self.rbt.ik(component_name, start_hnd_pos, start_hnd_rotmat, seed_conf=seed_jnt_values)
        goal_conf = self.rbt.ik(component_name, goal_hnd_pos, goal_hnd_rotmat, seed_conf=seed_jnt_values)
        self.rbt.fk(start_conf)
        rel_obj_pos, rel_obj_rotmat = self.rbt.hold(objcm, jaw_width, hnd_name)
        self.rbt.fk(jnt_values_bk)
        conf_list = self.rrtc_plnr.plan(component_name,
                                        start_conf,
                                        goal_conf,
                                        obstacle_list,
                                        ext_dist=.05,
                                        rand_rate=70,
                                        maxtime=300,
                                        component_name=component_name)
        jawwidth_list = self.gen_jawwidth_motion(conf_list, jaw_width)
        objpose_list = self.gen_object_motion(component_name, conf_list, rel_obj_pos, rel_obj_rotmat, type='relative')
        self.rbt.release(objcm, jaw_width, hnd_name)
        return conf_list, jawwidth_list, objpose_list

    def gen_pickup_linear(self,
                          component_name,
                          hnd_name,
                          objcm,
                          grasp_info,
                          goal_obj_pos,
                          goal_obj_rotmat,
                          approach_direction=np.array([0, 0, -1]),
                          approach_distance=.1,
                          approach_jawwidth=.05,
                          depart_direction=np.array([0, 0, 1]),
                          depart_distance=.1,
                          granularity=.03,
                          seed_jnt_values=None):
        """
        an action is a motion primitive
        :param component_name
        :param hnd_name
        :param objcm
        :param grasp_info:
        :param goal_obj_pos:
        :param goal_obj_rotmat:
        :param jlc_name:
        :param approach_direction:
        :param approach_distance:
        :param approach_jawwidth:
        :param depart_direction:
        :param depart_distance:
        :param depart_jawwidth:
        :return: pickup_conf_list, pickup_jawwidth_list
        author: weiwei
        date: 20191122, 20200105, 20210113
        """
        objcm = objcm.copy()
        objcm.set_pos(goal_obj_pos)
        objcm.set_rotmat(goal_obj_rotmat)
        depart_jawwidth, loc_tcp_pos, loc_hnd_pos, loc_hnd_rotmat = grasp_info
        goal_hnd_pos = goal_obj_rotmat.dot(loc_tcp_pos) + goal_obj_pos
        goal_hnd_rotmat = goal_obj_rotmat.dot(loc_hnd_rotmat)
        abs_obj_pos = objcm.get_pos()
        abs_obj_rotmat = objcm.get_rotmat()
        approach_conf_list, approach_jawwidth_list = self.gen_approach_linear(component_name,
                                                                              goal_hnd_pos,
                                                                              goal_hnd_rotmat,
                                                                              approach_direction,
                                                                              approach_distance,
                                                                              approach_jawwidth,
                                                                              granularity,
                                                                              seed_jnt_values)
        approach_objpose_list = self.gen_object_motion(component_name, abs_obj_pos, abs_obj_rotmat)
        if len(approach_conf_list) == 0:
            print('Cannot perform approach action!')
        else:
            rel_obj_pos, rel_obj_rotmat = self.rbt.hold(objcm, depart_jawwidth, hnd_name)
            depart_conf_list, depart_jawwidth_list = self.gen_depart_linear(component_name,
                                                                            goal_hnd_pos,
                                                                            goal_hnd_rotmat,
                                                                            depart_direction,
                                                                            depart_distance,
                                                                            depart_jawwidth,
                                                                            granularity,
                                                                            seed_jnt_values=approach_conf_list[-1])
            depart_objpose_list = self.gen_object_motion(depart_conf_list, rel_obj_pos, rel_obj_rotmat, type='relative')
            self.rbt.release(objcm, depart_jawwidth, hnd_name)
            if len(depart_conf_list) == 0:
                print('Cannot perform depart action!')
            else:
                return approach_conf_list + depart_conf_list, \
                       approach_jawwidth_list + depart_jawwidth_list, \
                       approach_objpose_list + depart_objpose_list

    def gen_pickup_motion(self,
                          component_name,
                          hnd_name,
                          objcm,
                          grasp_info,
                          goal_obj_pos,
                          goal_obj_rotmat,
                          start_conf=None,
                          goal_conf=None,
                          approach_direction=np.array([0, 0, -1]),
                          approach_distance=.1,
                          approach_jawwidth=.05,
                          depart_direction=np.array([0, 0, 1]),
                          depart_distance=.1,
                          granularity=.03,
                          obstacle_list=[],
                          seed_jnt_values=None):
        """
        degenerate into gen_pickup_primitive if both start_conf and goal_conf are None
        :param component_name:
        :param hnd_name:
        :param objcm:
        :param grasp_info:
        :param goal_obj_pos:
        :param goal_obj_rotmat:
        :param start_conf:
        :param goal_conf:
        :param approach_direction:
        :param approach_distance:
        :param approach_jawwidth:
        :param depart_direction:
        :param depart_distance:
        :param depart_jawwidth:
        :param granularity:
        :param obstacle_list:
        :param seed_jnt_values:
        :return: [conf_list, jawwidth_list, objpose_list, rel_oih_pos, rel_oih_rotmat]
        author: weiwei
        date: 20210125
        """
        objcm = objcm.copy()
        objcm.set_pos(goal_obj_pos)
        objcm.set_rotmat(goal_obj_rotmat)
        depart_jawwidth, loc_tcp_pos, loc_hnd_pos, loc_hnd_rotmat = grasp_info
        goal_hnd_pos = goal_obj_rotmat.dot(loc_tcp_pos) + goal_obj_pos
        goal_hnd_rotmat = goal_obj_rotmat.dot(loc_hnd_rotmat)
        abs_obj_pos = objcm.get_pos()
        abs_obj_rotmat = objcm.get_rotmat()
        approach_conf_list, approach_jawwidth_list = self.gen_approach_motion(component_name,
                                                                              goal_hnd_pos,
                                                                              goal_hnd_rotmat,
                                                                              start_conf,
                                                                              approach_direction,
                                                                              approach_distance,
                                                                              approach_jawwidth,
                                                                              granularity,
                                                                              obstacle_list,
                                                                              seed_jnt_values)
        if len(approach_conf_list) == 0:
            print('Cannot perform approach action!')
        else:
            approach_objpose_list = self.gen_object_motion(component_name, approach_conf_list, abs_obj_pos, abs_obj_rotmat)
            jnt_values_bk = self.rbt.get_jnt_values(component_name)
            self.rbt.fk(component_name, approach_conf_list[-1])
            rel_obj_pos, rel_obj_rotmat = self.rbt.hold(objcm, depart_jawwidth, hnd_name)
            depart_conf_list, depart_jawwidth_list = self.gen_depart_motion(component_name,
                                                                            goal_hnd_pos,
                                                                            goal_hnd_rotmat,
                                                                            goal_conf,
                                                                            depart_direction,
                                                                            depart_distance,
                                                                            depart_jawwidth,
                                                                            granularity,
                                                                            obstacle_list,
                                                                            seed_jnt_values=approach_conf_list[-1])
            depart_objpose_list = self.gen_object_motion(component_name, depart_conf_list, rel_obj_pos, rel_obj_rotmat, type='relative')
            self.rbt.release(objcm, depart_jawwidth, hnd_name) # we do not maintain inner states, directly return seqs
            self.rbt.fk(component_name, jnt_values_bk)
            if len(depart_conf_list) == 0:
                print('Cannot perform depart action!')
            else:
                return approach_conf_list + depart_conf_list, \
                       approach_jawwidth_list + depart_jawwidth_list, \
                       approach_objpose_list + depart_objpose_list

    def gen_placedown_linear(self,
                             component_name,
                             hnd_name,
                             objcm,
                             grasp_info,
                             goal_obj_pos,
                             goal_obj_rotmat,
                             down_direction=np.array([0, 0, -1]),
                             down_distance=.1,
                             up_direction=np.array([0, 0, 1]),
                             up_distance=.1,
                             up_jawwidth=.0,
                             granularity=.03,
                             obstacle_list=[],
                             seed_jnt_values=None):
        """
        degenerate into gen_pickup_primitive if both start_conf and goal_conf are None
        :param component_name:
        :param hnd_name:
        :param objcm:
        :param grasp_info:
        :param goal_obj_pos:
        :param goal_obj_rotmat:
        :param start_conf:
        :param goal_conf:
        :param down_direction:
        :param down_distance:
        :param up_direction:
        :param up_distance:
        :param up_jawwidth:
        :param granularity:
        :param obstacle_list:
        :param seed_jnt_values:
        :return:
        author: weiwei
        date: 20210125
        """
        objcm = objcm.copy()
        objcm.set_pos(goal_obj_pos)
        objcm.set_rotmat(goal_obj_rotmat)
        down_jawwidth, loc_tcp_pos, loc_hnd_pos, loc_hnd_rotmat = grasp_info
        start_obj_pos = goal_obj_pos - down_direction * down_distance
        start_obj_rotmat = goal_obj_rotmat
        down_conf_list, down_jawwidth_list, down_objpose_list = self.gen_moveto_linear(component_name,
                                                                                       hnd_name,
                                                                                       objcm,
                                                                                       grasp_info,
                                                                                       start_obj_pos,
                                                                                       start_obj_rotmat,
                                                                                       goal_obj_pos,
                                                                                       goal_obj_rotmat,
                                                                                       down_jawwidth,
                                                                                       granularity,
                                                                                       seed_jnt_values)
        if len(down_conf_list) == 0:
            print('Cannot perform place down action!')
        else:
            goal_hnd_pos = goal_obj_rotmat.dot(loc_tcp_pos) + goal_obj_pos
            goal_hnd_rotmat = goal_obj_rotmat.dot(loc_hnd_rotmat)
            up_conf_list, up_jawwidth_list = self.gen_depart_linear(component_name,
                                                                    goal_hnd_pos,
                                                                    goal_hnd_rotmat,
                                                                    up_direction,
                                                                    up_distance,
                                                                    up_jawwidth,
                                                                    granularity,
                                                                    obstacle_list,
                                                                    seed_jnt_values=down_conf_list[-1])
            up_objpose_list = self.gen_object_motion(up_conf_list, down_objpose_list[-1], down_objpose_list[-1],
                                                     type='absolute')
            if len(up_conf_list) == 0:
                print('Cannot perform depart action!')
            else:
                return down_conf_list + up_conf_list, \
                       down_jawwidth_list + up_jawwidth_list, \
                       down_objpose_list + up_objpose_list

    def gen_placedown_motion(self,
                             component_name,
                             hnd_name,
                             objcm,
                             grasp_info,
                             goal_obj_pos,
                             goal_obj_rotmat,
                             start_conf=None,
                             goal_conf=None,
                             approach_direction=np.array([0, 0, -1]),
                             approach_distance=.1,
                             approach_jawwidth=.05,
                             depart_direction=np.array([0, 0, 1]),
                             depart_distance=.1,
                             depart_jawwidth=.0,
                             granularity=.03,
                             obstacle_list=[],
                             seed_jnt_values=None):
        """
        degenerate into gen_pickup_primitive if both start_conf and goal_conf are None
        :param component_name:
        :param hnd_name:
        :param objcm:
        :param grasp_info:
        :param goal_obj_pos:
        :param goal_obj_rotmat:
        :param start_conf:
        :param goal_conf:
        :param approach_direction:
        :param approach_distance:
        :param approach_jawwidth:
        :param depart_direction:
        :param depart_distance:
        :param depart_jawwidth:
        :param granularity:
        :param obstacle_list:
        :param seed_jnt_values:
        :return:
        author: weiwei
        date: 20210125
        """
        objcm = objcm.copy()
        jnt_values_bk = self.rbt.get_jnt_values(component_name)
        self.rbt.fk(component_name, start_conf)
        jaw_width, loc_tcp_pos, loc_hnd_pos, loc_hnd_rotmat = grasp_info
        gl_obj_pos, gl_obj_rotmat = self.rbt.get_gl_pose_from_hio(loc_hnd_pos, loc_hnd_rotmat, component_name)
        objcm.set_pos(gl_obj_pos)
        objcm.set_rotmat(gl_obj_rotmat)
        rel_obj_pos, rel_obj_rotmat = self.rbt.hold(objcm, jaw_width, hnd_name)
        objcm.set_pos(goal_obj_pos)
        objcm.set_rotmat(goal_obj_rotmat)
        goal_hnd_pos = goal_obj_rotmat.dot(loc_tcp_pos) + goal_obj_pos
        goal_hnd_rotmat = goal_obj_rotmat.dot(loc_hnd_rotmat)
        approach_conf_list, approach_jawwidth_list = self.gen_approach_motion(component_name,
                                                                              goal_hnd_pos,
                                                                              goal_hnd_rotmat,
                                                                              start_conf,
                                                                              approach_direction,
                                                                              approach_distance,
                                                                              approach_jawwidth,
                                                                              granularity,
                                                                              obstacle_list,
                                                                              seed_jnt_values)
        if len(approach_conf_list) == 0:
            print('Cannot perform approach action!')
        else:
            approach_objpose_list = self.gen_object_motion(component_name, approach_conf_list, rel_obj_pos, rel_obj_rotmat, type='relative')
            self.rbt.release(objcm, jaw_width, hnd_name)
            abs_obj_pos = goal_obj_pos
            abs_obj_rotmat = goal_obj_rotmat
            depart_conf_list, depart_jawwidth_list = self.gen_depart_motion(component_name,
                                                                            goal_hnd_pos,
                                                                            goal_hnd_rotmat,
                                                                            goal_conf,
                                                                            depart_direction,
                                                                            depart_distance,
                                                                            depart_jawwidth,
                                                                            granularity,
                                                                            obstacle_list,
                                                                            seed_jnt_values=approach_conf_list[-1])
            depart_objpose_list = self.gen_object_motion(component_name, depart_conf_list, abs_obj_pos, abs_obj_rotmat)
            self.rbt.fk(component_name, jnt_values_bk)
            if len(depart_conf_list) == 0:
                print('Cannot perform depart action!')
            else:
                return approach_conf_list + depart_conf_list, \
                       approach_jawwidth_list + depart_jawwidth_list, \
                       approach_objpose_list + depart_objpose_list

    def gen_pick_and_place_motion(self, candidatepredgidlist, objinithomomat, objgoalhomomat, armname,
                                  rbtinitarmjnts=None, rbtgoalarmjnts=None, finalstate="io",
                                  primitivedirection_init_forward=None, primitivedirection_init_backward=None,
                                  primitivedistance_init_foward=100, premitivedistance_init_backward=100,
                                  primitivedirection_final_forward=None, primitivedirection_final_backward=None,
                                  primitivedistance_final_foward=100, premitivedistance_final_backward=100, obscmlist=[],
                                  userrt=True):
        """
        generate the pick and place motion for the speicified arm
        the rbt init armjnts must be explicitly specified to avoid wrong robot poses

        :param candidatepredgidlist: candidate predefined grasp id list [int, int, ...]
        :param objinithomomat:
        :param objgoalhomomat:
        :param armname:
        :param rbtinitarmjnts, rbtgoalarmjnts: [lftarmjnts, rgtarmjnts], initial robot arm will be used if not set
        :param finalstate: use to orchestrate the motion, could be: "io"->backtoaninitialpose, gripper open, "uo"->backtoup, gripperopen, "gc"->stopatgoal, gripperclose
        :param primitivedirection_init_forward: the vector to move to the object init pose
        :param primitivedirection_init_backward: the vector to move back from the object init pose
        :param primitivedistance_init_forward
        :param primitivedistance_init_backward
        :param primitivedirection_final_forward: the vector to move to the object goal pose
        :param primitivedirection_final_backward: the vector to move back from the object goal pose
        :param primitivedistance_final_forward
        :param primitivedistance_final_backward
        :param userrt: rrt generator
        :param obscmlist:
        :return:

        author: hao, revised by weiwei
        date: 20191122, 20200105
        """

        if rbtinitarmjnts is not None and rbtgoalarmjnts is not None:
            if armname is "rgt":
                if not np.isclose(rbtinitarmjnts[0], rbtgoalarmjnts[0]):
                    print("The lft arm must maintain unmoved during generating ppsglmotion for the rgt arm.")
                    raise ValueError("")
            elif armname is "lft":
                if not np.isclose(rbtinitarmjnts[1], rbtgoalarmjnts[1]):
                    print("The rgt arm must maintain unmoved during generating ppsglmotion for the lft arm.")
                    raise ValueError("")
            else:
                print("Wrong armname. Must be rgt or lft.")
                raise ValueError("")

        rbt = self.rhx.rbt
        hndfa = self.rhx.rgthndfa if armname is "rgt" else self.rhx.lfthndfa
        predefinedgrasps = self.identityglist_rgt if armname is "rgt" else self.identityglist_lft
        bk_armjnts_rgt = rbt.getarmjnts(armname="rgt")
        bk_armjnts_lft = rbt.getarmjnts(armname="lft")
        bk_jawwidth_rgt = rbt.getjawwidth(armname="rgt")
        bk_jawwidth_lft = rbt.getjawwidth(armname="lft")
        if rbtinitarmjnts is None:
            rbt.movearmfk(rbt.initrgtjnts, armname="rgt")
            rbt.movearmfk(rbt.initlftjnts, armname="lft")
            rbt.opengripper(armname="rgt")
            rbt.opengripper(armname="lft")
        else:
            rbt.movearmfk(rbtinitarmjnts[0], armname="rgt")
            rbt.movearmfk(rbtinitarmjnts[1], armname="lft")
            rbt.opengripper(armname="rgt")
            rbt.opengripper(armname="lft")
        initpose = rbt.getarmjnts(armname=armname)
        if rbtgoalarmjnts is None:
            goalpose = rbt.initrgtjnts if armname is "rgt" else rbt.initlftjnts
        else:
            goalpose = rbtgoalarmjnts[1] if armname is "rgt" else rbtgoalarmjnts[0]

        if primitivedirection_init_forward is None:
            primitivedirection_init_forward = np.array([0, 0, -1])
        if primitivedirection_init_backward is None:
            primitivedirection_init_backward = np.array([0, 0, 1])
        if primitivedirection_final_forward is None:
            primitivedirection_final_forward = np.array([0, 0, -1])
        if primitivedirection_final_backward is None:
            primitivedirection_final_backward = np.array([0, 0, 1])

        for candidatepredgid in candidatepredgidlist:
            initpickup = self.pickup(objinithomomat, predefinedgrasps[candidatepredgid],
                                     primitivedirection_forward=primitivedirection_init_forward,
                                     primitivedirection_backward=primitivedirection_init_backward,
                                     primitivedistance_forward=primitivedistance_init_foward,
                                     primitivedistance_backward=premitivedistance_init_backward,
                                     armname=armname, obstaclecmlist=obscmlist)
            if initpickup is None:
                continue
            goalpickup = self.pickup(objgoalhomomat, predefinedgrasps[candidatepredgid],
                                     primitivedirection_forward=primitivedirection_final_forward,
                                     primitivedirection_backward=primitivedirection_final_backward,
                                     primitivedistance_forward=primitivedistance_final_foward,
                                     primitivedistance_backward=premitivedistance_final_backward,
                                     armname=armname, msc=initpickup[1][-1], obstaclecmlist=obscmlist)
            if goalpickup is None:
                continue
            initpick, initup = initpickup
            goalpick, goalup = goalpickup
            absinitpos = objinithomomat[:3, 3]
            absinitrot = objinithomomat[:3, :3]
            absgoalpos = objgoalhomomat[:3, 3]
            absgoalrot = objgoalhomomat[:3, :3]
            relpos, relrot = self.rhx.rbt.getinhandpose(absinitpos, absinitrot, initpick[-1], armname)
            if userrt:
                rrtpathinit = self.rhx.planmotion(initpose, initpick[0], obscmlist, armname)
                if rrtpathinit is None:
                    print("No path found for moving from rbtinitarmjnts to pick up!")
                    continue
                rrtpathhold = self.rhx.planmotionhold(initup[-1], goalpick[0], self.objcm, relpos, relrot, obscmlist,
                                                      armname)
                if rrtpathhold is None:
                    print("No path found for moving from pick up to place down!")
                    continue
                if finalstate[0] is "i":
                    rrtpathgoal = self.rhx.planmotion(goalup[-1], goalpose, self.objcm, absgoalpos, absgoalrot,
                                                      obscmlist, armname)
                    if rrtpathgoal is None:
                        print("No path found for moving from place down to rbtgoalarmjnts!")
                        continue
            jawwidthopen = hndfa.jawwidthopen
            jawwidthclose = predefinedgrasps[candidatepredgid][0]
            break

        numikmsmp = []
        jawwidthmsmp = []
        objmsmp = []
        remainingarmjnts = rbt.getarmjnts(armname="rgt") if armname is "lft" else rbt.getarmjnts(armname="lft")
        if initpickup is not None and goalpickup is not None:
            if userrt:
                if rrtpathinit is not None and rrtpathhold is not None:
                    if finalstate[0] in ["i", "u"]:
                        numikmp, jawwidthmp, objmp = self.formulatemotionpath(rrtpathinit, armname,
                                                                              rbt.getjawwidth(armname), absinitpos,
                                                                              absinitrot, remainingarmjnts)
                        numikmsmp.append(numikmp)
                        jawwidthmsmp.append(jawwidthmp)
                        objmsmp.append(objmp)
                        numikmp, jawwidthmp, objmp = self.formulatemotionpath(initpick, armname,
                                                                              rbt.getjawwidth(armname), absinitpos,
                                                                              absinitrot, remainingarmjnts)
                        numikmsmp.append(numikmp)
                        jawwidthmsmp.append(jawwidthmp)
                        objmsmp.append(objmp)
                        numikmp, jawwidthmp, objmp = self.formulatemotionpath([initpick[-1]], armname, jawwidthclose,
                                                                              absinitpos, absinitrot, remainingarmjnts)
                        numikmsmp.append(numikmp)
                        jawwidthmsmp.append(jawwidthmp)
                        objmsmp.append(objmp)
                        numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(initup, armname, jawwidthclose,
                                                                                  relpos, relrot, remainingarmjnts)
                        numikmsmp.append(numikmp)
                        jawwidthmsmp.append(jawwidthmp)
                        objmsmp.append(objmp)
                        numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(rrtpathhold, armname, jawwidthclose,
                                                                                  relpos, relrot, remainingarmjnts)
                        numikmsmp.append(numikmp)
                        jawwidthmsmp.append(jawwidthmp)
                        objmsmp.append(objmp)
                        numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(goalpick, armname, jawwidthclose,
                                                                                  relpos, relrot, remainingarmjnts)
                        numikmsmp.append(numikmp)
                        jawwidthmsmp.append(jawwidthmp)
                        objmsmp.append(objmp)
                        numikmp, jawwidthmp, objmp = self.formulatemotionpath([goalpick[-1]], armname, jawwidthopen,
                                                                              absgoalpos, absgoalrot, remainingarmjnts)
                        numikmsmp.append(numikmp)
                        jawwidthmsmp.append(jawwidthmp)
                        objmsmp.append(objmp)
                        numikmp, jawwidthmp, objmp = self.formulatemotionpath(goalup, armname, jawwidthopen, absgoalpos,
                                                                              absgoalrot, remainingarmjnts)
                        numikmsmp.append(numikmp)
                        jawwidthmsmp.append(jawwidthmp)
                        objmsmp.append(objmp)
                        if finalstate[0] is "i":
                            numikmp, jawwidthmp, objmp = self.formulatemotionpath(rrtpathgoal, armname, jawwidthopen,
                                                                                  absgoalpos, absgoalrot,
                                                                                  remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                    else:
                        finaljawwidth = jawwidthopen if finalstate[1] is "o" else jawwidthclose
                        numikmp, jawwidthmp, objmp = self.formulatemotionpath(rrtpathinit, armname,
                                                                              rbt.getjawwidth(armname), absinitpos,
                                                                              absinitrot, remainingarmjnts)
                        numikmsmp.append(numikmp)
                        jawwidthmsmp.append(jawwidthmp)
                        objmsmp.append(objmp)
                        numikmp, jawwidthmp, objmp = self.formulatemotionpath(initpick, armname,
                                                                              rbt.getjawwidth(armname), absinitpos,
                                                                              absinitrot, remainingarmjnts)
                        numikmsmp.append(numikmp)
                        jawwidthmsmp.append(jawwidthmp)
                        objmsmp.append(objmp)
                        numikmp, jawwidthmp, objmp = self.formulatemotionpath([initpick[-1]], armname, jawwidthclose,
                                                                              absinitpos, absinitrot, remainingarmjnts)
                        numikmsmp.append(numikmp)
                        jawwidthmsmp.append(jawwidthmp)
                        objmsmp.append(objmp)
                        numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(initup, armname, jawwidthclose,
                                                                                  relpos, relrot, remainingarmjnts)
                        numikmsmp.append(numikmp)
                        jawwidthmsmp.append(jawwidthmp)
                        objmsmp.append(objmp)
                        numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(rrtpathhold, armname, jawwidthclose,
                                                                                  relpos, relrot, remainingarmjnts)
                        numikmsmp.append(numikmp)
                        jawwidthmsmp.append(jawwidthmp)
                        objmsmp.append(objmp)
                        numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(goalpick, armname, jawwidthclose,
                                                                                  relpos, relrot, remainingarmjnts)
                        numikmsmp.append(numikmp)
                        jawwidthmsmp.append(jawwidthmp)
                        objmsmp.append(objmp)
                        numikmp, jawwidthmp, objmp = self.formulatemotionpath([goalpick[-1]], armname, finaljawwidth,
                                                                              absgoalpos, absgoalrot, remainingarmjnts)
                        numikmsmp.append(numikmp)
                        jawwidthmsmp.append(jawwidthmp)
                        objmsmp.append(objmp)
                else:
                    print("There is possible sequence but no collision free motion!")
            else:
                if finalstate[0] in ["i", "u"]:
                    numikmp, jawwidthmp, objmp = self.formulatemotionpath([initpose], armname, rbt.getjawwidth(armname),
                                                                          absinitpos, absinitrot, remainingarmjnts)
                    numikmsmp.append(numikmp)
                    jawwidthmsmp.append(jawwidthmp)
                    objmsmp.append(objmp)
                    numikmp, jawwidthmp, objmp = self.formulatemotionpath(initpick, armname, rbt.getjawwidth(armname),
                                                                          absinitpos, absinitrot, remainingarmjnts)
                    numikmsmp.append(numikmp)
                    jawwidthmsmp.append(jawwidthmp)
                    objmsmp.append(objmp)
                    numikmp, jawwidthmp, objmp = self.formulatemotionpath([initpick[-1]], armname, jawwidthclose,
                                                                          absinitpos, absinitrot, remainingarmjnts)
                    numikmsmp.append(numikmp)
                    jawwidthmsmp.append(jawwidthmp)
                    objmsmp.append(objmp)
                    numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(initup, armname, jawwidthclose, relpos,
                                                                              relrot, remainingarmjnts)
                    numikmsmp.append(numikmp)
                    jawwidthmsmp.append(jawwidthmp)
                    objmsmp.append(objmp)
                    numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(goalpick, armname, jawwidthclose, relpos,
                                                                              relrot, remainingarmjnts)
                    numikmsmp.append(numikmp)
                    jawwidthmsmp.append(jawwidthmp)
                    objmsmp.append(objmp)
                    numikmp, jawwidthmp, objmp = self.formulatemotionpath([goalpick[-1]], armname, jawwidthopen,
                                                                          absgoalpos, absgoalrot, remainingarmjnts)
                    numikmsmp.append(numikmp)
                    jawwidthmsmp.append(jawwidthmp)
                    objmsmp.append(objmp)
                    numikmp, jawwidthmp, objmp = self.formulatemotionpath(goalup, armname, jawwidthopen, absgoalpos,
                                                                          absgoalrot, remainingarmjnts)
                    numikmsmp.append(numikmp)
                    jawwidthmsmp.append(jawwidthmp)
                    objmsmp.append(objmp)
                    if finalstate[0] is "i":
                        numikmp, jawwidthmp, objmp = self.formulatemotionpath([goalpose], armname, jawwidthopen,
                                                                              absgoalpos, absgoalrot, remainingarmjnts)
                        numikmsmp.append(numikmp)
                        jawwidthmsmp.append(jawwidthmp)
                        objmsmp.append(objmp)
                else:
                    finaljawwidth = jawwidthopen if finalstate[1] is "o" else jawwidthclose
                    numikmp, jawwidthmp, objmp = self.formulatemotionpath([initpose], armname, rbt.getjawwidth(armname),
                                                                          absinitpos, absinitrot, remainingarmjnts)
                    numikmsmp.append(numikmp)
                    jawwidthmsmp.append(jawwidthmp)
                    objmsmp.append(objmp)
                    numikmp, jawwidthmp, objmp = self.formulatemotionpath(initpick, armname, rbt.getjawwidth(armname),
                                                                          absinitpos, absinitrot, remainingarmjnts)
                    numikmsmp.append(numikmp)
                    jawwidthmsmp.append(jawwidthmp)
                    objmsmp.append(objmp)
                    numikmp, jawwidthmp, objmp = self.formulatemotionpath([initpick[-1]], armname, jawwidthclose,
                                                                          absinitpos, absinitrot, remainingarmjnts)
                    numikmsmp.append(numikmp)
                    jawwidthmsmp.append(jawwidthmp)
                    objmsmp.append(objmp)
                    numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(initup, armname, jawwidthclose, relpos,
                                                                              relrot, remainingarmjnts)
                    numikmsmp.append(numikmp)
                    jawwidthmsmp.append(jawwidthmp)
                    objmsmp.append(objmp)
                    numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(goalpick, armname, jawwidthclose, relpos,
                                                                              relrot, remainingarmjnts)
                    numikmsmp.append(numikmp)
                    jawwidthmsmp.append(jawwidthmp)
                    objmsmp.append(objmp)
                    numikmp, jawwidthmp, objmp = self.formulatemotionpath([goalpick[-1]], armname, finaljawwidth,
                                                                          absgoalpos, absgoalrot, remainingarmjnts)
                    numikmsmp.append(numikmp)
                    jawwidthmsmp.append(jawwidthmp)
                    objmsmp.append(objmp)
            rbt.movearmfk(bk_armjnts_rgt, armname="rgt")
            rbt.movearmfk(bk_armjnts_lft, armname="lft")
            rbt.opengripper(armname="rgt", jawwidth=bk_jawwidth_rgt)
            rbt.opengripper(armname="lft", jawwidth=bk_jawwidth_lft)
            return [numikmsmp, jawwidthmsmp, objmsmp]
        else:
            rbt.movearmfk(bk_armjnts_rgt, armname="rgt")
            rbt.movearmfk(bk_armjnts_lft, armname="lft")
            rbt.opengripper(armname="rgt", jawwidth=bk_jawwidth_rgt)
            rbt.opengripper(armname="lft", jawwidth=bk_jawwidth_lft)
            return [None, None, None]

        print("No solution!")
        return [None, None, None]

    def findppmotion_symmetric_err(self, objinithomomat, objgoalhomomat, obscmlist=[], symmetricaxis="z", nangles=9,
                                   armname='rgt', rbtinitarmjnts=None, rbtgoalarmjnts=None, finalstate="io",
                                   primitivedirection_init_forward=None, primitivedirection_init_backward=None,
                                   primitivedistance_init_foward=150, premitivedistance_init_backward=150,
                                   primitivedirection_final_forward=None, primitivedirection_final_backward=None,
                                   primitivedistance_final_foward=150, premitivedistance_final_backward=150,
                                   userrt=True,
                                   toggledebug=False):
        """
        this function performs findsharedgrasps and genppmotion in a loop
        the postfix err indicates this function will return the error type

        :param candidatepredgidlist: candidate predefined grasp id list [int, int, ...]
        :param objinithomomat:
        :param objgoalhomomat:
        :param armname:
        :param rbtinitarmjnts, rbtgoalarmjnts: [rgtsarmjnts, lftarmjnts], initial robot arm will be used if not set
        :param finalstate: use to orchestrate the motion, could be: "io"->backtoaninitialpose, gripper open, "uo"->backtoup, gripperopen, "gc"->stopatgoal, gripperclose
        :param primitivedirection_init_forward: the vector to move to the object init pose
        :param primitivedirection_init_backward: the vector to move back from the object init pose
        :param primitivedistance_init_forward
        :param primitivedistance_init_backward
        :param primitivedirection_final_forward: the vector to move to the object goal pose
        :param primitivedirection_final_backward: the vector to move back from the object goal pose
        :param primitivedistance_final_forward
        :param primitivedistance_final_backward
        :param userrt: rrt generator
        :param obscmlist:

        :return:
            status, numikmsmp, jawwidthmsmp, objmsmp
            status = "done", "nig", "ngg", "nil", "ngl", "nm"
            nig - no init grasp
            ngg - no goal grasp
            nil - no init linear
            ngl - no goal linear
            nm - no motion

        author: weiwei
        date: 20200425
        """

        # find sharedgid

        objcmcopy = copy.deepcopy(self.objcm)

        rbt = self.rhx.rbt
        bk_armjnts_rgt = rbt.getarmjnts(armname="rgt")
        bk_armjnts_lft = rbt.getarmjnts(armname="lft")
        bk_jawwidth_rgt = rbt.getjawwidth(armname="rgt")
        bk_jawwidth_lft = rbt.getjawwidth(armname="lft")
        rbt.movearmfk(rbt.initrgtjnts, armname="rgt")
        rbt.movearmfk(rbt.initlftjnts, armname="lft")
        rbt.opengripper(armname="rgt")
        rbt.opengripper(armname="lft")
        rbtmg = self.rhx.rbtmesh
        hndfa = self.rhx.rgthndfa if armname is "rgt" else self.rhx.lfthndfa
        predefinedgrasps = self.identityglist_rgt if armname is "rgt" else self.identityglist_lft
        pcdchecker = self.rhx.pcdchecker
        bcdchecker = self.rhx.bcdchecker
        np = self.rhx.np
        rm = self.rhx.rm
        # start pose
        objcmcopy.sethomomat(objinithomomat)
        ikfailedgraspsnum = 0
        ikcolliedgraspsnum = 0
        availablegraspsatinit = []
        for idpre, predefined_grasp in enumerate(predefinedgrasps):
            # if toggledebug:
            #     availablegraspsatinit.append(idpre)
            predefined_jawwidth, predefined_fc, predefined_homomat = predefined_grasp
            hndmat4 = np.dot(objinithomomat, predefined_homomat)
            eepos = rm.homotransformpoint(objinithomomat, predefined_fc)[:3]
            eerot = hndmat4[:3, :3]
            hndnew = hndfa.genHand()
            hndnew.sethomomat(hndmat4)
            # hndnew.setjawwidth(predefined_jawwidth)
            hndcmlist = hndnew.genrangecmlist(jawwidthstart=predefined_jawwidth, jawwidthend=hndnew.jawwidthopen)
            ishndcollided = bcdchecker.isMeshListMeshListCollided(hndcmlist, obscmlist)
            if not ishndcollided:
                armjnts = rbt.numik(eepos, eerot, armname)
                if armjnts is not None:
                    if toggledebug:
                        hndnew = hndfa.genHand()
                        hndnew.setColor(0, 1, 0, .2)
                        hndnew.sethomomat(hndmat4)
                        hndnew.setjawwidth(predefined_jawwidth)
                        # hndnew.reparentTo(self.rhx.base.render)
                    rbt.movearmfk(armjnts, armname)
                    isrbtcollided = pcdchecker.isRobotCollided(rbt, obscmlist, holdarmname=armname)
                    # isobjcollided = pcdchecker.isObjectsOthersCollided([obj], rbt, armname, obscmlist)
                    isobjcollided = False  # for future use
                    if (not isrbtcollided) and (not isobjcollided):
                        if toggledebug:
                            # rbtmg.genmnp(rbt, drawhand=False, togglejntscoord=False, toggleendcoord=False, rgbargt=[0, 1, 0, .5]).reparentTo(
                            #     self.rhx.base.render)
                            pass
                        # if not toggledebug:
                        #     availablegraspsatinit.append(idpre)
                        availablegraspsatinit.append(idpre)
                    elif (not isobjcollided):
                        if toggledebug:
                            # rbtmg.genmnp(rbt, drawhand=False, togglejntscoord=False, toggleendcoord=False, rgbargt=[1, 0, 1, .5]).reparentTo(
                            #     self.rhx.base.render)
                            pass
                else:
                    ikfailedgraspsnum += 1
                    if toggledebug:
                        hndnew = hndfa.genHand()
                        hndnew.setColor(1, .6, 0, .7)
                        hndnew.sethomomat(hndmat4)
                        hndnew.setjawwidth(predefined_jawwidth)
                        # hndnew.reparentTo(self.rhx.base.render)
                    # bcdchecker.showMeshList(hndnew.cmlist)
            else:
                ikcolliedgraspsnum += 1
                if toggledebug:
                    hndnew = hndfa.genHand()
                    hndnew.setColor(1, .5, .5, .7)
                    hndnew.sethomomat(hndmat4)
                    hndnew.setjawwidth(predefined_jawwidth)
                    # hndnew.reparentTo(self.rhx.base.render)
                # bcdchecker.showMeshList(hndnew.cmlist)

        print("IK failed at the init pose: ", ikfailedgraspsnum)
        print("Collision at the init pose: ", ikcolliedgraspsnum)
        print("Possible number of the grasps at the init pose: ", len(availablegraspsatinit))

        # if toggledebug:
        #     base.run()
        # goal pose
        finalsharedgrasps = []
        ikfailedgraspsnum = 0
        ikcolliedgraspsnum = 0

        if symmetricaxis is "z":
            rotax = objgoalhomomat[:3, 2]
        elif symmetricaxis is "y":
            rotax = objgoalhomomat[:3, 1]
        else:
            rotax = objgoalhomomat[:3, 0]
        candidateangles = np.linspace(0, 360, nangles)
        if toggledebug:
            candidateangles = [90.0]
        err = "no_"
        for rotangle in candidateangles:
            objmat4 = copy.deepcopy(objgoalhomomat)
            objmat4[:3, :3] = np.dot(rm.rodrigues(rotax, rotangle), objgoalhomomat[:3, :3])
            objcmcopy.sethomomat(objmat4)
            for idavailableinit in availablegraspsatinit:
                predefined_jawwidth, predefined_fc, predefined_homomat = predefinedgrasps[idavailableinit]
                hndmat4 = np.dot(objmat4, predefined_homomat)
                eepos = rm.homotransformpoint(objmat4, predefined_fc)[:3]
                eerot = hndmat4[:3, :3]
                hndnew = hndfa.genHand()
                hndnew.sethomomat(hndmat4)
                hndcmlist = hndnew.genrangecmlist(jawwidthstart=predefined_jawwidth, jawwidthend=hndnew.jawwidthopen)
                # hndnew.setjawwidth(predefined_jawwidth)
                ishndcollided = bcdchecker.isMeshListMeshListCollided(hndcmlist, obscmlist)
                if not ishndcollided:
                    armjnts = rbt.numik(eepos, eerot, armname)
                    if armjnts is not None:
                        if toggledebug:
                            hndnew = hndfa.genHand()
                            hndnew.setColor(0, 1, 0, .5)
                            hndnew.sethomomat(hndmat4)
                            hndnew.setjawwidth(predefined_jawwidth)
                            # hndnew.reparentTo(self.rhx.base.render)
                        rbt.movearmfk(armjnts, armname)
                        isrbtcollided = pcdchecker.isRobotCollided(rbt, obscmlist, holdarmname=armname)
                        # isobjcollided = pcdchecker.isObjectsOthersCollided([obj], rbt, armname, obscmlist)
                        isobjcollided = False
                        if (not isrbtcollided) and (not isobjcollided):
                            finalsharedgrasps.append(idavailableinit)
                            if toggledebug:
                                rbtmg.genmnp(rbt, togglejntscoord=False, toggleendcoord=False,
                                             rgbargt=[0, 1, 0, .5]).reparentTo(
                                    self.rhx.base.render)
                                # rbtmg.genmnp(rbt, togglejntscoord=False).reparentTo(self.rhx.base.render)
                                # # toggle the following one in case both the common start and goal shall be rendered
                                # hndmat4 = np.dot(objinithomomat, predefined_homomat)
                                # # hndnew = hndfa.genHand()
                                # # hndnew.setColor(0, 1, 0, .5)
                                # # hndnew.sethomomat(hndmat4)
                                # # hndnew.setjawwidth(predefined_jawwidth)
                                # # hndnew.reparentTo(self.rhx.base.render)
                                # eepos = rm.homotransformpoint(objinithomomat, predefined_fc)[:3]
                                # eerot = hndmat4[:3, :3]
                                # armjnts = rbt.numik(eepos, eerot, armname)
                                # rbt.movearmfk(armjnts, armname)
                                # rbtmg.genmnp(rbt, togglejntscoord=False, toggleendcoord=False, rgbargt=[0, 1, 0, .5]).reparentTo(
                                #     self.rhx.base.render)
                                # # base.run()
                                # rbtmg.genmnp(rbt, togglejntscoord=False).reparentTo(self.rhx.base.render)
                                pass
                        elif (not isobjcollided):
                            if toggledebug:
                                # rbtmg.genmnp(rbt, drawhand=False, togglejntscoord=False, toggleendcoord=False,
                                #              rgbargt=[1, 0, 1, .5]).reparentTo(self.rhx.base.render)
                                pass
                    else:
                        ikfailedgraspsnum += 1
                        if toggledebug:
                            hndnew = hndfa.genHand()
                            hndnew.setColor(1, .6, 0, .7)
                            hndnew.sethomomat(hndmat4)
                            hndnew.setjawwidth(predefined_jawwidth)
                            # hndnew.reparentTo(self.rhx.base.render)
                else:
                    ikcolliedgraspsnum += 1
                    if toggledebug:
                        hndnew = hndfa.genHand()
                        hndnew.setColor(1, .5, .5, .7)
                        hndnew.sethomomat(hndmat4)
                        hndnew.setjawwidth(predefined_jawwidth)
                        # hndnew.reparentTo(self.rhx.base.render)
            print("IK failed grasps at the goal pose: ", ikfailedgraspsnum)
            print("Collision at the goal pose: ", ikcolliedgraspsnum)
            print("Possble number of shared grasps: ", len(finalsharedgrasps))
            if toggledebug:
                base.run()

            if len(finalsharedgrasps) == 0:
                print("No shared grasps at symmetric angle " + str(rotangle) + ", trying the next...")
                if rotangle == candidateangles[-1]:
                    print("No shared grasps!")
                    if availablegraspsatinit == 0:
                        return ["nig", None, None, None]
                    else:
                        return ["ngg", None, None, None]
                continue
            else:
                resultinghomomat = copy.deepcopy(objmat4)
                resultinghomomat[:3, 3] = resultinghomomat[:3, 3] + resultinghomomat[:3,
                                                                    2] * 5  # move 5mm up, do not move until end
                # if toggledebug:
                #     for idsharedgrasps in finalsharedgrasps:
                #         predefined_jawwidth, predefined_fc, predefined_homomat = predefinedgrasps[idsharedgrasps]
                #         hndmat4 = np.dot(objinithomomat, predefined_homomat)
                #         eepos = rm.homotransformpoint(objinithomomat, predefined_fc)[:3]
                #         eerot = hndmat4[:3, :3]
                #         armjnts = rbt.numik(eepos, eerot, armname)
                #         rbt.movearmfk(armjnts, armname)
                #         rbtmg.genmnp(rbt, togglejntscoord=False).reparentTo(self.rhx.base.render)

                if rbtinitarmjnts is not None and rbtgoalarmjnts is not None:
                    if armname is "rgt":
                        if not np.isclose(rbtinitarmjnts[1], rbtgoalarmjnts[1]):
                            print("The lft arm must maintain unmoved during generating ppsglmotion for the rgt arm.")
                            raise ValueError("")
                    elif armname is "lft":
                        if not np.isclose(rbtinitarmjnts[0], rbtgoalarmjnts[0]):
                            print("The rgt arm must maintain unmoved during generating ppsglmotion for the lft arm.")
                            raise ValueError("")
                    else:
                        print("Wrong armname. Must be rgt or lft.")
                        raise ValueError("")
                if rbtinitarmjnts is not None:
                    rbt.movearmfk(rbtinitarmjnts[0], armname="rgt")
                    rbt.movearmfk(rbtinitarmjnts[1], armname="lft")
                    rbt.opengripper(armname="rgt")
                    rbt.opengripper(armname="lft")
                else:
                    rbt.movearmfk(rbt.initrgtjnts, armname="rgt")
                    rbt.movearmfk(rbt.initlftjnts, armname="lft")
                    rbt.opengripper(armname="rgt")
                    rbt.opengripper(armname="lft")
                initpose = rbt.getarmjnts(armname=armname)
                if rbtgoalarmjnts is None:
                    goalpose = rbt.initrgtjnts if armname is "rgt" else rbt.initlftjnts
                else:
                    goalpose = rbtgoalarmjnts[0] if armname is "rgt" else rbtgoalarmjnts[1]

                if primitivedirection_init_forward is None:
                    # primitivedirection_init_forward = np.array([0, 0, -1])
                    primitivedirection_init_forward = -objinithomomat[:3, 2]
                if primitivedirection_init_backward is None:
                    # primitivedirection_init_backward = np.array([0, 0, 1])
                    primitivedirection_init_backward = objinithomomat[:3, 2]
                if primitivedirection_final_forward is None:
                    # primitivedirection_final_forward = np.array([0, 0, -1])
                    primitivedirection_final_forward = -objgoalhomomat[:3, 2]
                if primitivedirection_final_backward is None:
                    # primitivedirection_final_backward = np.array([0, 0, 1])
                    primitivedirection_final_backward = objgoalhomomat[:3, 2]

                initpickup = None
                goalpickup = None
                for candidatepredgid in finalsharedgrasps:
                    print("picking up...")
                    initpickup = self.pickup(objinithomomat, predefinedgrasps[candidatepredgid],
                                             primitivedirection_forward=primitivedirection_init_forward,
                                             primitivedirection_backward=primitivedirection_init_backward,
                                             primitivedistance_forward=primitivedistance_init_foward,
                                             primitivedistance_backward=premitivedistance_init_backward,
                                             armname=armname, obstaclecmlist=obscmlist)
                    if initpickup is None:
                        if toggledebug and rotangle == candidateangles[-1]:
                            predefinedjawwidth, predefinedhndfc, predefinedhandhomomat = predefinedgrasps[
                                candidatepredgid]
                            hndmat4 = np.dot(objinithomomat, predefinedhandhomomat)
                            eepos = rm.homotransformpoint(objinithomomat, predefinedhndfc)
                            eerot = hndmat4[:3, :3]
                            amjnts = self.rhx.rbt.numik(eepos, eerot, armname)
                            rbt.movearmfk(armjnts, armname="rgt")
                            if amjnts is not None:
                                pickmotion = self.rhx.genmoveforwardmotion(primitivedirection_init_forward,
                                                                           primitivedistance_init_foward,
                                                                           amjnts, armname)
                                if pickmotion is None:
                                    rbtmg.genmnp(rbt, togglejntscoord=False, toggleendcoord=False,
                                                 rgbargt=np.array([1, 0, 0, .7])).reparentTo(self.rhx.base.render)
                                    if candidatepredgid == finalsharedgrasps[-1]:
                                        base.run()
                            rbt.movearmfk(rbt.initrgtjnts, armname="rgt")
                            rbt.movearmfk(rbt.initlftjnts, armname="lft")
                        continue
                    print("placing down...")
                    goalpickup = self.pickup(resultinghomomat, predefinedgrasps[candidatepredgid],
                                             primitivedirection_forward=primitivedirection_final_forward,
                                             primitivedirection_backward=primitivedirection_final_backward,
                                             primitivedistance_forward=primitivedistance_final_foward,
                                             primitivedistance_backward=premitivedistance_final_backward,
                                             armname=armname, msc=initpickup[1][-1], obstaclecmlist=obscmlist)
                    if goalpickup is None:
                        if toggledebug and rotangle == candidateangles[-1]:
                            predefinedjawwidth, predefinedhndfc, predefinedhandhomomat = predefinedgrasps[
                                candidatepredgid]
                            hndmat4 = np.dot(objinithomomat, predefinedhandhomomat)
                            eepos = rm.homotransformpoint(objinithomomat, predefinedhndfc)
                            eerot = hndmat4[:3, :3]
                            amjnts = self.rhx.rbt.numikmsc(eepos, eerot, initpickup[1][-1], armname)
                            rbt.movearmfk(armjnts, armname="rgt")
                            if amjnts is not None:
                                pickmotion = self.rhx.genmoveforwardmotion(primitivedirection_init_forward,
                                                                           primitivedistance_init_foward,
                                                                           amjnts, armname)
                                if pickmotion is None:
                                    # rbtmg.genmnp(rbt, togglejntscoord=False, toggleendcoord=False, rgbargt=np.array([1, 0, 0, .7])).reparentTo(self.rhx.base.render)
                                    if candidatepredgid == finalsharedgrasps[-1]:
                                        base.run()
                            rbt.movearmfk(rbt.initrgtjnts, armname="rgt")
                            rbt.movearmfk(rbt.initlftjnts, armname="lft")
                        continue
                    initpick, initup = initpickup
                    goalpick, goalup = goalpickup
                    absinitpos = objinithomomat[:3, 3]
                    absinitrot = objinithomomat[:3, :3]
                    absgoalpos = resultinghomomat[:3, 3]
                    absgoalrot = resultinghomomat[:3, :3]
                    relpos, relrot = self.rhx.rbt.getinhandpose(absinitpos, absinitrot, initpick[-1], armname)
                    if userrt:
                        rrtpathinit = self.rhx.planmotion(initpose, initpick[0], obscmlist, armname)
                        if rrtpathinit is None:
                            print("No path found for moving from rbtinitarmjnts to pick up!")
                            continue
                        rrtpathhold = self.rhx.planmotionhold(initup[-1], goalpick[0], self.objcm, relpos, relrot,
                                                              obscmlist, armname)
                        if rrtpathhold is None:
                            print("No path found for moving from pick up to place down!")
                            continue
                        if finalstate[0] is "i":
                            rrtpathgoal = self.rhx.planmotion(goalup[-1], goalpose, obscmlist, armname)
                            if rrtpathgoal is None:
                                print("No path found for moving from place down to rbtgoalarmjnts!")
                                continue
                    jawwidthopen = hndfa.jawwidthopen
                    jawwidthclose = predefinedgrasps[candidatepredgid][0]
                    break

                numikmsmp = []
                jawwidthmsmp = []
                objmsmp = []
                remainingarmjnts = rbt.getarmjnts(armname="rgt") if armname is "lft" else rbt.getarmjnts(armname="lft")
                if initpickup is not None and goalpickup is not None:
                    if userrt:
                        if rrtpathinit is not None and rrtpathhold is not None:
                            if finalstate[0] in ["i", "u"]:
                                numikmp, jawwidthmp, objmp = self.formulatemotionpath(rrtpathinit, armname,
                                                                                      rbt.getjawwidth(armname),
                                                                                      absinitpos,
                                                                                      absinitrot, remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpath(initpick, armname,
                                                                                      rbt.getjawwidth(armname),
                                                                                      absinitpos,
                                                                                      absinitrot, remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpath([initpick[-1]], armname,
                                                                                      jawwidthclose,
                                                                                      absinitpos, absinitrot,
                                                                                      remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(initup, armname,
                                                                                          jawwidthclose,
                                                                                          relpos, relrot,
                                                                                          remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(rrtpathhold, armname,
                                                                                          jawwidthclose,
                                                                                          relpos, relrot,
                                                                                          remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(goalpick, armname,
                                                                                          jawwidthclose,
                                                                                          relpos, relrot,
                                                                                          remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpath([goalpick[-1]], armname,
                                                                                      jawwidthopen,
                                                                                      absgoalpos, absgoalrot,
                                                                                      remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpath(goalup, armname, jawwidthopen,
                                                                                      absgoalpos,
                                                                                      absgoalrot, remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                if finalstate[0] is "i":
                                    numikmp, jawwidthmp, objmp = self.formulatemotionpath(rrtpathgoal, armname,
                                                                                          jawwidthopen,
                                                                                          absgoalpos, absgoalrot,
                                                                                          remainingarmjnts)
                                    numikmsmp.append(numikmp)
                                    jawwidthmsmp.append(jawwidthmp)
                                    objmsmp.append(objmp)
                            else:
                                finaljawwidth = jawwidthopen if finalstate[1] is "o" else jawwidthclose
                                numikmp, jawwidthmp, objmp = self.formulatemotionpath(rrtpathinit, armname,
                                                                                      rbt.getjawwidth(armname),
                                                                                      absinitpos,
                                                                                      absinitrot, remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpath(initpick, armname,
                                                                                      rbt.getjawwidth(armname),
                                                                                      absinitpos,
                                                                                      absinitrot, remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpath([initpick[-1]], armname,
                                                                                      jawwidthclose,
                                                                                      absinitpos, absinitrot,
                                                                                      remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(initup, armname,
                                                                                          jawwidthclose,
                                                                                          relpos, relrot,
                                                                                          remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(rrtpathhold, armname,
                                                                                          jawwidthclose,
                                                                                          relpos, relrot,
                                                                                          remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(goalpick, armname,
                                                                                          jawwidthclose,
                                                                                          relpos, relrot,
                                                                                          remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpath([goalpick[-1]], armname,
                                                                                      finaljawwidth,
                                                                                      absgoalpos, absgoalrot,
                                                                                      remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                        else:
                            print("There is possible sequence but no collision free motion at " + str(
                                rotangle) + ", trying the next...")
                    else:
                        if finalstate[0] in ["i", "u"]:
                            numikmp, jawwidthmp, objmp = self.formulatemotionpath([initpose], armname,
                                                                                  rbt.getjawwidth(armname),
                                                                                  absinitpos, absinitrot,
                                                                                  remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            numikmp, jawwidthmp, objmp = self.formulatemotionpath(initpick, armname,
                                                                                  rbt.getjawwidth(armname),
                                                                                  absinitpos, absinitrot,
                                                                                  remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            numikmp, jawwidthmp, objmp = self.formulatemotionpath([initpick[-1]], armname,
                                                                                  jawwidthclose,
                                                                                  absinitpos, absinitrot,
                                                                                  remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(initup, armname, jawwidthclose,
                                                                                      relpos,
                                                                                      relrot, remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(goalpick, armname, jawwidthclose,
                                                                                      relpos,
                                                                                      relrot, remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            numikmp, jawwidthmp, objmp = self.formulatemotionpath([goalpick[-1]], armname, jawwidthopen,
                                                                                  absgoalpos, absgoalrot,
                                                                                  remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            numikmp, jawwidthmp, objmp = self.formulatemotionpath(goalup, armname, jawwidthopen,
                                                                                  absgoalpos,
                                                                                  absgoalrot, remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            if finalstate[0] is "i":
                                numikmp, jawwidthmp, objmp = self.formulatemotionpath([goalpose], armname, jawwidthopen,
                                                                                      absgoalpos, absgoalrot,
                                                                                      remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                        else:
                            finaljawwidth = jawwidthopen if finalstate[1] is "o" else jawwidthclose
                            numikmp, jawwidthmp, objmp = self.formulatemotionpath([initpose], armname,
                                                                                  rbt.getjawwidth(armname),
                                                                                  absinitpos, absinitrot,
                                                                                  remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            numikmp, jawwidthmp, objmp = self.formulatemotionpath(initpick, armname,
                                                                                  rbt.getjawwidth(armname),
                                                                                  absinitpos, absinitrot,
                                                                                  remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            numikmp, jawwidthmp, objmp = self.formulatemotionpath([initpick[-1]], armname,
                                                                                  jawwidthclose,
                                                                                  absinitpos, absinitrot,
                                                                                  remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(initup, armname, jawwidthclose,
                                                                                      relpos,
                                                                                      relrot, remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(goalpick, armname, jawwidthclose,
                                                                                      relpos,
                                                                                      relrot, remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            numikmp, jawwidthmp, objmp = self.formulatemotionpath([goalpick[-1]], armname,
                                                                                  finaljawwidth,
                                                                                  absgoalpos, absgoalrot,
                                                                                  remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                    rbt.movearmfk(bk_armjnts_rgt, armname="rgt")
                    rbt.movearmfk(bk_armjnts_lft, armname="lft")
                    rbt.opengripper(armname="rgt", jawwidth=bk_jawwidth_rgt)
                    rbt.opengripper(armname="lft", jawwidth=bk_jawwidth_lft)

                    if toggledebug:
                        # draw motion
                        for i, numikms in enumerate(numikmsmp):
                            for j, numik in enumerate(numikms):
                                rbt.movearmfk(numik[1], armname="rgt")
                                rbt.movearmfk(numik[2], armname="lft")
                                rbt.opengripper(armname="rgt", jawwidth=jawwidthmsmp[i][j][0])
                                rbt.opengripper(armname="lft", jawwidth=jawwidthmsmp[i][j][1])
                                if j == 0 or j == len(numikms):
                                    rbtmg.genmnp(rbt, togglejntscoord=False, toggleendcoord=False,
                                                 rgbargt=np.array([0, 1, 0, 1])).reparentTo(self.rhx.base.render)
                                else:
                                    rbtmg.genmnp(rbt, togglejntscoord=False, toggleendcoord=False,
                                                 rgbargt=np.array([.5, .5, .5, .5])).reparentTo(self.rhx.base.render)
                                if j < len(numikms) - 1:
                                    numiknxt = numikms[j + 1][1]
                                    thispos, _ = rbt.getee(armname="rgt")
                                    rbt.movearmfk(numiknxt, armname="rgt")
                                    nxtpos, _ = rbt.getee(armname="rgt")
                                    p3dh.genlinesegnodepath([thispos, nxtpos], colors=[1, 0, 1, 1],
                                                            thickness=3.7).reparentTo(base.render)
                                objcp = copy.deepcopy(self.objcm)
                                objcp.sethomomat(objmsmp[i][j])
                                objcp.setColor(1, 1, 0, 1)
                                objcp.reparentTo(base.render)
                        base.run()

                    return ["done", numikmsmp, jawwidthmsmp, objmsmp]
                else:
                    rbt.movearmfk(bk_armjnts_rgt, armname="rgt")
                    rbt.movearmfk(bk_armjnts_lft, armname="lft")
                    rbt.opengripper(armname="rgt", jawwidth=bk_jawwidth_rgt)
                    rbt.opengripper(armname="lft", jawwidth=bk_jawwidth_lft)
                    if initpickup is None:
                        print("No feasible motion for pick up!")
                        if rotangle == candidateangles[-1]:
                            print("All trials for pick-up failed!")
                            return ["nil", None, None, None]
                    if goalpickup is None:
                        print("No feasible motion for placing down!")
                        if rotangle == candidateangles[-1]:
                            print("All trials for place-down failed!")
                            return ["ngl", None, None, None]
                    print("The shared grasps failed at symmetric angle " + str(rotangle) + ", trying the next...")
                    continue

        # if toggledebug:
        #     base.run()
        print("No feasible motion between two key poses!")
        return ["nm", None, None, None]

    def findppmotion_symmetric(self, objinithomomat, objgoalhomomat, obscmlist=[], symmetricaxis="z", nangles=9,
                               armname='rgt', rbtinitarmjnts=None, rbtgoalarmjnts=None, finalstate="io",
                               primitivedirection_init_forward=None, primitivedirection_init_backward=None,
                               primitivedistance_init_foward=150, premitivedistance_init_backward=150,
                               primitivedirection_final_forward=None, primitivedirection_final_backward=None,
                               primitivedistance_final_foward=150, premitivedistance_final_backward=150, userrt=True,
                               toggledebug=False):
        """
        this function performs findsharedgrasps and genppmotion in a loop

        :param candidatepredgidlist: candidate predefined grasp id list [int, int, ...]
        :param objinithomomat:
        :param objgoalhomomat:
        :param armname:
        :param rbtinitarmjnts, rbtgoalarmjnts: [rgtsarmjnts, lftarmjnts], initial robot arm will be used if not set
        :param finalstate: use to orchestrate the motion, could be: "io"->backtoaninitialpose, gripper open, "uo"->backtoup, gripperopen, "gc"->stopatgoal, gripperclose
        :param primitivedirection_init_forward: the vector to move to the object init pose
        :param primitivedirection_init_backward: the vector to move back from the object init pose
        :param primitivedistance_init_forward
        :param primitivedistance_init_backward
        :param primitivedirection_final_forward: the vector to move to the object goal pose
        :param primitivedirection_final_backward: the vector to move back from the object goal pose
        :param primitivedistance_final_forward
        :param primitivedistance_final_backward
        :param userrt: rrt generator
        :param obscmlist:

        :return:

        author: weiwei
        date: 20200107
        """

        # find sharedgid

        objcmcopy = copy.deepcopy(self.objcm)

        rbt = self.rhx.rbt
        bk_armjnts_rgt = rbt.getarmjnts(armname="rgt")
        bk_armjnts_lft = rbt.getarmjnts(armname="lft")
        bk_jawwidth_rgt = rbt.getjawwidth(armname="rgt")
        bk_jawwidth_lft = rbt.getjawwidth(armname="lft")
        rbt.movearmfk(rbt.initrgtjnts, armname="rgt")
        rbt.movearmfk(rbt.initlftjnts, armname="lft")
        rbt.opengripper(armname="rgt")
        rbt.opengripper(armname="lft")
        rbtmg = self.rhx.rbtmesh
        hndfa = self.rhx.rgthndfa if armname is "rgt" else self.rhx.lfthndfa
        predefinedgrasps = self.identityglist_rgt if armname is "rgt" else self.identityglist_lft
        pcdchecker = self.rhx.pcdchecker
        bcdchecker = self.rhx.bcdchecker
        np = self.rhx.np
        rm = self.rhx.rm
        # start pose
        objcmcopy.sethomomat(objinithomomat)
        ikfailedgraspsnum = 0
        ikcolliedgraspsnum = 0
        availablegraspsatinit = []
        for idpre, predefined_grasp in enumerate(predefinedgrasps):
            # if toggledebug:
            #     availablegraspsatinit.append(idpre)
            predefined_jawwidth, predefined_fc, predefined_homomat = predefined_grasp
            hndmat4 = np.dot(objinithomomat, predefined_homomat)
            eepos = rm.homotransformpoint(objinithomomat, predefined_fc)[:3]
            eerot = hndmat4[:3, :3]
            hndnew = hndfa.genHand()
            hndnew.sethomomat(hndmat4)
            # hndnew.setjawwidth(predefined_jawwidth)
            hndcmlist = hndnew.genrangecmlist(jawwidthstart=predefined_jawwidth, jawwidthend=hndnew.jawwidthopen)
            ishndcollided = bcdchecker.isMeshListMeshListCollided(hndcmlist, obscmlist)
            if not ishndcollided:
                armjnts = rbt.numik(eepos, eerot, armname)
                if armjnts is not None:
                    if toggledebug:
                        hndnew = hndfa.genHand()
                        hndnew.setColor(0, 1, 0, .2)
                        hndnew.sethomomat(hndmat4)
                        hndnew.setjawwidth(predefined_jawwidth)
                        # hndnew.reparentTo(self.rhx.base.render)
                    rbt.movearmfk(armjnts, armname)
                    isrbtcollided = pcdchecker.isRobotCollided(rbt, obscmlist, holdarmname=armname)
                    # isobjcollided = pcdchecker.isObjectsOthersCollided([obj], rbt, armname, obscmlist)
                    isobjcollided = False  # for future use
                    if (not isrbtcollided) and (not isobjcollided):
                        if toggledebug:
                            # rbtmg.genmnp(rbt, drawhand=False, togglejntscoord=False, toggleendcoord=False, rgbargt=[0, 1, 0, .5]).reparentTo(
                            #     self.rhx.base.render)
                            pass
                        # if not toggledebug:
                        #     availablegraspsatinit.append(idpre)
                        availablegraspsatinit.append(idpre)
                    elif (not isobjcollided):
                        if toggledebug:
                            # rbtmg.genmnp(rbt, drawhand=False, togglejntscoord=False, toggleendcoord=False, rgbargt=[1, 0, 1, .5]).reparentTo(
                            #     self.rhx.base.render)
                            pass
                else:
                    ikfailedgraspsnum += 1
                    if toggledebug:
                        hndnew = hndfa.genHand()
                        hndnew.setColor(1, .6, 0, .7)
                        hndnew.sethomomat(hndmat4)
                        hndnew.setjawwidth(predefined_jawwidth)
                        # hndnew.reparentTo(self.rhx.base.render)
                    # bcdchecker.showMeshList(hndnew.cmlist)
            else:
                ikcolliedgraspsnum += 1
                if toggledebug:
                    hndnew = hndfa.genHand()
                    hndnew.setColor(1, .5, .5, .7)
                    hndnew.sethomomat(hndmat4)
                    hndnew.setjawwidth(predefined_jawwidth)
                    # hndnew.reparentTo(self.rhx.base.render)
                # bcdchecker.showMeshList(hndnew.cmlist)

        print("IK failed at the init pose: ", ikfailedgraspsnum)
        print("Collision at the init pose: ", ikcolliedgraspsnum)
        print("Possible number of the grasps at the init pose: ", len(availablegraspsatinit))

        # if toggledebug:
        #     base.run()
        # goal pose
        finalsharedgrasps = []
        ikfailedgraspsnum = 0
        ikcolliedgraspsnum = 0

        if symmetricaxis is "z":
            rotax = objgoalhomomat[:3, 2]
        elif symmetricaxis is "y":
            rotax = objgoalhomomat[:3, 1]
        else:
            rotax = objgoalhomomat[:3, 0]
        candidateangles = np.linspace(0, 360, nangles)
        if toggledebug:
            candidateangles = [90.0]
        for rotangle in candidateangles:
            objmat4 = copy.deepcopy(objgoalhomomat)
            objmat4[:3, :3] = np.dot(rm.rodrigues(rotax, rotangle), objgoalhomomat[:3, :3])
            objcmcopy.sethomomat(objmat4)
            for idavailableinit in availablegraspsatinit:
                predefined_jawwidth, predefined_fc, predefined_homomat = predefinedgrasps[idavailableinit]
                hndmat4 = np.dot(objmat4, predefined_homomat)
                eepos = rm.homotransformpoint(objmat4, predefined_fc)[:3]
                eerot = hndmat4[:3, :3]
                hndnew = hndfa.genHand()
                hndnew.sethomomat(hndmat4)
                hndcmlist = hndnew.genrangecmlist(jawwidthstart=predefined_jawwidth, jawwidthend=hndnew.jawwidthopen)
                # hndnew.setjawwidth(predefined_jawwidth)
                ishndcollided = bcdchecker.isMeshListMeshListCollided(hndcmlist, obscmlist)
                if not ishndcollided:
                    armjnts = rbt.numik(eepos, eerot, armname)
                    if armjnts is not None:
                        if toggledebug:
                            hndnew = hndfa.genHand()
                            hndnew.setColor(0, 1, 0, .5)
                            hndnew.sethomomat(hndmat4)
                            hndnew.setjawwidth(predefined_jawwidth)
                            # hndnew.reparentTo(self.rhx.base.render)
                        rbt.movearmfk(armjnts, armname)
                        isrbtcollided = pcdchecker.isRobotCollided(rbt, obscmlist, holdarmname=armname)
                        # isobjcollided = pcdchecker.isObjectsOthersCollided([obj], rbt, armname, obscmlist)
                        isobjcollided = False
                        if (not isrbtcollided) and (not isobjcollided):
                            finalsharedgrasps.append(idavailableinit)
                            if toggledebug:
                                rbtmg.genmnp(rbt, togglejntscoord=False, toggleendcoord=False,
                                             rgbargt=[0, 1, 0, .5]).reparentTo(
                                    self.rhx.base.render)
                                # rbtmg.genmnp(rbt, togglejntscoord=False).reparentTo(self.rhx.base.render)
                                # # toggle the following one in case both the common start and goal shall be rendered
                                # hndmat4 = np.dot(objinithomomat, predefined_homomat)
                                # # hndnew = hndfa.genHand()
                                # # hndnew.setColor(0, 1, 0, .5)
                                # # hndnew.sethomomat(hndmat4)
                                # # hndnew.setjawwidth(predefined_jawwidth)
                                # # hndnew.reparentTo(self.rhx.base.render)
                                # eepos = rm.homotransformpoint(objinithomomat, predefined_fc)[:3]
                                # eerot = hndmat4[:3, :3]
                                # armjnts = rbt.numik(eepos, eerot, armname)
                                # rbt.movearmfk(armjnts, armname)
                                # rbtmg.genmnp(rbt, togglejntscoord=False, toggleendcoord=False, rgbargt=[0, 1, 0, .5]).reparentTo(
                                #     self.rhx.base.render)
                                # # base.run()
                                # rbtmg.genmnp(rbt, togglejntscoord=False).reparentTo(self.rhx.base.render)
                                pass
                        elif (not isobjcollided):
                            if toggledebug:
                                # rbtmg.genmnp(rbt, drawhand=False, togglejntscoord=False, toggleendcoord=False,
                                #              rgbargt=[1, 0, 1, .5]).reparentTo(self.rhx.base.render)
                                pass
                    else:
                        ikfailedgraspsnum += 1
                        if toggledebug:
                            hndnew = hndfa.genHand()
                            hndnew.setColor(1, .6, 0, .7)
                            hndnew.sethomomat(hndmat4)
                            hndnew.setjawwidth(predefined_jawwidth)
                            # hndnew.reparentTo(self.rhx.base.render)
                else:
                    ikcolliedgraspsnum += 1
                    if toggledebug:
                        hndnew = hndfa.genHand()
                        hndnew.setColor(1, .5, .5, .7)
                        hndnew.sethomomat(hndmat4)
                        hndnew.setjawwidth(predefined_jawwidth)
                        # hndnew.reparentTo(self.rhx.base.render)
            print("IK failed grasps at the goal pose: ", ikfailedgraspsnum)
            print("Collision at the goal pose: ", ikcolliedgraspsnum)
            print("Possble number of shared grasps: ", len(finalsharedgrasps))
            if toggledebug:
                base.run()

            if len(finalsharedgrasps) == 0:
                print("No shared grasps at symmetric angle " + str(rotangle) + ", trying the next...")
                if rotangle == candidateangles[-1]:
                    print("No shared grasps!")
                    return [None, None, None]
                continue
            else:
                resultinghomomat = copy.deepcopy(objmat4)
                resultinghomomat[:3, 3] = resultinghomomat[:3, 3] + resultinghomomat[:3,
                                                                    2] * 5  # move 5mm up, do not move until end
                # if toggledebug:
                #     for idsharedgrasps in finalsharedgrasps:
                #         predefined_jawwidth, predefined_fc, predefined_homomat = predefinedgrasps[idsharedgrasps]
                #         hndmat4 = np.dot(objinithomomat, predefined_homomat)
                #         eepos = rm.homotransformpoint(objinithomomat, predefined_fc)[:3]
                #         eerot = hndmat4[:3, :3]
                #         armjnts = rbt.numik(eepos, eerot, armname)
                #         rbt.movearmfk(armjnts, armname)
                #         rbtmg.genmnp(rbt, togglejntscoord=False).reparentTo(self.rhx.base.render)

                if rbtinitarmjnts is not None and rbtgoalarmjnts is not None:
                    if armname is "rgt":
                        if not np.isclose(rbtinitarmjnts[1], rbtgoalarmjnts[1]):
                            print("The lft arm must maintain unmoved during generating ppsglmotion for the rgt arm.")
                            raise ValueError("")
                    elif armname is "lft":
                        if not np.isclose(rbtinitarmjnts[0], rbtgoalarmjnts[0]):
                            print("The rgt arm must maintain unmoved during generating ppsglmotion for the lft arm.")
                            raise ValueError("")
                    else:
                        print("Wrong armname. Must be rgt or lft.")
                        raise ValueError("")
                if rbtinitarmjnts is not None:
                    rbt.movearmfk(rbtinitarmjnts[0], armname="rgt")
                    rbt.movearmfk(rbtinitarmjnts[1], armname="lft")
                    rbt.opengripper(armname="rgt")
                    rbt.opengripper(armname="lft")
                else:
                    rbt.movearmfk(rbt.initrgtjnts, armname="rgt")
                    rbt.movearmfk(rbt.initlftjnts, armname="lft")
                    rbt.opengripper(armname="rgt")
                    rbt.opengripper(armname="lft")
                initpose = rbt.getarmjnts(armname=armname)
                if rbtgoalarmjnts is None:
                    goalpose = rbt.initrgtjnts if armname is "rgt" else rbt.initlftjnts
                else:
                    goalpose = rbtgoalarmjnts[0] if armname is "rgt" else rbtgoalarmjnts[1]

                if primitivedirection_init_forward is None:
                    # primitivedirection_init_forward = np.array([0, 0, -1])
                    primitivedirection_init_forward = -objinithomomat[:3, 2]
                if primitivedirection_init_backward is None:
                    # primitivedirection_init_backward = np.array([0, 0, 1])
                    primitivedirection_init_backward = objinithomomat[:3, 2]
                if primitivedirection_final_forward is None:
                    # primitivedirection_final_forward = np.array([0, 0, -1])
                    primitivedirection_final_forward = -objgoalhomomat[:3, 2]
                if primitivedirection_final_backward is None:
                    # primitivedirection_final_backward = np.array([0, 0, 1])
                    primitivedirection_final_backward = objgoalhomomat[:3, 2]

                initpickup = None
                goalpickup = None
                for candidatepredgid in finalsharedgrasps:
                    print("picking up...")
                    initpickup = self.pickup(objinithomomat, predefinedgrasps[candidatepredgid],
                                             primitivedirection_forward=primitivedirection_init_forward,
                                             primitivedirection_backward=primitivedirection_init_backward,
                                             primitivedistance_forward=primitivedistance_init_foward,
                                             primitivedistance_backward=premitivedistance_init_backward,
                                             armname=armname, obstaclecmlist=obscmlist)
                    if initpickup is None:
                        if toggledebug and rotangle == candidateangles[-1]:
                            predefinedjawwidth, predefinedhndfc, predefinedhandhomomat = predefinedgrasps[
                                candidatepredgid]
                            hndmat4 = np.dot(objinithomomat, predefinedhandhomomat)
                            eepos = rm.homotransformpoint(objinithomomat, predefinedhndfc)
                            eerot = hndmat4[:3, :3]
                            amjnts = self.rhx.rbt.numik(eepos, eerot, armname)
                            rbt.movearmfk(armjnts, armname="rgt")
                            if amjnts is not None:
                                pickmotion = self.rhx.genmoveforwardmotion(primitivedirection_init_forward,
                                                                           primitivedistance_init_foward,
                                                                           amjnts, armname)
                                if pickmotion is None:
                                    rbtmg.genmnp(rbt, togglejntscoord=False, toggleendcoord=False,
                                                 rgbargt=np.array([1, 0, 0, .7])).reparentTo(self.rhx.base.render)
                                    if candidatepredgid == finalsharedgrasps[-1]:
                                        base.run()
                            rbt.movearmfk(rbt.initrgtjnts, armname="rgt")
                            rbt.movearmfk(rbt.initlftjnts, armname="lft")
                        continue
                    print("placing down...")
                    goalpickup = self.pickup(resultinghomomat, predefinedgrasps[candidatepredgid],
                                             primitivedirection_forward=primitivedirection_final_forward,
                                             primitivedirection_backward=primitivedirection_final_backward,
                                             primitivedistance_forward=primitivedistance_final_foward,
                                             primitivedistance_backward=premitivedistance_final_backward,
                                             armname=armname, msc=initpickup[1][-1], obstaclecmlist=obscmlist)
                    if goalpickup is None:
                        if toggledebug and rotangle == candidateangles[-1]:
                            predefinedjawwidth, predefinedhndfc, predefinedhandhomomat = predefinedgrasps[
                                candidatepredgid]
                            hndmat4 = np.dot(objinithomomat, predefinedhandhomomat)
                            eepos = rm.homotransformpoint(objinithomomat, predefinedhndfc)
                            eerot = hndmat4[:3, :3]
                            amjnts = self.rhx.rbt.numikmsc(eepos, eerot, initpickup[1][-1], armname)
                            rbt.movearmfk(armjnts, armname="rgt")
                            if amjnts is not None:
                                pickmotion = self.rhx.genmoveforwardmotion(primitivedirection_init_forward,
                                                                           primitivedistance_init_foward,
                                                                           amjnts, armname)
                                if pickmotion is None:
                                    # rbtmg.genmnp(rbt, togglejntscoord=False, toggleendcoord=False, rgbargt=np.array([1, 0, 0, .7])).reparentTo(self.rhx.base.render)
                                    if candidatepredgid == finalsharedgrasps[-1]:
                                        base.run()
                            rbt.movearmfk(rbt.initrgtjnts, armname="rgt")
                            rbt.movearmfk(rbt.initlftjnts, armname="lft")
                        continue
                    initpick, initup = initpickup
                    goalpick, goalup = goalpickup
                    absinitpos = objinithomomat[:3, 3]
                    absinitrot = objinithomomat[:3, :3]
                    absgoalpos = resultinghomomat[:3, 3]
                    absgoalrot = resultinghomomat[:3, :3]
                    relpos, relrot = self.rhx.rbt.getinhandpose(absinitpos, absinitrot, initpick[-1], armname)
                    if userrt:
                        rrtpathinit = self.rhx.planmotion(initpose, initpick[0], obscmlist, armname)
                        if rrtpathinit is None:
                            print("No path found for moving from rbtinitarmjnts to pick up!")
                            continue
                        rrtpathhold = self.rhx.planmotionhold(initup[-1], goalpick[0], self.objcm, relpos, relrot,
                                                              obscmlist, armname)
                        if rrtpathhold is None:
                            print("No path found for moving from pick up to place down!")
                            continue
                        if finalstate[0] is "i":
                            rrtpathgoal = self.rhx.planmotion(goalup[-1], goalpose, obscmlist, armname)
                            if rrtpathgoal is None:
                                print("No path found for moving from place down to rbtgoalarmjnts!")
                                continue
                    jawwidthopen = hndfa.jawwidthopen
                    jawwidthclose = predefinedgrasps[candidatepredgid][0]
                    break

                numikmsmp = []
                jawwidthmsmp = []
                objmsmp = []
                remainingarmjnts = rbt.getarmjnts(armname="rgt") if armname is "lft" else rbt.getarmjnts(armname="lft")
                if initpickup is not None and goalpickup is not None:
                    if userrt:
                        if rrtpathinit is not None and rrtpathhold is not None:
                            if finalstate[0] in ["i", "u"]:
                                numikmp, jawwidthmp, objmp = self.formulatemotionpath(rrtpathinit, armname,
                                                                                      rbt.getjawwidth(armname),
                                                                                      absinitpos,
                                                                                      absinitrot, remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpath(initpick, armname,
                                                                                      rbt.getjawwidth(armname),
                                                                                      absinitpos,
                                                                                      absinitrot, remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpath([initpick[-1]], armname,
                                                                                      jawwidthclose,
                                                                                      absinitpos, absinitrot,
                                                                                      remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(initup, armname,
                                                                                          jawwidthclose,
                                                                                          relpos, relrot,
                                                                                          remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(rrtpathhold, armname,
                                                                                          jawwidthclose,
                                                                                          relpos, relrot,
                                                                                          remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(goalpick, armname,
                                                                                          jawwidthclose,
                                                                                          relpos, relrot,
                                                                                          remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpath([goalpick[-1]], armname,
                                                                                      jawwidthopen,
                                                                                      absgoalpos, absgoalrot,
                                                                                      remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpath(goalup, armname, jawwidthopen,
                                                                                      absgoalpos,
                                                                                      absgoalrot, remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                if finalstate[0] is "i":
                                    numikmp, jawwidthmp, objmp = self.formulatemotionpath(rrtpathgoal, armname,
                                                                                          jawwidthopen,
                                                                                          absgoalpos, absgoalrot,
                                                                                          remainingarmjnts)
                                    numikmsmp.append(numikmp)
                                    jawwidthmsmp.append(jawwidthmp)
                                    objmsmp.append(objmp)
                            else:
                                finaljawwidth = jawwidthopen if finalstate[1] is "o" else jawwidthclose
                                numikmp, jawwidthmp, objmp = self.formulatemotionpath(rrtpathinit, armname,
                                                                                      rbt.getjawwidth(armname),
                                                                                      absinitpos,
                                                                                      absinitrot, remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpath(initpick, armname,
                                                                                      rbt.getjawwidth(armname),
                                                                                      absinitpos,
                                                                                      absinitrot, remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpath([initpick[-1]], armname,
                                                                                      jawwidthclose,
                                                                                      absinitpos, absinitrot,
                                                                                      remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(initup, armname,
                                                                                          jawwidthclose,
                                                                                          relpos, relrot,
                                                                                          remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(rrtpathhold, armname,
                                                                                          jawwidthclose,
                                                                                          relpos, relrot,
                                                                                          remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(goalpick, armname,
                                                                                          jawwidthclose,
                                                                                          relpos, relrot,
                                                                                          remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpath([goalpick[-1]], armname,
                                                                                      finaljawwidth,
                                                                                      absgoalpos, absgoalrot,
                                                                                      remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                        else:
                            print("There is possible sequence but no collision free motion!")
                    else:
                        if finalstate[0] in ["i", "u"]:
                            numikmp, jawwidthmp, objmp = self.formulatemotionpath([initpose], armname,
                                                                                  rbt.getjawwidth(armname),
                                                                                  absinitpos, absinitrot,
                                                                                  remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            numikmp, jawwidthmp, objmp = self.formulatemotionpath(initpick, armname,
                                                                                  rbt.getjawwidth(armname),
                                                                                  absinitpos, absinitrot,
                                                                                  remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            numikmp, jawwidthmp, objmp = self.formulatemotionpath([initpick[-1]], armname,
                                                                                  jawwidthclose,
                                                                                  absinitpos, absinitrot,
                                                                                  remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(initup, armname, jawwidthclose,
                                                                                      relpos,
                                                                                      relrot, remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(goalpick, armname, jawwidthclose,
                                                                                      relpos,
                                                                                      relrot, remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            numikmp, jawwidthmp, objmp = self.formulatemotionpath([goalpick[-1]], armname, jawwidthopen,
                                                                                  absgoalpos, absgoalrot,
                                                                                  remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            numikmp, jawwidthmp, objmp = self.formulatemotionpath(goalup, armname, jawwidthopen,
                                                                                  absgoalpos,
                                                                                  absgoalrot, remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            if finalstate[0] is "i":
                                numikmp, jawwidthmp, objmp = self.formulatemotionpath([goalpose], armname, jawwidthopen,
                                                                                      absgoalpos, absgoalrot,
                                                                                      remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                        else:
                            finaljawwidth = jawwidthopen if finalstate[1] is "o" else jawwidthclose
                            numikmp, jawwidthmp, objmp = self.formulatemotionpath([initpose], armname,
                                                                                  rbt.getjawwidth(armname),
                                                                                  absinitpos, absinitrot,
                                                                                  remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            numikmp, jawwidthmp, objmp = self.formulatemotionpath(initpick, armname,
                                                                                  rbt.getjawwidth(armname),
                                                                                  absinitpos, absinitrot,
                                                                                  remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            numikmp, jawwidthmp, objmp = self.formulatemotionpath([initpick[-1]], armname,
                                                                                  jawwidthclose,
                                                                                  absinitpos, absinitrot,
                                                                                  remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(initup, armname, jawwidthclose,
                                                                                      relpos,
                                                                                      relrot, remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(goalpick, armname, jawwidthclose,
                                                                                      relpos,
                                                                                      relrot, remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            numikmp, jawwidthmp, objmp = self.formulatemotionpath([goalpick[-1]], armname,
                                                                                  finaljawwidth,
                                                                                  absgoalpos, absgoalrot,
                                                                                  remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                    rbt.movearmfk(bk_armjnts_rgt, armname="rgt")
                    rbt.movearmfk(bk_armjnts_lft, armname="lft")
                    rbt.opengripper(armname="rgt", jawwidth=bk_jawwidth_rgt)
                    rbt.opengripper(armname="lft", jawwidth=bk_jawwidth_lft)

                    if toggledebug:
                        # draw motion
                        for i, numikms in enumerate(numikmsmp):
                            for j, numik in enumerate(numikms):
                                rbt.movearmfk(numik[1], armname="rgt")
                                rbt.movearmfk(numik[2], armname="lft")
                                rbt.opengripper(armname="rgt", jawwidth=jawwidthmsmp[i][j][0])
                                rbt.opengripper(armname="lft", jawwidth=jawwidthmsmp[i][j][1])
                                if j == 0 or j == len(numikms):
                                    rbtmg.genmnp(rbt, togglejntscoord=False, toggleendcoord=False,
                                                 rgbargt=np.array([0, 1, 0, 1])).reparentTo(self.rhx.base.render)
                                else:
                                    rbtmg.genmnp(rbt, togglejntscoord=False, toggleendcoord=False,
                                                 rgbargt=np.array([.5, .5, .5, .5])).reparentTo(self.rhx.base.render)
                                if j < len(numikms) - 1:
                                    numiknxt = numikms[j + 1][1]
                                    thispos, _ = rbt.getee(armname="rgt")
                                    rbt.movearmfk(numiknxt, armname="rgt")
                                    nxtpos, _ = rbt.getee(armname="rgt")
                                    p3dh.genlinesegnodepath([thispos, nxtpos], colors=[1, 0, 1, 1],
                                                            thickness=3.7).reparentTo(base.render)
                                objcp = copy.deepcopy(self.objcm)
                                objcp.sethomomat(objmsmp[i][j])
                                objcp.setColor(1, 1, 0, 1)
                                objcp.reparentTo(base.render)
                        base.run()

                    return [numikmsmp, jawwidthmsmp, objmsmp]
                else:
                    rbt.movearmfk(bk_armjnts_rgt, armname="rgt")
                    rbt.movearmfk(bk_armjnts_lft, armname="lft")
                    rbt.opengripper(armname="rgt", jawwidth=bk_jawwidth_rgt)
                    rbt.opengripper(armname="lft", jawwidth=bk_jawwidth_lft)
                    if initpickup is None:
                        print("No feasible motion for pick up!")
                    if goalpickup is None:
                        print("No feasible motion for placing down!")
                    print("The shared grasps failed at symmetric angle " + str(rotangle) + ", trying the next...")
                    continue

        # if toggledebug:
        #     base.run()
        print("No feasible motion between two key poses!")
        return [None, None, None]

    def formulatemotionpath(self, motionpath, armname, jawwidth, objpos, objrot, remainingarmjnts):
        """

        :param motionpath:
        :param jawwidth:
        :param relpos:
        :param relrot:
        :param remainingarmjnts
        :param armname:
        :return:

        author: weiwei
        date: 20200105
        """

        numikmp = []
        jawwidthmp = []
        objmp = []

        rbt = self.rhx.rbt
        rm = self.rhx.rm

        for elemp in motionpath:
            if armname is "rgt":
                lftelemp = remainingarmjnts
                rgtelemp = np.asarray(elemp)
                lftelejw = rbt.getjawwidth("lft")
                rgtelejw = jawwidth
            elif armname is "lft":
                lftelemp = np.asarray(elemp)
                rgtelemp = remainingarmjnts
                lftelejw = jawwidth
                rgtelejw = rbt.getjawwidth("rgt")
            numikmp.append(np.array([0, rgtelemp, lftelemp]))
            jawwidthmp.append([rgtelejw, lftelejw])
            objmp.append(rm.homobuild(objpos, objrot))

        return [numikmp, jawwidthmp, objmp]

    def formulatemotionpathhold(self, motionpath, armname, jawwidth, relpos, relrot, remainingarmjnts):
        """

        :param motionpath:
        :param jawwidth:
        :param relpos:
        :param relrot:
        :param remainingarmjnts
        :param armname:
        :return:

        author: weiwei
        date: 20200105
        """

        numikmp = []
        jawwidthmp = []
        objmp = []

        rbt = self.rhx.rbt
        rm = self.rhx.rm
        bk_armjnts = rbt.getarmjnts(armname=armname)

        for elemp in motionpath:
            if armname is "rgt":
                lftelemp = remainingarmjnts
                rgtelemp = np.asarray(elemp)
                lftelejw = rbt.getjawwidth("lft")
                rgtelejw = jawwidth
            elif armname is "lft":
                rgtelemp = remainingarmjnts
                lftelemp = np.asarray(elemp)
                lftelejw = jawwidth
                rgtelejw = rbt.getjawwidth("rgt")
            numikmp.append(np.array([0, rgtelemp, lftelemp]))
            jawwidthmp.append([rgtelejw, lftelejw])
            rbt.movearmfk(elemp, armname=armname)
            objpos, objrot = rbt.getworldpose(relpos, relrot, armname=armname)
            objmp.append(rm.homobuild(objpos, objrot))
        rbt.movearmfk(bk_armjnts, armname=armname)

        return [numikmp, jawwidthmp, objmp]


if __name__ == '__main__':
    import time
    import basis.robot_math as rm
    import robotsim.robots.yumi.yumi as ym
    import visualization.panda.world as wd
    import modeling.geometricmodel as gm
    import grasping.annotation.utils as gutil

    base = wd.World(campos=[2, 0, 1.5], lookatpos=[0, 0, .2])
    gm.gen_frame().attach_to(base)
    objcm = cm.CollisionModel('tubebig.stl')
    yumi_instance = ym.Yumi(enable_cc=True)
    manipulator_name = 'rgt_arm'
    hnd_name = 'rgt_hnd'
    goal_homomat_list = []
    for i in range(6):
        goal_pos = np.array([.55, -.1, .3]) - np.array([i * .1, i * .1, 0])
        # goal_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
        goal_rotmat = np.eye(3)
        goal_homomat_list.append(rm.homomat_from_posrot(goal_pos, goal_rotmat))
        tmp_objcm = objcm.copy()
        tmp_objcm.set_homomat(rm.homomat_from_posrot(goal_pos, goal_rotmat))
        tmp_objcm.attach_to(base)
    grasp_info_list = gutil.load_pickle_file(objcm_name='tubebig', file_name='preannotated_grasps.pickle')
    ppp = PickPlacePlanner(robot=yumi_instance)
    common_grasp_id_list, _ = ppp.find_common_graspids(manipulator_name,
                                                       hnd_name,
                                                       grasp_info_list,
                                                       goal_homomat_list)
    grasp_info = grasp_info_list[common_grasp_id_list[0]]
    conf_list = []
    jawwidth_list = []
    objpose_list = []
    start_conf = yumi_instance.get_jnt_values(manipulator_name)
    for i in range(len(goal_homomat_list)-1):
        goal_homomat0 = goal_homomat_list[i]
        obj_pos0 = goal_homomat0[:3, 3]
        obj_rotmat0 = goal_homomat0[:3, :3]
        conf_list0, jawwidth_list0, objpose_list0 = ppp.gen_pickup_motion(manipulator_name,
                                                                          hnd_name,
                                                                          objcm,
                                                                          grasp_info,
                                                                          goal_obj_pos=obj_pos0,
                                                                          goal_obj_rotmat=obj_rotmat0,
                                                                          start_conf=start_conf,
                                                                          approach_direction=np.array([0,0,-1]),
                                                                          approach_distance=.1,
                                                                          depart_direction=np.array([0,0,1]),
                                                                          depart_distance=.2)
        goal_homomat1 = goal_homomat_list[i+1]
        obj_pos1 = goal_homomat1[:3, 3]
        obj_rotmat1 = goal_homomat1[:3, :3]
        conf_list1, jawwidth_list1, objpose_list1 = ppp.gen_placedown_motion(manipulator_name,
                                                                          hnd_name,
                                                                          objcm,
                                                                          grasp_info,
                                                                          goal_obj_pos=obj_pos1,
                                                                          goal_obj_rotmat=obj_rotmat1,
                                                                          start_conf=conf_list0[-1],
                                                                          approach_direction=np.array([0,0,-1]),
                                                                          approach_distance=.1,
                                                                          depart_direction=np.array([.5,.5,1]),
                                                                          depart_distance=.2)
        start_conf=conf_list1[-1]
        # objcm.set_pos(objpose_list1[-1][3,:3])
        # objcm.set_rotmat(objpose_list1[-1][:3,:3])
        conf_list = conf_list+conf_list0+conf_list1
        jawwidth_list = jawwidth_list+jawwidth_list0+jawwidth_list1
        objpose_list= objpose_list+objpose_list0+objpose_list1
        # break
    for i, conf in enumerate(conf_list):
        yumi_instance.fk(manipulator_name, conf)
        yumi_instance.jaw_to(hnd_name, jawwidth_list[i])
        yumi_instance.gen_meshmodel().attach_to(base)
        tmp_objcm = objcm.copy()
        tmp_objcm.set_homomat(objpose_list[i])
        tmp_objcm.attach_to(base)
    # hnd_instance = yumi_instance.hnd_dict[hnd_name].copy()
    # for goal_homomat in goal_homomat_list:
    #     obj_pos = goal_homomat[:3, 3]
    #     obj_rotmat = goal_homomat[:3, :3]
    #     for grasp_id in common_grasp_id_list:
    #         jawwidth, tcp_pos, hnd_pos, hnd_rotmat = grasp_info_list[grasp_id]
    #         new_tcp_pos = obj_rotmat.dot(tcp_pos) + obj_pos
    #         new_hnd_pos = obj_rotmat.dot(hnd_pos) + obj_pos
    #         new_hnd_rotmat = obj_rotmat.dot(hnd_rotmat)
    #         tmp_hnd = hnd_instance.copy()
    #         tmp_hnd.fix_to(new_hnd_pos, new_hnd_rotmat)
    #         tmp_hnd.jaw_to(jawwidth)
    #         tmp_hnd.gen_meshmodel().attach_to(base)
    #         jnt_values = yumi_instance.ik(manipulator_name,
    #                                       new_tcp_pos,
    #                                       new_hnd_rotmat)
    #         try:
    #             yumi_instance.fk(manipulator_name, jnt_values)
    #             yumi_instance.gen_meshmodel().attach_to(base)
    #         except:
    #             continue
    base.run()
    # goal_pos = np.array([.55, -.1, .3])
    # goal_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    # gm.gen_frame(pos=goal_pos, rotmat=goal_rotmat).attach_to(base)
