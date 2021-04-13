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

    def __init__(self, robot_s):
        """
        :param object:
        :param robot_helper:
        author: weiwei, hao
        date: 20191122, 20210113
        """
        super().__init__(robot_s)

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
                gl_obj_pos, gl_obj_rotmat = self.rbt.cvt_loc_tcp_to_gl(component_name, obj_pos, obj_rotmat)
                objpose_list.append(rm.homomat_from_posrot(gl_obj_pos, gl_obj_rotmat))
            self.rbt.fk(component_name, jnt_values_bk)
        else:
            raise ValueError('Type must be absolute or relative!')
        return objpose_list

    def find_common_graspids(self,
                             manipulator_name,
                             hnd_name,
                             grasp_info_list,
                             goal_homomat_list,
                             obstacle_list=[],
                             toggle_debug=False):
        """
        find the common collision free and IK feasible graspids
        :param manipulator_name:
        :param hnd_name: a component may have multiple hands
        :param grasp_info_list: a list like [[jaw_width, gl_jaw_center, pos, rotmat], ...]
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
        jnt_values_bk = self.rbt.get_jnt_values(manipulator_name)
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
                    jnt_values = self.rbt.ik(manipulator_name, gl_hnd_pos, gl_hnd_rotmat)
                    if jnt_values is not None:  # common graspid consdiering robot_s ik
                        if toggle_debug:
                            hnd_tmp = hnd_instance.copy()
                            hnd_tmp.gen_meshmodel(rgba=[0, 1, 0, .2]).attach_to(base)
                        self.rbt.fk(manipulator_name, jnt_values)
                        is_rbt_collided = self.rbt.is_collided(obstacle_list)  # common graspid consdiering robot_s cd
                        # TODO is_obj_collided
                        is_obj_collided = False  # common graspid consdiering obj cd
                        if (not is_rbt_collided) and (
                                not is_obj_collided):  # hnd cdfree, robot_s ikfeasible, robot_s cdfree
                            if toggle_debug:
                                self.rbt.gen_meshmodel(rgba=[0, 1, 0, .5]).attach_to(base)
                            previously_available_graspids.append(graspid)
                        elif (not is_obj_collided):  # hnd cdfree, robot_s ikfeasible, robot_s collided
                            rbtcollided_grasps_num += 1
                            if toggle_debug:
                                self.rbt.gen_meshmodel(rgba=[1, 0, 1, .5]).attach_to(base)
                    else:  # hnd cdfree, robot_s ik infeasible
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
        self.rbt.fk(manipulator_name, jnt_values_bk)
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
        self.rbt.fk(self.rbt.ik(component_name, start_hnd_pos, start_hnd_rotmat, seed_jnt_values=seed_jnt_values))
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
        start_conf = self.rbt.ik(component_name, start_hnd_pos, start_hnd_rotmat, seed_jnt_values=seed_jnt_values)
        goal_conf = self.rbt.ik(component_name, goal_hnd_pos, goal_hnd_rotmat, seed_jnt_values=seed_jnt_values)
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
        :param component_name:
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
        degenerate into gen_pickup_primitive if both seed_jnt_values and goal_conf are None
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
        degenerate into gen_pickup_primitive if both seed_jnt_values and goal_conf are None
        :param component_name:
        :param hnd_name:
        :param objcm:
        :param grasp_info:
        :param goal_obj_pos:
        :param goal_obj_rotmat:
        :param seed_jnt_values:
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
        degenerate into gen_pickup_primitive if both seed_jnt_values and goal_conf are None
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

    def gen_pick_and_place_motion(self,
                                  manipulator_name,
                                  hand_name,
                                  objcm,
                                  grasp_info_list,
                                  start_conf,
                                  goal_homomat_list):
        """

        :param manipulator_name:
        :param hand_name:
        :param grasp_info_list: a list like [[jaw_width, gl_jaw_center, pos, rotmat], ...]
        :param start_conf:
        :param goal_homomat_list: a list of tcp goals like [homomat0, homomat1, ...]
        :return:
        author: weiwei
        date: 20191122, 20200105
        """
        common_grasp_id_list, _ = self.find_common_graspids(manipulator_name,
                                                            hand_name,
                                                            grasp_info_list,
                                                            goal_homomat_list)
        grasp_info = grasp_info_list[common_grasp_id_list[0]]
        conf_list = []
        jawwidth_list = []
        objpose_list = []
        for i in range(len(goal_homomat_list)-1):
            goal_homomat0 = goal_homomat_list[i]
            obj_pos0 = goal_homomat0[:3, 3]
            obj_rotmat0 = goal_homomat0[:3, :3]
            conf_list0, jawwidth_list0, objpose_list0 = self.gen_pickup_motion(manipulator_name,
                                                                               hand_name,
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
            conf_list1, jawwidth_list1, objpose_list1 = self.gen_placedown_motion(manipulator_name,
                                                                                  hand_name,
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
            conf_list = conf_list+conf_list0+conf_list1
            jawwidth_list = jawwidth_list+jawwidth_list0+jawwidth_list1
            objpose_list= objpose_list+objpose_list0+objpose_list1
        return conf_list, jawwidth_list, objpose_list



if __name__ == '__main__':
    import time
    import basis.robot_math as rm
    import robotsim.robots.yumi.yumi as ym
    import visualization.panda.world as wd
    import modeling.geometricmodel as gm
    import grasping.annotation.utils as gutil

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    gm.gen_frame().attach_to(base)
    objcm = cm.CollisionModel('tubebig.stl')
    yumi_s = ym.Yumi(enable_cc=True)
    manipulator_name = 'rgt_arm'
    hand_name = 'rgt_hnd'
    start_conf = yumi_s.get_jnt_values(manipulator_name)
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
    pp_planner = PickPlacePlanner(robot_s=yumi_s)
    conf_list, jawwidth_list, objpose_list = pp_planner.gen_pick_and_place_motion(manipulator_name,
                                                                                  hand_name,
                                                                                  objcm,
                                                                                  grasp_info_list,
                                                                                  start_conf,
                                                                                  goal_homomat_list)
    for i, conf in enumerate(conf_list):
        yumi_s.fk(manipulator_name, conf)
        yumi_s.jaw_to(hand_name, jawwidth_list[i])
        yumi_s.gen_meshmodel().attach_to(base)
        tmp_objcm = objcm.copy()
        tmp_objcm.set_homomat(objpose_list[i])
        tmp_objcm.attach_to(base)
    # hnd_instance = yumi_s.hnd_dict[hand_name].copy()
    # for goal_homomat in goal_homomat_list:
    #     obj_pos = goal_homomat[:3, 3]
    #     obj_rotmat = goal_homomat[:3, :3]
    #     for grasp_id in common_grasp_id_list:
    #         jaw_width, tcp_pos, hnd_pos, hnd_rotmat = grasp_info_list[grasp_id]
    #         new_tcp_pos = obj_rotmat.dot(tcp_pos) + obj_pos
    #         new_hnd_pos = obj_rotmat.dot(hnd_pos) + obj_pos
    #         new_hnd_rotmat = obj_rotmat.dot(hnd_rotmat)
    #         tmp_hnd = hnd_instance.copy()
    #         tmp_hnd.fix_to(new_hnd_pos, new_hnd_rotmat)
    #         tmp_hnd.jaw_to(jaw_width)
    #         tmp_hnd.gen_meshmodel().attach_to(base)
    #         jnt_values = yumi_s.ik(component_name,
    #                                       new_tcp_pos,
    #                                       new_hnd_rotmat)
    #         try:
    #             yumi_s.fk(component_name, jnt_values)
    #             yumi_s.gen_meshmodel().attach_to(base)
    #         except:
    #             continue
    # goal_pos = np.array([.55, -.1, .3])
    # goal_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    # gm.gen_frame(pos=goal_pos, rotmat=goal_rotmat).attach_to(base)
    base.run()
