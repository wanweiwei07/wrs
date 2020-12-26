import os
import copy
import math
import numpy as np
import basis.robotmath as rm
import modeling.modelcollection as mc
import modeling.collisionmodel as cm
import robotsim._kinematics.jlchain as jl
import robotsim.manipulators.irb14050.irb14050 as ya
import robotsim.grippers.yumi_gripper.yumi_gripper as yg
from panda3d.core import CollisionNode, CollisionBox, Point3
import robotsim.robots.robot_interface as ri


class Yumi(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='yumi', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        # lft
        self.lft_body = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=np.zeros(7), name='lft_body')
        self.lft_body.jnts[1]['loc_pos'] = np.array([0, 0, 0])
        self.lft_body.jnts[2]['loc_pos'] = np.array([0, 0, 0])
        self.lft_body.jnts[3]['loc_pos'] = np.array([0, 0, 0])
        self.lft_body.jnts[4]['loc_pos'] = np.array([0, 0, 0])
        self.lft_body.jnts[5]['loc_pos'] = np.array([0, 0, 0])
        self.lft_body.jnts[6]['loc_pos'] = np.array([0, 0, 0])
        self.lft_body.jnts[7]['loc_pos'] = np.array([0, 0, 0])
        self.lft_body.jnts[8]['loc_pos'] = np.array([0.05355, -0.0725, 0.41492])
        self.lft_body.jnts[8]['loc_rotmat'] = rm.rotmat_from_euler(-0.9795, -0.5682, -2.3155)  # left from robot view
        self.lft_body.lnks[0]['name'] = "yumi_lft_stand"
        self.lft_body.lnks[0]['loc_pos'] = np.array([0, 0, 0])
        self.lft_body.lnks[0]['meshfile'] = os.path.join(this_dir, "meshes", "yumi_tablenotop.stl")
        self.lft_body.lnks[0]['rgba'] = [.35, .35, .35, 1.0]
        self.lft_body.lnks[1]['name'] = "yumi_lft_body"
        self.lft_body.lnks[1]['loc_pos'] = np.array([0, 0, 0])
        self.lft_body.lnks[1]['collisionmodel'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "body.stl"),
            cdprimitive_type="userdefined", expand_radius=.005,
            userdefined_cdprimitive_fn=self._base_combined_cdnp)
        self.lft_body.lnks[1]['rgba'] = [.7, .7, .7, 1]
        self.lft_body.lnks[2]['name'] = "yumi_lft_column"
        self.lft_body.lnks[2]['loc_pos'] = np.array([-.327, -.24, -1.015])
        self.lft_body.lnks[2]['meshfile'] = os.path.join(this_dir, "meshes", "yumi_column60602100.stl")
        self.lft_body.lnks[2]['rgba'] = [.35, .35, .35, 1.0]
        self.lft_body.lnks[3]['name'] = "yumi_rgt_column"
        self.lft_body.lnks[3]['loc_pos'] = np.array([-.327, .24, -1.015])
        self.lft_body.lnks[3]['meshfile'] = os.path.join(this_dir, "meshes", "yumi_column60602100.stl")
        self.lft_body.lnks[3]['rgba'] = [.35, .35, .35, 1.0]
        self.lft_body.lnks[4]['name'] = "yumi_top_back"
        self.lft_body.lnks[4]['loc_pos'] = np.array([-.327, 0, 1.085])
        self.lft_body.lnks[4]['meshfile'] = os.path.join(this_dir, "meshes", "yumi_column6060540.stl")
        self.lft_body.lnks[4]['rgba'] = [.35, .35, .35, 1.0]
        self.lft_body.lnks[5]['name'] = "yumi_top_lft"
        self.lft_body.lnks[5]['loc_pos'] = np.array([-.027, -.24, 1.085])
        self.lft_body.lnks[5]['loc_rotmat'] = rm.rotmat_from_axangle([0, 0, 1], -math.pi / 2)
        self.lft_body.lnks[5]['meshfile'] = os.path.join(this_dir, "meshes", "yumi_column6060540.stl")
        self.lft_body.lnks[5]['rgba'] = [.35, .35, .35, 1.0]
        self.lft_body.lnks[6]['name'] = "yumi_top_rgt"
        self.lft_body.lnks[6]['loc_pos'] = np.array([-.027, .24, 1.085])
        self.lft_body.lnks[6]['loc_rotmat'] = rm.rotmat_from_axangle([0, 0, 1], -math.pi / 2)
        self.lft_body.lnks[6]['meshfile'] = os.path.join(this_dir, "meshes", "yumi_column6060540.stl")
        self.lft_body.lnks[6]['rgba'] = [.35, .35, .35, 1.0]
        self.lft_body.lnks[7]['name'] = "yumi_top_front"
        self.lft_body.lnks[7]['loc_pos'] = np.array([.273, 0, 1.085])
        self.lft_body.lnks[7]['meshfile'] = os.path.join(this_dir, "meshes", "yumi_column6060540.stl")
        self.lft_body.lnks[7]['rgba'] = [.35, .35, .35, 1.0]
        self.lft_body.reinitialize()
        lft_arm_homeconf = np.radians(np.array([-20, -90, -120, 30, .0, 40, 0]))
        self.lft_arm = ya.IRB14050(pos=self.lft_body.jnts[-1]['gl_posq'],
                                   rotmat=self.lft_body.jnts[-1]['gl_rotmatq'],
                                   homeconf=lft_arm_homeconf, enable_cc=False)
        self.lft_hnd = yg.YumiGripper(pos=self.lft_arm.jnts[-1]['gl_posq'],
                                      rotmat=self.lft_arm.jnts[-1]['gl_rotmatq'],
                                      enable_cc=False)
        # rgt
        self.rgt_body = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=np.zeros(0), name='rgt_body')
        self.rgt_body.jnts[1]['loc_pos'] = np.array([0.05355, 0.07250, 0.41492])
        self.rgt_body.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(0.9781, -0.5716, 2.3180)  # left from robot view
        self.rgt_body.lnks[0]['name'] = "yumi_rgt_body"
        self.rgt_body.lnks[0]['loc_pos'] = np.array([0, 0, 0])
        self.rgt_body.lnks[0]['rgba'] = [.35, .35, .35, 1.0]
        self.rgt_body.reinitialize()
        rgt_arm_homeconf = np.radians(np.array([20, -90, 120, 30, 0, 40, 0]))
        self.rgt_arm = self.lft_arm.copy()
        self.rgt_arm.fix_to(pos=self.rgt_body.jnts[-1]['gl_posq'], rotmat=self.rgt_body.jnts[-1]['gl_rotmatq'])
        self.rgt_arm.set_homeconf(rgt_arm_homeconf)
        self.rgt_arm.goto_homeconf()
        self.rgt_hnd = self.lft_hnd.copy()
        self.rgt_hnd.fix_to(pos=self.rgt_arm.jnts[-1]['gl_posq'], rotmat=self.rgt_arm.jnts[-1]['gl_rotmatq'])
        # collision detection
        if enable_cc:
            self.cc.add_cdlnks(self.lft_body, [0, 1, 2, 3, 4, 5, 6, 7])
            self.cc.add_cdlnks(self.lft_arm, [1, 2, 3, 4, 5, 6])
            self.cc.add_cdlnks(self.lft_hnd.lft, [0, 1])
            self.cc.add_cdlnks(self.lft_hnd.rgt, [1])
            self.cc.add_cdlnks(self.rgt_arm, [1, 2, 3, 4, 5, 6])
            self.cc.add_cdlnks(self.rgt_hnd.lft, [0, 1])
            self.cc.add_cdlnks(self.rgt_hnd.rgt, [1])
            activelist = [self.lft_arm.lnks[1],
                          self.lft_arm.lnks[2],
                          self.lft_arm.lnks[3],
                          self.lft_arm.lnks[4],
                          self.lft_arm.lnks[5],
                          self.lft_arm.lnks[6],
                          self.lft_hnd.lft.lnks[0],
                          self.lft_hnd.lft.lnks[1],
                          self.lft_hnd.rgt.lnks[1],
                          self.rgt_arm.lnks[1],
                          self.rgt_arm.lnks[2],
                          self.rgt_arm.lnks[3],
                          self.rgt_arm.lnks[4],
                          self.rgt_arm.lnks[5],
                          self.rgt_arm.lnks[6],
                          self.rgt_hnd.lft.lnks[0],
                          self.rgt_hnd.lft.lnks[1],
                          self.rgt_hnd.rgt.lnks[1]]
            self.cc.set_active_cdlnks(activelist)
            fromlist = [self.lft_body.lnks[0],  # table
                        self.lft_body.lnks[1],  # body
                        self.lft_arm.lnks[1],
                        self.rgt_arm.lnks[1]]
            intolist = [self.lft_arm.lnks[5],
                        self.lft_arm.lnks[6],
                        self.lft_hnd.lft.lnks[0],
                        self.lft_hnd.lft.lnks[1],
                        self.lft_hnd.rgt.lnks[1],
                        self.rgt_arm.lnks[5],
                        self.rgt_arm.lnks[6],
                        self.rgt_hnd.lft.lnks[0],
                        self.rgt_hnd.lft.lnks[1],
                        self.rgt_hnd.rgt.lnks[1]]
            self.cc.set_cdpair(fromlist, intolist)
            fromlist = [self.lft_arm.lnks[3],
                        self.lft_arm.lnks[4],
                        self.lft_arm.lnks[5],
                        self.lft_arm.lnks[6],
                        self.lft_hnd.lft.lnks[0],
                        self.lft_hnd.lft.lnks[1],
                        self.lft_hnd.rgt.lnks[1]]
            intolist = [self.rgt_arm.lnks[3],
                        self.rgt_arm.lnks[4],
                        self.rgt_arm.lnks[5],
                        self.rgt_arm.lnks[6],
                        self.rgt_hnd.lft.lnks[0],
                        self.rgt_hnd.lft.lnks[1],
                        self.rgt_hnd.rgt.lnks[1]]
            self.cc.set_cdpair(fromlist, intolist)
        # tool center point
        # lft
        self.lft_tcp_jlc = self.lft_arm  # which jlc is the tcp located at?
        self.lft_tcp_jlc.tcp_jntid = -1
        self.lft_tcp_jlc.tcp_loc_pos = np.array([0, 0, .12])
        self.lft_tcp_jlc.tcp_loc_rotmat = np.eye(3)
        # rgt
        self.rgt_tcp_jlc = self.rgt_arm  # which jlc is the tcp located at?
        self.rgt_tcp_jlc.tcp_jntid = -1
        self.rgt_tcp_jlc.tcp_loc_pos = np.array([0, 0, .12])
        self.rgt_tcp_jlc.tcp_loc_rotmat = np.eye(3)
        # a list of detailed information about objects in hand, see CollisionChecker.add_objinhnd
        self.lft_oih_infos = []
        self.rgt_oih_infos = []

    @staticmethod
    def _base_combined_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(-.2, 0, 0.04),
                                              x=.16 + radius, y=.2 + radius, z=.04 + radius)
        collision_node.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(-.24, 0, 0.24),
                                              x=.12 + radius, y=.125 + radius, z=.24 + radius)
        collision_node.addSolid(collision_primitive_c1)
        collision_primitive_c2 = CollisionBox(Point3(-.07, 0, 0.4),
                                              x=.07 + radius, y=.125 + radius, z=.06 + radius)
        collision_node.addSolid(collision_primitive_c2)
        collision_primitive_l0 = CollisionBox(Point3(0, 0.145, 0.03),
                                              x=.135 + radius, y=.055 + radius, z=.03 + radius)
        collision_node.addSolid(collision_primitive_l0)
        collision_primitive_r0 = CollisionBox(Point3(0, -0.145, 0.03),
                                              x=.135 + radius, y=.055 + radius, z=.03 + radius)
        collision_node.addSolid(collision_primitive_r0)
        return collision_node

    @property
    def is_fk_updated(self):
        return self.rgt_hnd.is_fk_updated or self.lft_hnd.is_fk_updated

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.lft_body.fix_to(self.pos, self.rotmat)
        self.lft_arm.fix_to(pos=self.lft_body.jnts[-1]['gl_posq'],
                            rotmat=self.lft_body.jnts[-1]['gl_rotmatq'])
        self.lft_hnd.fix_to(pos=self.lft_arm.jnts[-1]['gl_posq'],
                            rotmat=self.lft_arm.jnts[-1]['gl_rotmatq'])
        self.rgt_arm.fix_to(pos=self.rgt_body.jnts[-1]['gl_posq'],
                            rotmat=self.rgt_body.jnts[-1]['gl_rotmatq'])
        self.rgt_hnd.fix_to(pos=self.rgt_arm.jnts[-1]['gl_posq'],
                            rotmat=self.rgt_arm.jnts[-1]['gl_rotmatq'])

    def fk(self, general_jnt_values, armname='lft'):
        """
        :param general_jnt_values: [nparray, nparray], 7+7, meter-radian
        :armname 'lft', 'rgt', 'both'
        :return:
        author: weiwei
        date: 20201208toyonaka
        """
        # examine length
        if armname == 'lft' or armname == 'rgt':
            if not isinstance(general_jnt_values, np.ndarray) or general_jnt_values.size != 7:
                raise ValueError("An 1x7 npdarray must be specified to move a single arm!")
            if armname == 'lft':
                self.lft_arm.fix_to(pos=self.lft_body.jnts[-1]['gl_posq'],
                                    rotmat=self.lft_body.jnts[-1]['gl_rotmatq'],
                                    jnt_values=general_jnt_values)
            else:
                self.rgt_arm.fix_to(pos=self.rgt_body.jnts[-1]['gl_posq'],
                                    rotmat=self.rgt_body.jnts[-1]['gl_rotmatq'],
                                    jnt_values=general_jnt_values)
        elif armname == 'both':
            if (not isinstance(general_jnt_values, list)
                    or general_jnt_values[0].size != 7
                    or general_jnt_values[1].size != 7):
                raise ValueError("A list of two 1x7 npdarrays must be specified to move both arm!")
            self.lft_arm.fix_to(pos=self.lft_body.jnts[-1]['gl_posq'],
                                rotmat=self.lft_body.jnts[-1]['gl_rotmatq'],
                                jnt_values=general_jnt_values[0])
            self.rgt_arm.fix_to(pos=self.rgt_body.jnts[-1]['gl_posq'],
                                rotmat=self.rgt_body.jnts[-1]['gl_rotmatq'],
                                jnt_values=general_jnt_values[1])

    def jaw_to(self, jawwidth, armname='lft'):
        if armname == 'lft':
            self.lft_hnd.jaw_to(jawwidth)
        elif armname == 'rgt':
            self.rgt_hnd.jaw_to(jawwidth)
        else:
            raise ValueError("Armname must be lft or rgt!")

    def hold(self, objcm, jawwidth=None, armname='lft'):
        """
        the objcm is added as a part of the robot to the cd checker
        :param jawwidth:
        :param objcm:
        :return:
        """
        if armname == 'lft':
            rel_pos, rel_rotmat = self.lft_tcp_jlc.get_loc_pose(objcm.get_pos(), objcm.get_rotmat())
            intolist = [self.lft_body.lnks[0],
                        self.lft_body.lnks[1],
                        self.lft_arm.lnks[1],
                        self.lft_arm.lnks[2],
                        self.lft_arm.lnks[3],
                        self.lft_arm.lnks[4],
                        self.rgt_arm.lnks[1],
                        self.rgt_arm.lnks[2],
                        self.rgt_arm.lnks[3],
                        self.rgt_arm.lnks[4],
                        self.rgt_arm.lnks[5],
                        self.rgt_arm.lnks[6],
                        self.rgt_hnd.lft.lnks[0],
                        self.rgt_hnd.lft.lnks[1],
                        self.rgt_hnd.rgt.lnks[1]]
            self.lft_oih_infos.append(self.cc.add_cdobj(objcm, rel_pos, rel_rotmat, intolist))
        elif armname == 'rgt':
            rel_pos, rel_rotmat = self.lft_tcp_jlc.get_loc_pose(objcm.get_pos(), objcm.get_rotmat())
            intolist = [self.rgt_body.lnks[0],
                        self.rgt_body.lnks[1],
                        self.rgt_arm.lnks[1],
                        self.rgt_arm.lnks[2],
                        self.rgt_arm.lnks[3],
                        self.rgt_arm.lnks[4],
                        self.lft_arm.lnks[1],
                        self.lft_arm.lnks[2],
                        self.lft_arm.lnks[3],
                        self.lft_arm.lnks[4],
                        self.lft_arm.lnks[5],
                        self.lft_arm.lnks[6],
                        self.lft_hnd.lft.lnks[0],
                        self.lft_hnd.lft.lnks[1],
                        self.lft_hnd.rgt.lnks[1]]
            self.rgt_oih_infos.append(self.cc.add_cdobj(objcm, rel_pos, rel_rotmat, intolist))
        else:
            raise ValueError("Armname must be lft or rgt!")
        if jawwidth is not None:
            self.jaw_to(jawwidth, armname=armname)

    def get_oih_list(self, armname='lft'):
        """
        oih = object in hand list
        :param armname:
        :return:
        """
        if armname == 'lft':
            oih_infos = self.lft_oih_infos
        elif armname == 'rgt':
            oih_infos = self.rgt_oih_infos
        else:
            raise ValueError("Armname must be lft or rgt!")
        return_list = []
        for obj_info in oih_infos:
            objcm = obj_info['collisionmodel']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            return_list.append(objcm)
        return return_list

    def release(self, objcm, jawwidth=None, armname='lft'):
        """
        the objcm is added as a part of the robot to the cd checker
        :param jawwidth:
        :param objcm:
        :param armname:
        :return:
        """
        if armname == 'lft':
            oih_infos = self.lft_oih_infos
        elif armname == 'rgt':
            oih_infos = self.rgt_oih_infos
        else:
            raise ValueError("Armname must be lft or rgt!")
        if jawwidth is not None:
            self.jaw_to(jawwidth, armname)
        for obj_info in oih_infos:
            if obj_info['collisionmodel'] is objcm:
                self.cc.delete_cdobj(obj_info)
                self.oih_infos.remove(obj_info)
                break

    def gen_stickmodel(self,
                       tcp_jntid=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='yumi'):
        stickmodel = mc.ModelCollection(name=name)
        self.lft_body.gen_stickmodel(tcp_loc_pos=None,
                                     tcp_loc_rotmat=None,
                                     toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        self.lft_arm.gen_stickmodel(tcp_jntid=tcp_jntid,
                                    tcp_loc_pos=tcp_loc_pos,
                                    tcp_loc_rotmat=tcp_loc_rotmat,
                                    toggle_tcpcs=toggle_tcpcs,
                                    toggle_jntscs=toggle_jntscs,
                                    toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.lft_hnd.gen_stickmodel(tcp_loc_pos=None,
                                    tcp_loc_rotmat=None,
                                    toggle_tcpcs=False,
                                    toggle_jntscs=toggle_jntscs,
                                    toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.rgt_body.gen_stickmodel(tcp_loc_pos=None,
                                     tcp_loc_rotmat=None,
                                     toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        self.rgt_arm.gen_stickmodel(tcp_jntid=tcp_jntid,
                                    tcp_loc_pos=tcp_loc_pos,
                                    tcp_loc_rotmat=tcp_loc_rotmat,
                                    toggle_tcpcs=toggle_tcpcs,
                                    toggle_jntscs=toggle_jntscs,
                                    toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.rgt_hnd.gen_stickmodel(tcp_loc_pos=None,
                                    tcp_loc_rotmat=None,
                                    toggle_tcpcs=False,
                                    toggle_jntscs=toggle_jntscs,
                                    toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self,
                      tcp_jntid=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='xarm_gripper_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        self.lft_body.gen_meshmodel(tcp_loc_pos=None,
                                    tcp_loc_rotmat=None,
                                    toggle_tcpcs=False,
                                    toggle_jntscs=toggle_jntscs,
                                    rgba=rgba).attach_to(meshmodel)
        self.lft_arm.gen_meshmodel(tcp_jntid=tcp_jntid,
                                   tcp_loc_pos=tcp_loc_pos,
                                   tcp_loc_rotmat=tcp_loc_rotmat,
                                   toggle_tcpcs=toggle_tcpcs,
                                   toggle_jntscs=toggle_jntscs,
                                   rgba=rgba).attach_to(meshmodel)
        self.lft_hnd.gen_meshmodel(tcp_loc_pos=None,
                                   tcp_loc_rotmat=None,
                                   toggle_tcpcs=False,
                                   toggle_jntscs=toggle_jntscs,
                                   rgba=rgba).attach_to(meshmodel)
        self.rgt_arm.gen_meshmodel(tcp_jntid=tcp_jntid,
                                   tcp_loc_pos=tcp_loc_pos,
                                   tcp_loc_rotmat=tcp_loc_rotmat,
                                   toggle_tcpcs=toggle_tcpcs,
                                   toggle_jntscs=toggle_jntscs,
                                   rgba=rgba).attach_to(meshmodel)
        self.rgt_hnd.gen_meshmodel(tcp_loc_pos=None,
                                   tcp_loc_rotmat=None,
                                   toggle_tcpcs=False,
                                   toggle_jntscs=toggle_jntscs,
                                   rgba=rgba).attach_to(meshmodel)
        return meshmodel


if __name__ == '__main__':
    import time
    import visualization.panda.world as wd
    import modeling.geometricmodel as gm

    base = wd.World(campos=[1.5, 0, 3], lookatpos=[0, 0, .5])
    gm.gen_frame().attach_to(base)
    yumi_instance = Yumi(enable_cc=True)
    yumi_meshmodel = yumi_instance.gen_meshmodel()
    yumi_meshmodel.attach_to(base)
    yumi_instance.show_cdprimit()
    yumi_instance.gen_stickmodel().attach_to(base)
    tic = time.time()
    result = yumi_instance.is_collided()
    toc = time.time()
    print(result, toc - tic)
    base.run()
