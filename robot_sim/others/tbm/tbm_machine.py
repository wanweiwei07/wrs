import os
import math
import numpy as np
import basis.robot_math as rm
import modeling.model_collection as mc
import robot_sim._kinematics.jlchain as jl
import robot_sim.robots.robot_interface as ri


class TBM(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="tbm"):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        # agv
        self.front = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=np.zeros(1), name='head')
        self.front.jnts[1]['loc_pos'] = np.zeros(3)
        self.front.jnts[1]['type'] = 'prismatic'
        self.front.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        self.front.lnks[0]['name'] = 'tbm_front_shield'
        self.front.lnks[0]['loc_pos'] = np.array([0, 0, 0])
        self.front.lnks[0]['meshfile'] = os.path.join(this_dir, 'meshes', 'tbm_front_shield.stl')
        self.front.lnks[0]['rgba'] = [.7, .7, .7, 1.0]
        self.front.lnks[1]['name'] = 'tbm_cutter_head'
        self.front.lnks[1]['meshfile'] = os.path.join(this_dir, 'meshes', 'tbm_cutter_head.stl')
        self.front.lnks[1]['rgba'] = [.35, .35, .35, 1.0]
        self.front.lnks[1]['loc_pos'] = np.array([0, 0, 0])
        self.front.tgtjnts = [1]

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.front.fix_to(self.pos, self.rotmat)

    def fk(self, jnt_values=np.zeros(1)):
        """
        :param jnt_values: 7 or 3+7, 3=agv, 7=arm, 1=grpr; metrics: meter-radian
        :param component_name: 'arm', 'agv', or 'all'
        :return:
        author: weiwei
        date: 20201208toyonaka
        """
        self.front.fk(jnt_values=jnt_values)

    def get_jnt_values(self):
        return self.front.get_jnt_values()

    def gen_stickmodel(self,
                       tcp_jntid=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='xarm7_shuidi_mobile_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.agv.gen_stickmodel(tcp_loc_pos=None,
                                tcp_loc_rotmat=None,
                                toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        self.arm.gen_stickmodel(tcp_jntid=tcp_jntid,
                                tcp_loc_pos=tcp_loc_pos,
                                tcp_loc_rotmat=tcp_loc_rotmat,
                                toggle_tcpcs=toggle_tcpcs,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.hnd.gen_stickmodel(toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self,
                      tcp_jntid=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='xarm_shuidi_mobile_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        self.agv.gen_meshmodel(tcp_loc_pos=None,
                               tcp_loc_rotmat=None,
                               toggle_tcpcs=False,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(meshmodel)
        self.arm.gen_meshmodel(tcp_jntid=tcp_jntid,
                               tcp_loc_pos=tcp_loc_pos,
                               tcp_loc_rotmat=tcp_loc_rotmat,
                               toggle_tcpcs=toggle_tcpcs,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(meshmodel)
        self.hnd.gen_meshmodel(tcp_loc_pos=None,
                               tcp_loc_rotmat=None,
                               toggle_tcpcs=False,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(meshmodel)
        for obj_info in self.oih_infos:
            objcm = obj_info['collisionmodel']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            objcm.copy().attach_to(meshmodel)
        return meshmodel


if __name__ == '__main__':
    import time
    import basis.robot_math as rm
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[1.5, 0, 3], lookat_pos=[0, 0, .5])

    gm.gen_frame().attach_to(base)
    xav = XArm7YunjiMobile(enable_cc=True)
    xav.fk(component_name='all', jnt_values=np.array([0, 0, 0, 0, 0, 0, math.pi, 0, -math.pi / 6, 0, 0]))
    xav.jaw_to(.08)
    tgt_pos = np.array([.85, 0, .55])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    jnt_values = xav.ik(component_name='arm', tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
    tgt_pos2 = np.array([.45, 0, .07])
    tgt_rotmat2 = rm.rotmat_from_euler(0,math.pi,0)
    jnt_values2 = xav.ik(component_name='arm', tgt_pos=tgt_pos2, tgt_rotmat=tgt_rotmat2, seed_jnt_values=jnt_values, max_niter=10000)
    print(jnt_values)
    xav.fk(component_name='arm', jnt_values=jnt_values2)
    xav.fk(component_name='agv', jnt_values=np.array([.2, -.5, math.radians(30)]))
    xav_meshmodel = xav.gen_meshmodel(toggle_tcpcs=True)
    xav_meshmodel.attach_to(base)
    xav.show_cdprimit()
    xav.gen_stickmodel().attach_to(base)
    tic = time.time()
    result = xav.is_collided()
    toc = time.time()
    print(result, toc - tic)

    # xav_cpy = xav.copy()
    # xav_cpy.move_to(pos=np.array([.5,.5,0]),rotmat=rm.rotmat_from_axangle([0,0,1],-math.pi/3))
    # xav_meshmodel = xav_cpy.gen_meshmodel()
    # xav_meshmodel.attach_to(base)
    # xav_cpy.show_cdprimit()
    # tic = time.time()
    # result = xav_cpy.is_collided(otherrobot_list=[xav])
    # toc = time.time()
    # print(result, toc - tic)
    base.run()
