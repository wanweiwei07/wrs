import os
import math
import numpy as np
import basis.robotmath as rm
import robotsim._kinematics.jlchain as jl
import robotsim._kinematics.collisionchecker as cc


class IRB14050(jl.JLChain):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), homeconf=np.zeros(7), name='irb14050'):
        super().__init__(pos=pos, rotmat=rotmat, homeconf=homeconf, name=name)
        this_dir, this_filename = os.path.split(__file__)
        # seven joints, njnts = 7+2 (tgt ranges from 1-7), nlinks = 7+1
        jnt_saferngmargin = math.pi/18.0
        # self.jnts[1]['loc_pos'] = np.array([0.05355, -0.0725, 0.41492])
        # self.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(-0.9795, -0.5682, -2.3155)
        self.jnts[1]['loc_pos'] = np.array([0., 0., 0.])
        self.jnts[1]['rngmin'] = -2.94087978961+jnt_saferngmargin
        self.jnts[1]['rngmax'] = 2.94087978961-jnt_saferngmargin
        self.jnts[2]['loc_pos'] = np.array([0.03, 0.0, 0.1])
        self.jnts[2]['loc_rotmat'] = rm.rotmat_from_euler(1.57079632679, 0.0, 0.0)
        self.jnts[2]['rngmin'] = -2.50454747661+jnt_saferngmargin
        self.jnts[2]['rngmax'] = 0.759218224618-jnt_saferngmargin
        self.jnts[3]['loc_pos'] = np.array([-0.03, 0.17283, 0.0])
        self.jnts[3]['loc_rotmat'] = rm.rotmat_from_euler(-1.57079632679, 0.0, 0.0)
        self.jnts[3]['rngmin'] = -2.94087978961+jnt_saferngmargin
        self.jnts[3]['rngmax'] = 2.94087978961-jnt_saferngmargin
        self.jnts[4]['loc_pos'] = np.array([-0.04188, 0.0, 0.07873])
        self.jnts[4]['loc_rotmat'] = rm.rotmat_from_euler(1.57079632679, -1.57079632679, 0.0)
        self.jnts[4]['rngmin'] = -2.15548162621+jnt_saferngmargin
        self.jnts[4]['rngmax'] = 1.3962634016-jnt_saferngmargin
        self.jnts[5]['loc_pos'] = np.array([0.0405, 0.16461, 0.0])
        self.jnts[5]['loc_rotmat'] = rm.rotmat_from_euler(-1.57079632679, 0.0, 0.0)
        self.jnts[5]['rngmin'] = -5.06145483078+jnt_saferngmargin
        self.jnts[5]['rngmax'] = 5.06145483078-jnt_saferngmargin
        self.jnts[6]['loc_pos'] = np.array([-0.027, 0, 0.10039])
        self.jnts[6]['loc_rotmat'] = rm.rotmat_from_euler(1.57079632679, 0.0, 0.0)
        self.jnts[6]['rngmin'] = -1.53588974176+jnt_saferngmargin
        self.jnts[6]['rngmax'] = 2.40855436775-jnt_saferngmargin
        self.jnts[7]['loc_pos'] = np.array([0.027, 0.029, 0.0])
        self.jnts[7]['loc_rotmat'] = rm.rotmat_from_euler(-1.57079632679, 0.0, 0.0)
        self.jnts[7]['rngmin'] = -3.99680398707+jnt_saferngmargin
        self.jnts[7]['rngmax'] = 3.99680398707-jnt_saferngmargin
        # links
        self.lnks[1]['name'] = "link_1"
        self.lnks[1]['meshfile'] = os.path.join(this_dir, "meshes", "link_1.stl")
        self.lnks[2]['name'] = "link_2"
        self.lnks[2]['meshfile'] = os.path.join(this_dir, "meshes", "link_2.stl")
        self.lnks[2]['rgba'] = [.5,.5,.5,1.0]
        self.lnks[3]['name'] = "link_3"
        self.lnks[3]['meshfile'] = os.path.join(this_dir, "meshes", "link_3.stl")
        self.lnks[4]['name'] = "link_4"
        self.lnks[4]['meshfile'] = os.path.join(this_dir, "meshes", "link_4.stl")
        self.lnks[4]['rgba'] = [.3,.5,.3,1.0]
        self.lnks[5]['name'] = "link_5"
        self.lnks[5]['meshfile'] = os.path.join(this_dir, "meshes", "link_5.stl")
        self.lnks[6]['name'] = "link_6"
        self.lnks[6]['meshfile'] = os.path.join(this_dir, "meshes", "link_6.stl")
        self.lnks[6]['rgba'] = [.5,.5,.5,1.0]
        self.lnks[7]['name'] = "link_7"
        # self.lnks[7]['meshfile'] = os.path.join(this_dir, "meshes", "link_7.stl") # not really needed to visualize
        # reinitialization
        self.reinitialize()
        # collision detection
        self._mt.add_cdpair([1], [6])


if __name__ == '__main__':
    import time
    import visualization.panda.world as wd
    import modeling.geometricmodel as gm

    base = wd.World(campos=[1, 0, 1], lookatpos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    manipulator_instance = IRB14050()
    manipulator_instance.fk(jnt_values=[0,0,manipulator_instance.jnts[3]['rngmax']/2, manipulator_instance.jnts[4]['rngmax'], 0, 0, 0])
    manipulator_meshmodel = manipulator_instance.gen_meshmodel()
    manipulator_meshmodel.attach_to(base)
    manipulator_instance.gen_stickmodel().attach_to(base)
    manipulator_meshmodel.show_cdprimit()
    tic = time.time()
    print(manipulator_instance.is_selfcollided())
    toc = time.time()
    print(toc - tic)
    base.run()
