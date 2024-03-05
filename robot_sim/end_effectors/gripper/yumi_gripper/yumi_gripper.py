import os
import math
import numpy as np
import modeling.model_collection as mmc
import modeling.collision_model as mcm
import robot_sim._kinematics.jlchain as rkjl
import basis.robot_math as rm
import robot_sim.end_effectors.gripper.gripper_interface as gpi
import robot_sim._kinematics.model_generator as rkmg


class YumiGripper(gpi.GripperInterface):

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 cdmesh_type=mcm.mc.CDMType.DEFAULT,
                 name='yumi_gripper'):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        current_file_dir = os.path.dirname(__file__)
        # jaw range
        self.jaw_range = np.array([.0, .05])
        # coupling
        self.coupling.finalize()
        # jlc
        self.jlc = rkjl.JLChain(pos=self.coupling.gl_flange_pos, rotmat=self.coupling.gl_flange_pos, n_dof=2, name=name)
        # anchor
        self.jlc.anchor.lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "base.stl"),
                                                        cdmesh_type=self.cdmesh_type)
        self.jlc.anchor.lnk.cmodel.rgba = np.array([.75, .75, .75, 1])
        # the 1st joint (left finger)
        self.jlc.jnts[0].change_type(rkjl.rkc.JntType.PRISMATIC, np.array([0, self.jaw_range[1] / 2]))
        self.jlc.jnts[0].loc_pos = np.array([-0.0065, 0, 0.0837])
        self.jlc.jnts[0].loc_motion_ax = rm.bc.y_ax
        self.jlc.jnts[0].motion_range = np.array([0.0, 0.025])
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "finger_sheet_metal.stl"),
            cdmesh_type=self.cdmesh_type)
        self.jlc.jnts[0].lnk.cmodel.rgba = np.array([.5, .5, .5, 1])
        # the 2nd joint (right finger)
        self.jlc.jnts[1].change_type(rkjl.rkc.JntType.PRISMATIC, np.array([0, -self.jaw_range[1]]))
        self.jlc.jnts[1].loc_pos = np.array([0.013, 0, 0])
        self.jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(0, 0, np.pi)
        self.jlc.jnts[1].loc_motion_ax = rm.bc.y_ax
        self.jlc.jnts[1].motion_range = np.array([self.jaw_range[1], 0.0])
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "finger_sheet_metal.stl"),
            cdmesh_type=self.cdmesh_type)
        self.jlc.jnts[1].lnk.cmodel.rgba = np.array([.5, .5, .5, 1])
        # reinitialize
        self.jlc.finalize()
        # acting center
        self.loc_acting_center_pos = np.array([0, 0, 0.13])
        # collisions
        self.cdmesh_elements = (self.jlc.anchor.lnk,
                                self.jlc.jnts[0].lnk,
                                self.jlc.jnts[1].lnk)

    def fix_to(self, pos, rotmat, jaw_width=None):
        self.pos = pos
        self.rotmat = rotmat
        if jaw_width is not None:
            side_jawwidth = jaw_width / 2.0
            if 0 <= side_jawwidth <= self.jaw_range[1] / 2:
                self.jlc.jnts[0].motion_value = side_jawwidth
                self.jlc.jnts[1].motion_value = -jaw_width
            else:
                raise ValueError("The angle parameter is out of range!")
        self.coupling.fix_to(self.pos, self.rotmat)
        self.jlc.fix_to(self.coupling.gl_flange_pos, self.coupling.gl_flange_rotmat)
        self.update_oiee()

    def change_jaw_width(self, jaw_width):
        super().change_jaw_width(jaw_width=jaw_width)  # assert oiee
        side_jawwidth = jaw_width / 2.0
        if 0 <= side_jawwidth <= self.jaw_range[1] / 2:
            self.jlc.go_given_conf(jnt_values=[side_jawwidth, jaw_width])
        else:
            raise ValueError("The angle parameter is out of range!")

    def get_jaw_width(self):
        return -self.jlc.jnts[1].motion_value

    def gen_stickmodel(self, toggle_tcp_frame=False, toggle_jnt_frames=False, name='yumi_gripper_stickmodel'):
        m_col = mmc.ModelCollection(name=name)
        rkmg.gen_jlc_stick(self.coupling, toggle_jnt_frames=False, toggle_flange_frame=False).attach_to(m_col)
        rkmg.gen_jlc_stick(self.jlc, toggle_jnt_frames=toggle_jnt_frames, toggle_flange_frame=False).attach_to(m_col)
        if toggle_tcp_frame:
            self._toggle_tcp_frame(m_col)
        return m_col

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False,
                      name='yumi_gripper_meshmodel'):
        m_col = mmc.ModelCollection(name=name)
        rkmg.gen_jlc_mesh(self.coupling,
                          rgb=rgb,
                          alpha=alpha,
                          toggle_flange_frame=False,
                          toggle_jnt_frames=False,
                          toggle_cdmesh=toggle_cdmesh,
                          toggle_cdprim=toggle_cdprim).attach_to(m_col)
        rkmg.gen_jlc_mesh(self.jlc,
                          rgb=rgb,
                          alpha=alpha,
                          toggle_flange_frame=False,
                          toggle_jnt_frames=toggle_jnt_frames,
                          toggle_cdmesh=toggle_cdmesh,
                          toggle_cdprim=toggle_cdprim).attach_to(m_col)
        if toggle_tcp_frame:
            self._toggle_tcp_frame(m_col)
        # oiee
        self.gen_oiee_meshmodel(m_col, rgb=rgb, alpha=alpha, toggle_cdprim=toggle_cdprim,
                                toggle_cdmesh=toggle_cdmesh, toggle_frame=toggle_jnt_frames)
        return m_col


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    grpr = YumiGripper()
    # 1
    grpr.fix_to(pos=np.array([0, .3, .2]), rotmat=rm.rotmat_from_euler(math.pi / 3, math.pi / 3, math.pi / 3))
    grpr.change_jaw_width(.02)
    print(grpr.get_jaw_width())
    grpr.gen_stickmodel().attach_to(base)
    grpr.gen_meshmodel(alpha=.3).attach_to(base)
    # 2
    grpr.fix_to(pos=np.zeros(3), rotmat=np.eye(3))
    grpr.gen_meshmodel().attach_to(base)
    # 3
    grpr.fix_to(pos=np.array([.3, .3, .2]), rotmat=rm.rotmat_from_axangle([0, 1, 0], .01))
    model = grpr.gen_meshmodel(rgb=np.array([0.5, .5, 0]), alpha=.5, toggle_cdmesh=True, toggle_cdprim=True)
    model.attach_to(base)
    base.run()
