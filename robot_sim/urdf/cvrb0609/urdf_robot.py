import robot_sim._kinematics.jlchain as rkjlc
import numpy as np
import networkx as nx
import modeling.collision_model as mcm
import modeling.model_collection as mmc
from robot_sim.urdf.urdf_parser import URDF
import basis.robot_math as rm


class CVRB0609(object):

    def __init__(self, urdf_file="cvrb0609.urdf"):
        self._urdf_file = urdf_file
        self._urdf = URDF.load(urdf_file)
        self._jlg_segments = self._urdf.segment(toggle_debug=True)
        self.components = {}
        for jlg in self._jlg_segments:
            if jlg.number_of_edges() == 1:
                # anchor
                pass
            else:
                sorted_nodes = list(nx.topological_sort(jlg))
                sorted_edges = list(zip(sorted_nodes[:-1], sorted_nodes[1:]))
                start_fixed = jlg[sorted_edges[0][0]][sorted_edges[0][1]]['joint'] # fixed
                end_fixed = jlg[sorted_edges[-1][0]][sorted_edges[-1][1]]['joint'] # fixed
                jlc = rkjlc.JLChain(pos=start_fixed.origin[:3, 3], rotmat=start_fixed.origin[:3, :3],
                                    name=jlg.graph['name'], n_dof=jlg.number_of_edges()-2)
                jlc.anchor.loc_flange_pose_list = [[start_fixed.origin[:3,3], start_fixed.origin[:3,:3]]]
                for i, jnt in enumerate(jlc.jnts):
                    urdf_jnt = jlg[sorted_edges[i+1][0]][sorted_edges[i+1][1]]['joint']
                    if urdf_jnt.joint_type == 'prismatic':
                        jnt.change_type(rkjlc.rkc.JntType.PRISMATIC)
                    jnt.loc_pos = urdf_jnt.origin[:3, 3]
                    jnt.loc_motion_ax = urdf_jnt.axis
                    print(urdf_jnt.joint_type)
                    jnt.motion_range = np.asarray([urdf_jnt.limit.lower, urdf_jnt.limit.upper])
                    jnt.lnk.cmodel = mcm.CollisionModel(sorted_edges[i+1][1].collision_mesh)
                    jnt.lnk.cmodel.rgba = rm.bc.cool_map(1-i / jlc.n_dof)
                jlc.set_flange(loc_flange_pos=end_fixed.origin[:3, 3], loc_flange_rotmat=end_fixed.origin[:3, :3])
                jlc.finalize(iksolver=None)
                self.components[jlg.graph['name']] = jlc

    def gen_stickmodel(self):
        m_col = mmc.ModelCollection()
        for jlc in self.components.values():
            jlc.gen_stickmodel().attach_to(m_col)
        return m_col

    def gen_meshmodel(self):
        m_col = mmc.ModelCollection()
        for jlc in self.components.values():
            jlc.gen_meshmodel().attach_to(m_col)
        return m_col


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometric_model as mgm

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, .3])
    mgm.gen_frame().attach_to(base)
    robot = CVRB0609(urdf_file="cvrb0609.urdf")
    robot.gen_stickmodel().attach_to(base)
    robot.gen_meshmodel().attach_to(base)
    base.run()

