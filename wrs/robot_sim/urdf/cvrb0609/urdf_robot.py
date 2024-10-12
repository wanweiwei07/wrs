import wrs.robot_sim._kinematics.jl as rkjl
import numpy as np
import networkx as nx
import wrs.modeling.model_collection as mmc
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
import wrs.basis.robot_math as rm
import wrs.robot_sim._kinematics.jlchain as rkjlc
from wrs.robot_sim.urdf.urdf_parser import URDF
from collections import OrderedDict


class CVRB0609(object):

    def __init__(self, urdf_file="cvrb0609.urdf"):
        self.urdf_file = urdf_file
        self._urdf = URDF.load(urdf_file)
        self._jlg_segments = self._urdf.segment(toggle_debug=True)
        self.components = {}
        self.anchors = {}
        self.jlg_map = OrderedDict()
        for jlg in self._jlg_segments:
            if jlg.number_of_edges() == 1:
                # anchor
                sorted_nodes = list(nx.topological_sort(jlg))
                anchor = rkjl.Anchor(pos=jlg[sorted_nodes[0]][sorted_nodes[1]]['joint'].origin[:3, 3],
                                     rotmat=jlg[sorted_nodes[0]][sorted_nodes[1]]['joint'].origin[:3, :3])
                # TODO anchor cmodel
                self.anchors[jlg.graph['name']] = anchor
                self.jlg_map[jlg] = anchor
            else:
                sorted_nodes = list(nx.topological_sort(jlg))
                sorted_edges = list(zip(sorted_nodes[:-1], sorted_nodes[1:]))
                start_fixed = jlg[sorted_edges[0][0]][sorted_edges[0][1]]['joint']  # fixed
                end_fixed = jlg[sorted_edges[-1][0]][sorted_edges[-1][1]]['joint']  # fixed
                jlc = rkjlc.JLChain(pos=start_fixed.origin[:3, 3], rotmat=start_fixed.origin[:3, :3],
                                    name=jlg.graph['name'], n_dof=jlg.number_of_edges() - 2)
                # jlc.anchor.loc_flange_pose_list = [[start_fixed.origin[:3, 3], start_fixed.origin[:3, :3]]]
                for i, jnt in enumerate(jlc.jnts):
                    urdf_jnt = jlg[sorted_edges[i + 1][0]][sorted_edges[i + 1][1]]['joint']
                    if urdf_jnt.joint_type == 'prismatic':
                        jnt.change_type(rkjlc.const.JntType.PRISMATIC)
                    jnt.loc_pos = urdf_jnt.origin[:3, 3]
                    jnt.loc_motion_ax = urdf_jnt.axis
                    jnt.motion_range = np.asarray([urdf_jnt.limit.lower, urdf_jnt.limit.upper])
                    jnt.lnk.cmodel = mcm.CollisionModel(sorted_edges[i + 1][1].collision_mesh)
                    jnt.lnk.cmodel.rgba = rm.const.cool_map(i / jlc.n_dof)
                jlc.set_flange(loc_flange_pos=end_fixed.origin[:3, 3], loc_flange_rotmat=end_fixed.origin[:3, :3])
                jlc.finalize(iksolver=None)
                values = self.components.setdefault(jlg.graph['name'], [])
                values.append(jlc)
                self.jlg_map[jlg] = jlc
        self.jlg_map=OrderedDict(sorted(self.jlg_map.items(), key=lambda x: x[0].graph['priority']))
        self.update_fixed()

    def update_fixed(self):
        for pair in self.jlg_map.items():
            if pair[0].graph['previous'] is None:
                pair[1].fix_to(pos=np.zeros(3), rotmat=np.eye(3))
            else:
                pair[1].fix_to(pos=self.jlg_map[pair[0].graph['previous']].gl_flange_pose[0],
                               rotmat=self.jlg_map[pair[0].graph['previous']].gl_flange_pose[1])

    def gen_stickmodel(self):
        m_col = mmc.ModelCollection()
        for jlc_list in self.components.values():
            for jlc in jlc_list:
                jlc.gen_stickmodel().attach_to(m_col)
        return m_col

    def gen_meshmodel(self):
        m_col = mmc.ModelCollection()
        for jlc_list in self.components.values():
            for jlc in jlc_list:
                jlc.gen_meshmodel().attach_to(m_col)
        return m_col


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[5, 0, 3], lookat_pos=[0, 0, 1.5])
    mgm.gen_frame().attach_to(base)
    robot = CVRB0609(urdf_file="cvrb0609.urdf")
    robot.gen_stickmodel().attach_to(base)
    robot.gen_meshmodel().attach_to(base)
    base.run()
