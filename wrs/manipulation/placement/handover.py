import pickle
import wrs.basis.robot_math as rm
import wrs.modeling.model_collection as mmc
import wrs.grasping.reasoner as gr


class HOPG(object):

    def __init__(self,
                 obj_pose,
                 sender_gid,
                 sender_grasp,
                 sender_conf,
                 receiver_gids,
                 receiver_grasps,
                 receiver_confs):
        self._obj_pose = obj_pose
        self._sender_gid = sender_gid
        self._sender_grasp = sender_grasp
        self._sender_conf = sender_conf
        self._receiver_gids = receiver_gids
        self._receiver_grasps = receiver_grasps
        self._receiver_confs = receiver_confs

    @property
    def obj_pose(self):
        return self._obj_pose

    @property
    def sender_gid(self):
        return self._sender_gid

    @property
    def sender_grasp(self):
        return self._sender_grasp

    @property
    def sender_conf(self):
        return self._sender_conf

    @property
    def receiver_gids(self):
        return self._receiver_gids

    @property
    def receiver_grasps(self):
        return self._receiver_grasps

    @property
    def receiver_confs(self):
        return self._receiver_confs

    def __str(self):
        return (f"sender_gid= {repr(self._sender_gid)}, "
                f"sender_conf= {repr(self._sender_conf)}, "
                f"pose= {repr(self._obj_pose)}, "
                f"receiver gids= {repr(self._receiver_gids)}, "
                f"receiver grasps= {repr(self._receiver_grasps)}, "
                f"receiver confs= {repr(self._receiver_confs)}")


class HOPGCollection(object):
    def __init__(self, obj_cmodel=None, sender_robot=None, receiver_robot=None,
                 sender_reference_gc=None, receiver_reference_gc=None):
        """
        :param robot:
        :param reference_fsp_poses: an instance of ReferenceFSPPoses
        :param reference_gc: an instance of GraspCollection
        """
        self.obj_cmodel = obj_cmodel
        self.sender_robot = sender_robot
        self.receiver_robot = receiver_robot
        self.sender_reasoner = gr.GraspReasoner(sender_robot, sender_reference_gc)
        self.receiver_reasoner = gr.GraspReasoner(receiver_robot, receiver_reference_gc)
        self._hopg_list = []

    @property
    def sender_reference_gc(self):
        return self.sender_reasoner.reference_gc

    @property
    def receiver_reference_gc(self):
        return self.receiver_reasoner.reference_gc

    def load_from_disk(self, file_name="hopg_collection.pickle"):
        with open(file_name, 'rb') as file:
            self._hopg_list = pickle.load(file)
        return self

    def save_to_disk(self, file_name='hopg_collection.pickle'):
        with open(file_name, 'wb') as file:
            pickle.dump(self._hopg_list, file)

    def __getitem__(self, index):
        return self._hopg_list[index]

    def __len__(self):
        return len(self._hopg_list)

    def __iter__(self):
        return iter(self._hopg_list)

    def __add__(self, other):
        self._hopg_list += other._hopg_list
        return self

    def add_new_hop(self, pos, rotmat, obstacle_list=None, consider_robot=True, toggle_dbg=False):
        sender_gids, sender_grasps, sender_confs = self.sender_reasoner.find_feasible_gids(
            goal_pose=(pos, rotmat),
            obstacle_list=obstacle_list,
            consider_robot=consider_robot,
            toggle_dbg=False)
        receiver_gids, receiver_grasps, receiver_confs = self.receiver_reasoner.find_feasible_gids(
            goal_pose=(pos, rotmat),
            obstacle_list=obstacle_list,
            consider_robot=consider_robot,
            toggle_dbg=False)
        # ee mesh collisions
        sid2rid_dict = {}
        for sid, sender_gid in enumerate(sender_gids):
            sid2rid_dict[sid] = []
            self.sender_robot.end_effector.grip_at_by_pose(jaw_center_pos=sender_grasps[sid].ac_pos,
                                                           jaw_center_rotmat=sender_grasps[sid].ac_rotmat,
                                                           jaw_width=sender_grasps[sid].ee_values)
            for rid, receiver_gid in enumerate(receiver_gids):
                self.receiver_robot.end_effector.grip_at_by_pose(
                    jaw_center_pos=receiver_grasps[rid].ac_pos,
                    jaw_center_rotmat=receiver_grasps[rid].ac_rotmat,
                    jaw_width=receiver_grasps[rid].ee_values)
                if self.sender_robot.end_effector.is_mesh_collided(
                        cmodel_list=self.receiver_robot.end_effector.cdmesh_list):
                    continue
                else:
                    sid2rid_dict[sid].append(rid)
        # robot prim collisions
        self.sender_robot.toggle_off_eecd()
        for sid, rid_list in sid2rid_dict.items():
            feasible_receiver_gids = []
            feasible_receiver_grasps = []
            feasible_receiver_confs = []
            if consider_robot:
                self.sender_robot.goto_given_conf(jnt_values=sender_confs[sid], ee_values=sender_grasps[sid].ee_values)
                for rid in rid_list:
                    # print(sid, rid, rid_list)
                    self.receiver_robot.goto_given_conf(jnt_values=receiver_confs[rid],
                                                        ee_values=receiver_grasps[rid].ee_values)
                    if self.sender_robot.is_collided(obstacle_list=None, other_robot_list=[self.receiver_robot],
                                                     toggle_dbg=False):
                        # self.obj_cmodel.pose=(pos, rotmat)
                        # self.obj_cmodel.attach_to(base)
                        # self.sender_robot.gen_meshmodel(rgb=rm.const.yellow, alpha=.7).attach_to(base)
                        # self.receiver_robot.gen_meshmodel(rgb=rm.const.green, alpha=.7).attach_to(base)
                        # base.run()
                        continue
                    else:
                        # print(rid)
                        # self.obj_cmodel.pose=(pos, rotmat)
                        # self.obj_cmodel.attach_to(base)
                        # self.sender_robot.gen_meshmodel(rgb=rm.const.yellow, alpha=.7).attach_to(base)
                        # self.receiver_robot.gen_meshmodel(rgb=rm.const.green, alpha=.7).attach_to(base)
                        feasible_receiver_gids.append(receiver_gids[rid])
                        feasible_receiver_grasps.append(receiver_grasps[rid])
                        feasible_receiver_confs.append(receiver_confs[rid])
            else:
                for rid in rid_list:
                    feasible_receiver_gids.append(receiver_gids[rid])
                    feasible_receiver_grasps.append(receiver_grasps[rid])
                    feasible_receiver_confs.append(receiver_confs[rid])
            if len(feasible_receiver_gids) > 0:
                self._hopg_list.append(HOPG(sender_gid=sender_gids[sid],
                                            sender_grasp=sender_grasps[sid],
                                            sender_conf=sender_confs[sid],
                                            obj_pose=(pos, rotmat),
                                            receiver_gids=feasible_receiver_gids,
                                            receiver_grasps=feasible_receiver_grasps,
                                            receiver_confs=feasible_receiver_confs))
        # base.run()
        self.sender_robot.toggle_on_eecd()

    def copy(self):
        return HOPGCollection(obj_cmodel=self.obj_cmodel,
                              sender_robot=self.sender_robot,
                              receiver_robot=self.receiver_robot,
                              sender_reference_gc=self.sender_reference_gc,
                              receiver_reference_gc=self.receiver_reference_gc)

    def gen_meshmodel(self):
        meshmodel_list = []
        for hopg in self._hopg_list:
            m_col = mmc.ModelCollection()
            obj_cmodel_copy = self.obj_cmodel.copy()
            obj_cmodel_copy.pose = hopg.obj_pose
            obj_cmodel_copy.attach_to(m_col)
            sender_grasp = hopg.sender_grasp
            sender_conf = hopg.sender_conf
            self.sender_robot.goto_given_conf(jnt_values=sender_conf, ee_values=sender_grasp.ee_values)
            self.sender_robot.gen_meshmodel(rgb=rm.const.red, alpha=.7).attach_to(m_col)
            for grasp, conf in zip(hopg.receiver_grasps, hopg.receiver_confs):
                self.receiver_robot.goto_given_conf(jnt_values=conf, ee_values=grasp.ee_values)
                self.receiver_robot.gen_meshmodel(rgb=rm.const.blue, alpha=.7).attach_to(m_col)
            meshmodel_list.append(m_col)
        return meshmodel_list
