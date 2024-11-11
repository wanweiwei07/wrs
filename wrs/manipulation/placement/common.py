import wrs.grasping.reasoner as gr


class GPG(object):
    """
    A container class that holds the object pose and reference grasps id
    """

    def __init__(self,
                 obj_pose=None,
                 feasible_gids=None,
                 feasible_grasps=None,
                 feasible_confs=None):
        self._obj_pose = obj_pose
        self._feasible_gids = feasible_gids
        self._feasible_grasps = feasible_grasps
        self._feasible_confs = feasible_confs

    @property
    def obj_pose(self):
        return self._obj_pose

    @property
    def feasible_gids(self):
        return self._feasible_gids

    @property
    def feasible_grasps(self):
        return self._feasible_grasps

    @property
    def feasible_confs(self):
        return self._feasible_confs

    @staticmethod
    def create_from_pose(robot, reference_gc, obj_pose, obstacle_list=None,
                         consider_robot=True, toggle_dbg=False):
        pos = obj_pose[0]
        rotmat = obj_pose[1]
        grasp_reasoner = gr.GraspReasoner(robot, reference_gc)
        feasible_gids, feasible_grasps, feasible_confs = grasp_reasoner.find_feasible_gids(
            obstacle_list=obstacle_list,
            goal_pose=(pos, rotmat),
            consider_robot=consider_robot,
            toggle_dbg=toggle_dbg)
        if feasible_gids is None:
            print("No feasible grasps found.")
            return None
        return GPG(obj_pose=obj_pose,
                   feasible_gids=feasible_gids,
                   feasible_grasps=feasible_grasps,
                   feasible_confs=feasible_confs)

    def __str__(self):
        return (f"pose= {repr(self._obj_pose)}, feasible gids= {repr(self._feasible_gids)}, "
                f"feasible confs= {repr(self._feasible_confs)}")
