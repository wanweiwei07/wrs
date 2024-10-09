import wrs.grasping.reasoner as gr


class PG(object):
    """
    A container class that holds the object pose and reference grasps id
    """

    def __init__(self,
                 obj_pose=None,
                 feasible_gids=None,
                 feasible_grasps=None,
                 feasible_jv_list=None):
        self._obj_pose = obj_pose
        self._feasible_gids = feasible_gids
        self._feasible_grasps = feasible_grasps
        self._feasible_jv_list = feasible_jv_list

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
    def feasible_jv_list(self):
        return self._feasible_jv_list

    @staticmethod
    def create_from_pose(robot, reference_grasp_collection, obj_pose, obstacle_list=None, consider_robot=True):
        pos = obj_pose[0]
        rotmat = obj_pose[1]
        grasp_reasoner = gr.GraspReasoner(robot)
        feasible_gids, feasible_grasps, feasible_jv_list = grasp_reasoner.find_feasible_gids(
            reference_grasp_collection=reference_grasp_collection,
            obstacle_list=obstacle_list,
            goal_pose=(pos, rotmat),
            consider_robot=consider_robot,
            toggle_keep=True,
            toggle_dbg=False)
        if feasible_gids is None:
            print("No feasible grasps found.")
            return None
        return PG(obj_pose=obj_pose,
                  feasible_gids=feasible_gids,
                  feasible_grasps=feasible_grasps,
                  feasible_jv_list=feasible_jv_list)

    def __str__(self):
        return f"pose= {repr(self._obj_pose)}, feasible gids= {repr(self._feasible_gids)}, feasible_jv_list= {repr(self._feasible_jv_list)}"
