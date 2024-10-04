import time
import warnings

import numpy as np
from wrs import basis as rm, robot_sim as xyb
import networkx as nx
import matplotlib.pyplot as plt
import uuid
import wrs.motion.probabilistic.rrt as rrt
from operator import itemgetter


class RRTStar(rrt.RRT):

    def __init__(self, robot, nearby_ratio=2):
        """
        :param robot:
        :param nearby_ratio: the threshold_hold = ext_dist*nearby_ratio
        """
        super().__init__(robot)
        self.roadmap = nx.DiGraph()
        self.nearby_ratio = nearby_ratio

    # def _get_nearest_nid(self, roadmap, new_conf):
    #     """
    #     convert to numpy to accelerate access
    #     :param roadmap:
    #     :param new_conf:
    #     :return:
    #     author: weiwei
    #     date: 20210523
    #     """
    #     nodes_dict = dict(roadmap.nodes(data="conf"))
    #     nodes_key_list = list(nodes_dict.keys())  # use python > 3.7, or else there is no guarantee on the order
    #     nodes_value_list = list(nodes_dict.values())  # attention, correspondence is not guanranteed in python
    #     # ===============
    #     # the following code computes euclidean distances. it is decprecated and replaced using cdtree
    #     # ***** date: 20240304, correspondent: weiwei *****
    #     # conf_array = np.array(nodes_value_list)
    #     # diff_conf_array = np.linalg.norm(conf_array - new_conf, axis=1)
    #     # min_dist_nid = np.argmin(diff_conf_array)
    #     # return nodes_key_list[min_dist_nid]
    #     # ===============
    #     querry_tree = scipy.spatial.cKDTree(nodes_value_list)
    #     if roadmap.number_of_nodes() > 5:
    #         dist_values, indices = querry_tree.query(new_conf, k=5, workers=-1)
    #         nodes_cost_list = list(dict(roadmap.nodes(data="cost")).values())
    #         print(indices)
    #         print(nodes_cost_list)
    #         nearby_cost_list = itemgetter(*indices)(nodes_cost_list)
    #         indx = np.argmin(np.asarray(nearby_cost_list))
    #         return nodes_key_list[indx]
    #     else:
    #         return super()._get_nearest_nid(roadmap=roadmap, new_conf=new_conf)

    def _extend_sgl_conf(self, src_conf, end_conf, ext_dist):
        """
        :param src_conf:
        :param end_conf:
        :param ext_dist:
        :return: a single of 1xn nparray
        """
        len, vec = rm.unit_vector(end_conf - src_conf, toggle_length=True)
        return [src_conf + ext_dist * vec] if len > 1e-6 else []

    def _get_nearby_nid_with_min_cost(self, roadmap, new_conf, ext_dist):
        """
        :param roadmap:
        :param new_conf:
        :return:
        author: weiwei
        date: 20210523
        """
        nodes_conf_dict = dict(roadmap.nodes(data='conf'))
        nodes_conf_key_list = list(nodes_conf_dict.keys())
        nodes_conf_value_list = list(nodes_conf_dict.values())
        conf_array = np.array(nodes_conf_value_list)
        diff_conf_array = np.linalg.norm(conf_array - new_conf, axis=1)
        candidate_mask = diff_conf_array < ext_dist * self.nearby_ratio  # warninng: assumes no collision
        nodes_conf_key_array = np.array(nodes_conf_key_list, dtype=object)
        nearby_nid_list = list(nodes_conf_key_array[candidate_mask])
        return nearby_nid_list

    def _extend_roadmap(self,
                        roadmap,
                        conf,
                        ext_dist,
                        goal_conf,
                        obstacle_list=[],
                        other_robot_list=[],
                        animation=False):
        """
        find the nearest point between the given roadmap and the conf and then extend towards the conf
        :return:
        author: weiwei
        date: 20201228
        """
        nearest_nid = self._get_nearest_nid(roadmap, conf)
        new_conf_list = self._extend_sgl_conf(roadmap.nodes[nearest_nid]["conf"], conf, ext_dist)
        for new_conf in new_conf_list:
            if self._is_collided(new_conf, obstacle_list, other_robot_list):
                return nearest_nid
            else:
                new_nid = uuid.uuid4()
                # find nearby_nid_list
                nearby_nid_list = self._get_nearby_nid_with_min_cost(roadmap, new_conf, ext_dist)
                # costs
                nodes_cost_dict = dict(roadmap.nodes(data="cost"))
                nearby_cost_list = itemgetter(*nearby_nid_list)(nodes_cost_dict)
                if type(nearby_cost_list) == np.ndarray:
                    nearby_cost_list = [nearby_cost_list]
                nearby_min_cost_nid = nearby_nid_list[np.argmin(np.asarray(nearby_cost_list))]
                roadmap.add_node(new_nid, conf=new_conf, cost=roadmap.nodes[nearby_min_cost_nid]["cost"] + 1)
                roadmap.add_edge(nearby_min_cost_nid, new_nid)  # add new edge
                # rewire
                for nearby_nid in nearby_nid_list:
                    if nearby_nid != nearby_min_cost_nid:
                        if roadmap.nodes[new_nid]['cost'] + 1 < roadmap.nodes[nearby_nid]['cost']:
                            nearby_parent_nid = next(roadmap.predecessors(nearby_nid))
                            roadmap.remove_edge(nearby_parent_nid, nearby_nid)
                            roadmap.add_edge(new_nid, nearby_nid)
                            roadmap.nodes[nearby_nid]['cost'] = roadmap.nodes[new_nid]['cost'] + 1
                            cost_counter = 0
                            for nid in roadmap.successors(nearby_nid):
                                cost_counter += 1
                                roadmap.nodes[nid]['cost'] = roadmap.nodes[nearby_nid]['cost'] + cost_counter
                if animation:
                    self.draw_wspace([roadmap], self.start_conf, self.goal_conf,
                                     obstacle_list, [roadmap.nodes[nearest_nid]['conf'], conf],
                                     new_conf, '^c')
                # check goal
                if self._is_goal_reached(conf=roadmap.nodes[new_nid]['conf'], goal_conf=goal_conf, threshold=ext_dist):
                    roadmap.add_node('goal', conf=goal_conf)  # TODO current name -> connection
                    roadmap.add_edge(new_nid, 'goal')
                    return 'goal'
                return new_nid

    @rrt.RRT.keep_states_decorator
    def plan(self,
             start_conf,
             goal_conf,
             obstacle_list=[],
             other_robot_list=[],
             ext_dist=.2,
             rand_rate=70,
             max_n_iter=1000,
             max_time=15.0,
             smoothing_n_iter=0,
             animation=False):
        """
        :return: [path, all_sampled_confs]
        author: weiwei
        date: 20201226, 20240304
        """
        if smoothing_n_iter != 0:
            warnings.warn("I would suggest not using smoothing for RRT star...")
        self.roadmap.clear()
        self.start_conf = start_conf
        self.goal_conf = goal_conf
        # check seed_jnt_values and end_conf
        if self._is_collided(start_conf, obstacle_list, other_robot_list):
            print("The start robot configuration is in collision!")
            return None
        if self._is_collided(goal_conf, obstacle_list, other_robot_list):
            print("The goal robot configuration is in collision!")
            return None
        if self._is_goal_reached(conf=start_conf, goal_conf=goal_conf, threshold=ext_dist):
            mot_data = rrt.motu.MotionData(self.robot)
            mot_data.extend(jv_list=[start_conf, goal_conf])
            return mot_data
        self.roadmap.add_node('start', conf=start_conf, cost=0)
        tic = time.time()
        n = 0
        for _ in range(max_n_iter):
            n += 1
            toc = time.time()
            if max_time > 0.0:
                if toc - tic > max_time:
                    print("Failed to find a path in the given max_time!")
                    return None
            # Random Sampling
            rand_conf = self._sample_conf(rand_rate=rand_rate, default_conf=goal_conf)
            last_nid = self._extend_roadmap(roadmap=self.roadmap,
                                            conf=rand_conf,
                                            ext_dist=ext_dist,
                                            goal_conf=goal_conf,
                                            obstacle_list=obstacle_list,
                                            other_robot_list=other_robot_list,
                                            animation=animation)
            if last_nid == 'goal':
                path = self._path_from_roadmap()
                smoothed_path = self._smooth_path(path=path,
                                                  obstacle_list=obstacle_list,
                                                  other_robot_list=other_robot_list,
                                                  granularity=ext_dist,
                                                  n_iter=smoothing_n_iter,
                                                  animation=animation)
                mot_data = rrt.motu.MotionData(self.robot)
                if getattr(base, "toggle_mesh", True):
                    mot_data.extend(jv_list=smoothed_path)
                else:
                    mot_data.extend(jv_list=smoothed_path, mesh_list=[])
                return mot_data
        else:
            print("Failed to find a path with the given max_n_ter!")
            return None


if __name__ == '__main__':
    # ====Search Path with RRT====
    obstacle_list = [
        ((.5, .5), .3),
        ((.3, .6), .3),
        ((.3, .8), .3),
        ((.3, 1.0), .3),
        ((.7, .5), .3),
        ((.9, .5), .3),
        ((1.0, .5), .3)
    ]  # [x,y,size]
    # Set Initial parameters
    robot = xyb.XYBot()
    rrts = RRTStar(robot)
    path = rrts.plan(start_conf=np.array([0, 0]), goal_conf=np.array([.6, .9]), obstacle_list=obstacle_list,
                     ext_dist=.1, rand_rate=70, max_time=300, animation=True)
    # Draw final path
    print(path)
    rrts.draw_wspace([rrts.roadmap], rrts.start_conf, rrts.goal_conf, obstacle_list)
    plt.plot([conf[0] for conf in path], [conf[1] for conf in path], linewidth=4, linestyle='-', color='y')
    plt.pause(0.001)
    plt.show()
