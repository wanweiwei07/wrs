import time
import math
import random
import numpy as np
import basis.robot_math as rm
import networkx as nx
import matplotlib.pyplot as plt
import rrt
from operator import itemgetter


class RRTStar(rrt.RRT):

    def __init__(self, robot_s, nearby_ratio=2):
        """
        :param robot_s:
        :param nearby_ratio: the threshold_hold = ext_dist*nearby_ratio
        """
        super().__init__(robot_s)
        self.roadmap = nx.DiGraph()
        self.nearby_ratio = nearby_ratio

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

    def _extend_conf(self, conf1, conf2, ext_dist):
        """
        :param conf1:
        :param conf2:
        :param ext_dist:
        :return: a list of 1xn nparray
        """
        len, vec = rm.unit_vector(conf2 - conf1, toggle_length=True)
        return conf1 + ext_dist * vec if len > 1e-6 else None

    def _extend_roadmap(self,
                        component_name,
                        roadmap,
                        conf,
                        ext_dist,
                        goal_conf,
                        obstacle_list=[],
                        otherrobot_list=[],
                        animation=False):
        """
        find the nearest point between the given roadmap and the conf and then extend towards the conf
        :return:
        author: weiwei
        date: 20201228
        """
        nearest_nid = self._get_nearest_nid(roadmap, conf)
        new_conf = self._extend_conf(roadmap.nodes[nearest_nid]['conf'], conf, ext_dist)
        if new_conf is not None:
            if self._is_collided(component_name, new_conf, obstacle_list, otherrobot_list):
                return -1
            else:
                new_nid = random.randint(0, 1e16)
                # find nearby_nid_list
                nearby_nid_list = self._get_nearby_nid_with_min_cost(roadmap, new_conf, ext_dist)
                print(nearby_nid_list) # 20210523 cannot continue to simplify
                # costs
                nodes_cost_dict = dict(roadmap.nodes(data='cost'))
                nearby_cost_list = itemgetter(*nearby_nid_list)(nodes_cost_dict)
                if type(nearby_cost_list) == np.ndarray:
                    nearby_cost_list = [nearby_cost_list]
                nearby_min_cost_nid = nearby_nid_list[np.argmin(np.array(nearby_cost_list))]
                roadmap.add_node(new_nid, conf=new_conf, cost=0)  # add new nid
                roadmap.add_edge(nearby_min_cost_nid, new_nid)  # add new edge
                roadmap.nodes[new_nid]['cost'] = roadmap.nodes[nearby_min_cost_nid]['cost'] + 1  # update cost
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
                if self._goal_test(conf=roadmap.nodes[new_nid]['conf'], goal_conf=goal_conf, threshold=ext_dist):
                    roadmap.add_node('connection', conf=goal_conf)  # TODO current name -> connection
                    roadmap.add_edge(new_nid, 'connection')
                    return 'connection'
                return nearby_min_cost_nid

    def plan(self,
             component_name,
             start_conf,
             goal_conf,
             obstacle_list=[],
             otherrobot_list=[],
             ext_dist=2,
             rand_rate=70,
             max_iter=1000,
             max_time=15.0,
             smoothing_iterations=17,
             animation=False):
        """
        :return: [path, all_sampled_confs]
        author: weiwei
        date: 20201226
        """
        self.roadmap.clear()
        self.start_conf = start_conf
        self.goal_conf = goal_conf
        # check seed_jnt_values and end_conf
        if self._is_collided(component_name, start_conf, obstacle_list, otherrobot_list):
            print("The start robot_s configuration is in collision!")
            return None
        if self._is_collided(component_name, goal_conf, obstacle_list, otherrobot_list):
            print("The goal robot_s configuration is in collision!")
            return None
        if self._goal_test(conf=start_conf, goal_conf=goal_conf, threshold=ext_dist):
            return [[start_conf, goal_conf], None]
        self.roadmap.add_node('start', conf=start_conf, cost=0)
        tic = time.time()
        n = 0
        for _ in range(max_iter):
            n+=1
            toc = time.time()
            if max_time > 0.0:
                if toc - tic > max_time:
                    print("Too much motion time! Failed to find a path.")
                    return None
            # Random Sampling
            rand_conf = self._sample_conf(component_name=component_name, rand_rate=rand_rate, default_conf=goal_conf)
            last_nid = self._extend_roadmap(component_name=component_name,
                                            roadmap=self.roadmap,
                                            conf=rand_conf,
                                            ext_dist=ext_dist,
                                            goal_conf=goal_conf,
                                            obstacle_list=obstacle_list,
                                            otherrobot_list=otherrobot_list,
                                            animation=animation)
            if last_nid == 'connection' and n > 1000:
                mapping = {'connection': 'goal'}
                self.roadmap = nx.relabel_nodes(self.roadmap, mapping)
                path = self._path_from_roadmap()
                smoothed_path = self._smooth_path(component_name=component_name,
                                                  path=path,
                                                  obstacle_list=obstacle_list,
                                                  otherrobot_list=otherrobot_list,
                                                  granularity=ext_dist,
                                                  iterations=smoothing_iterations,
                                                  animation=animation)
                return smoothed_path
        else:
            print("Reach to maximum iteration! Failed to find a path.")
            return None


if __name__ == '__main__':
    import robot_sim.robots.xybot.xybot as xyb


    # ====Search Path with RRT====
    obstacle_list = [
        ((5, 5), 3),
        ((3, 6), 3),
        ((3, 8), 3),
        ((3, 10), 3),
        ((7, 5), 3),
        ((9, 5), 3),
        ((10, 5), 3)
    ]  # [x,y,size]
    # Set Initial parameters
    robot_s = xyb.XYBot()
    rrtstar_s = RRTStar(robot_s, nearby_ratio=2)
    path = rrtstar_s.plan(start_conf=np.array([0, 0]), goal_conf=np.array([6, 9]), obstacle_list=obstacle_list,
                          ext_dist=1, rand_rate=70, max_time=300, component_name='all', smoothing_iterations=0,
                          animation=True)
    # plt.show()
    # nx.draw(rrt.roadmap, with_labels=True, font_weight='bold')
    # plt.show()
    # import time
    # total_t = 0
    # for i in range(1):
    #     tic = time.time()
    #     path, sampledpoints = rrt.motion(obstaclelist=obstaclelist, animation=True)
    #     toc = time.time()
    #     total_t = total_t + toc - tic
    # print(total_t)
    # Draw final path
    print(path)
    rrtstar_s.draw_wspace([rrtstar_s.roadmap], rrtstar_s.start_conf, rrtstar_s.goal_conf, obstacle_list, delay_time=0)
    plt.plot([conf[0] for conf in path], [conf[1] for conf in path], linewidth=7, linestyle='-', color='c')
    # pathsm = smoother.pathsmoothing(path, rrt, 30)
    # plt.plot([point[0] for point in pathsm], [point[1] for point in pathsm], '-r')
    # plt.pause(0.001)  # Need for Mac
    plt.show()
