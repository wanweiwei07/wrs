import time
import random
import numpy as np
import basis.robot_math as rm
import networkx as nx
import matplotlib.pyplot as plt
import rrt_star as rrtst
from operator import itemgetter


class RRTStarConnect(rrtst.RRTStar):

    def __init__(self, robot_s, nearby_ratio=2):
        """
        :param robot_s:
        :param nearby_ratio: the threshold_hold = ext_dist*nearby_ratio
        """
        super().__init__(robot_s)
        self.nearby_ratio = nearby_ratio
        self.roadmap_start = nx.Graph()
        self.roadmap_goal = nx.Graph()

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
                        if roadmap.nodes[nearby_min_cost_nid]['cost'] + 1 < roadmap.nodes[nearby_nid]['cost']:
                            nearby_e_nid = list(roadmap.neighbors(nearby_nid))[0]
                            roadmap.remove_edge(nearby_nid, nearby_e_nid)
                            roadmap.add_edge(nearby_nid, nearby_min_cost_nid)
                            roadmap.nodes[nearby_nid]['cost'] = roadmap.nodes[nearby_min_cost_nid]['cost'] + 1
                if animation:
                    self.draw_wspace([self.roadmap_start, self.roadmap_goal], self.start_conf, self.goal_conf,
                                     obstacle_list, [roadmap.nodes[nearest_nid]['conf'], conf], new_conf, '^c')
                # check goal
                if self._goal_test(conf=roadmap.nodes[new_nid]['conf'], goal_conf=goal_conf, threshold=ext_dist):
                    roadmap.add_node('connection', conf=goal_conf)  # TODO current name -> connection
                    roadmap.add_edge(new_nid, 'connection')
                    return 'connection'
                return new_nid
        return nearest_nid

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
        self.roadmap_start.clear()
        self.roadmap_goal.clear()
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
        self.roadmap_start.add_node('start', conf=start_conf, cost=0)
        self.roadmap_goal.add_node('goal', conf=goal_conf, cost=0)
        tic = time.time()
        tree_a = self.roadmap_start
        tree_b = self.roadmap_goal
        tree_a_goal_conf = self.roadmap_goal.nodes['goal']['conf']
        tree_b_goal_conf = self.roadmap_start.nodes['start']['conf']
        for _ in range(max_iter):
            toc = time.time()
            if max_time > 0.0:
                if toc - tic > max_time:
                    print("Too much motion time! Failed to find a path.")
                    return None
            # Random Sampling
            rand_conf = self._sample_conf(component_name=component_name,
                                          rand_rate=100,
                                          default_conf=None)
            last_nid = self._extend_roadmap(component_name=component_name,
                                            roadmap=tree_a,
                                            conf=rand_conf,
                                            ext_dist=ext_dist,
                                            goal_conf=tree_a_goal_conf,
                                            obstacle_list=obstacle_list,
                                            otherrobot_list=otherrobot_list,
                                            animation=animation)
            if last_nid != -1:  # not trapped:
                goal_nid = last_nid
                tree_b_goal_conf = tree_a.nodes[goal_nid]['conf']
                last_nid = self._extend_roadmap(component_name=component_name,
                                                roadmap=tree_b,
                                                conf=tree_a.nodes[last_nid]['conf'],
                                                ext_dist=ext_dist,
                                                goal_conf=tree_b_goal_conf,
                                                obstacle_list=obstacle_list,
                                                otherrobot_list=otherrobot_list,
                                                animation=animation)
                if last_nid == 'connection':
                    self.roadmap = nx.compose(tree_a, tree_b)
                    self.roadmap.add_edge(last_nid, goal_nid)
                    break
                elif last_nid != -1:
                    goal_nid = last_nid
                    tree_a_goal_conf = tree_b.nodes[goal_nid]['conf']
            if tree_a.number_of_nodes() > tree_b.number_of_nodes():
                tree_a, tree_b = tree_b, tree_a
                tree_a_goal_conf, tree_b_goal_conf = tree_b_goal_conf, tree_a_goal_conf
        else:
            print("Reach to maximum iteration! Failed to find a path.")
            return None
        path = self._path_from_roadmap()
        smoothed_path = self._smooth_path(component_name=component_name,
                                          path=path,
                                          obstacle_list=obstacle_list,
                                          otherrobot_list=otherrobot_list,
                                          granularity=ext_dist,
                                          iterations=smoothing_iterations,
                                          animation=animation)
        return smoothed_path


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import robot_sim.robots.xybot.xybot as xyb

    # ====Search Path with RRT====
    obstacle_list = [
        ((5, 5), 3),
        ((3, 6), 3),
        ((3, 8), 3),
        ((3, 10), 3),
        ((7, 5), 3),
        ((9, 5), 3),
        ((10, 5), 3),
        ((10, 0), 3),
        ((10, -2), 3),
        ((10, -4), 3),
        ((15, 5), 3),
        ((15, 7), 3),
        ((15, 9), 3),
        ((15, 11), 3),
        ((0, 12), 3),
        ((-1, 10), 3),
        ((-2, 8), 3)
    ]  # [x,y,size]
    # Set Initial parameters
    robot = xyb.XYBot()
    rrtsc = RRTStarConnect(robot)
    path = rrtsc.plan(component_name='all', start_conf=np.array([0, 0]), goal_conf=np.array([5, 10]),
                      obstacle_list=obstacle_list,
                      ext_dist=1, rand_rate=70, max_time=300, animation=True)
    # import time
    # total_t = 0
    # for i in range(100):
    #     tic = time.time()
    #     path = rrtc.plan(seed_jnt_values=np.array([0, 0]), end_conf=np.array([5, 10]), obstacle_list=obstacle_list,
    #                      ext_dist=1, rand_rate=70, max_time=300, hnd_name=None, animation=False)
    #     toc = time.time()
    #     total_t = total_t + toc - tic
    # print(total_t)
    # Draw final path
    print(path)
    rrtsc.draw_wspace([rrtsc.roadmap_start, rrtsc.roadmap_goal],
                     rrtsc.start_conf, rrtsc.goal_conf, obstacle_list, delay_time=0)
    plt.plot([conf[0] for conf in path], [conf[1] for conf in path], linewidth=7, linestyle='-', color='c')
    # plt.savefig(str(rrtc.img_counter)+'.jpg')
    # pathsm = smoother.pathsmoothing(path, rrt, 30)
    # plt.plot([point[0] for point in pathsm], [point[1] for point in pathsm], '-r')
    # plt.pause(0.001)  # Need for Mac
    plt.show()
