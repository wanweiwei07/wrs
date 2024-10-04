import time
import random
import networkx as nx
from wrs.motion.probabilistic import rrt_differential_wheel as rrtdw


class RRTDWConnect(rrtdw.RRTDW):

    def __init__(self, robot_s):
        super().__init__(robot_s)
        self.roadmap_start = nx.Graph()
        self.roadmap_goal = nx.Graph()

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
        new_conf_list = self._extend_conf(roadmap.nodes[nearest_nid]['conf'], conf, ext_dist)[1:]
        for new_conf in new_conf_list:
            if self._is_collided(component_name, new_conf, obstacle_list, otherrobot_list):
                return -1
            else:
                new_nid = random.randint(0, 1e16)
                roadmap.add_node(new_nid, conf=new_conf)
                roadmap.add_edge(nearest_nid, new_nid)
                nearest_nid = new_nid
                # all_sampled_confs.append([new_node.point, False])
                if animation:
                    self.draw_wspace([self.roadmap_start, self.roadmap_goal], self.start_conf, self.goal_conf,
                                     obstacle_list, [roadmap.nodes[nearest_nid]['conf'], conf], new_conf)
                # check goal
                if self._goal_test(conf=roadmap.nodes[new_nid]['conf'], goal_conf=goal_conf, threshold=ext_dist):
                    roadmap.add_node('connection', conf=goal_conf)  # TODO current name -> connection
                    roadmap.add_edge(new_nid, 'connection')
                    return 'connection'
        return nearest_nid

    def _smooth_path(self,
                     component_name,
                     path,
                     obstacle_list=[],
                     otherrobot_list=[],
                     granularity=2,
                     iterations=50,
                     animation=False):
        smoothed_path = path
        for _ in range(iterations):
            if len(smoothed_path) <= 2:
                return smoothed_path
            i = random.randint(0, len(smoothed_path) - 1)
            j = random.randint(0, len(smoothed_path) - 1)
            if abs(i - j) <= 1:
                continue
            if j < i:
                i, j = j, i
            shortcut = self._extend_conf(smoothed_path[i], smoothed_path[j], granularity)
            if (len(shortcut) <= (j - i) + 1) and all(not self._is_collided(component_name=component_name,
                                                                            conf=conf,
                                                                            obstacle_list=obstacle_list,
                                                                            otherrobot_list=otherrobot_list)
                                                      for conf in shortcut):
                smoothed_path = smoothed_path[:i] + shortcut + smoothed_path[j + 1:]
            if animation:
                self.draw_wspace([self.roadmap_start, self.roadmap_goal], self.start_conf, self.goal_conf,
                                 obstacle_list, shortcut=shortcut, smoothed_path=smoothed_path)
        return smoothed_path

    def plan(self,
             component_name,
             start_conf,
             goal_conf,
             obstacle_list=[],
             otherrobot_list=[],
             ext_dist=2,
             max_iter=1000,
             max_time=15.0,
             smoothing_iterations=50,
             animation=False):
        self.roadmap.clear()
        self.roadmap_start.clear()
        self.roadmap_goal.clear()
        self.start_conf = start_conf
        self.goal_conf = goal_conf
        # check start and goal
        if self._is_collided(component_name, start_conf, obstacle_list, otherrobot_list):
            print("The start robot_s configuration is in collision!")
            return None
        if self._is_collided(component_name, goal_conf, obstacle_list, otherrobot_list):
            print("The goal robot_s configuration is in collision!")
            return None
        if self._goal_test(conf=start_conf, goal_conf=goal_conf, threshold=ext_dist):
            return [start_conf, goal_conf]
        self.roadmap_start.add_node('start', conf=start_conf)
        self.roadmap_goal.add_node('goal', conf=goal_conf)
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
            # one tree grown using random target
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
    from wrs import robot_sim as xyb

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
    robot = xyb.XYWBot()
    rrtdwc = RRTDWConnect(robot)
    path = rrtdwc.plan(component_name='all', start_conf=np.array([0, 0, 0]), goal_conf=np.array([5, 10, 0]),
                       obstacle_list=obstacle_list,
                       ext_dist=1, max_time=300, animation=True)
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
    rrtdwc.draw_wspace([rrtdwc.roadmap], rrtdwc.start_conf, rrtdwc.goal_conf, obstacle_list, delay_time=0)
    for conf in path:
        RRTDWConnect.draw_robot(plt, conf, edgecolor='r')
    # plt.savefig(str(rrtc.img_counter)+'.jpg')
    # pathsm = smoother.pathsmoothing(path, rrt, 30)
    # plt.plot([point[0] for point in pathsm], [point[1] for point in pathsm], '-r')
    # plt.pause(0.001)  # Need for Mac
    plt.show()
