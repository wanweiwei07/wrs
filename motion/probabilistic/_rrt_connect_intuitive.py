import time
import random
import networkx as nx
from motion.probabilistic import rrt


class RRTConnect(rrt.RRT):

    def __init__(self, robot_s):
        super().__init__(robot_s)
        self.roadmap_start = nx.Graph()
        self.roadmap_goal = nx.Graph()

    def _extend_roadmap(self,
                        roadmap,
                        conf,
                        ext_dist,
                        component_name,
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
        new_conf_list = self._extend_conf(roadmap.nodes[nearest_nid]['conf'], conf, ext_dist)
        for new_conf in new_conf_list:
            if self._is_collided(component_name, new_conf, obstacle_list, otherrobot_list):
                return nearest_nid
            else:
                new_nid = random.randint(0, 1e16)
                roadmap.add_node(new_nid, conf=new_conf)
                roadmap.add_edge(nearest_nid, new_nid)
                nearest_nid = new_nid
                # all_sampled_confs.append([new_node.point, False])
                if animation:
                    self.draw_wspace([self.roadmap_start, self.roadmap_goal], self.start_conf, self.goal_conf,
                                     obstacle_list, [roadmap.nodes[nearest_nid]['conf'], conf], new_conf, '^c')
                # check goal
                if self._goal_test(conf=roadmap.nodes[new_nid]['conf'], goal_conf=goal_conf, threshold=ext_dist):
                    roadmap.add_node('connection', conf=goal_conf) # TODO current name -> connection
                    roadmap.add_edge(new_nid, 'connection')
                    return 'connection'
        else:
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
            shortcut = self._shortcut_conf(smoothed_path[i], smoothed_path[j], granularity, exact_end=True)
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
             rand_rate=70,
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
            return [None, None]
        if self._is_collided(component_name, goal_conf, obstacle_list, otherrobot_list):
            print("The goal robot_s configuration is in collision!")
            return [None, None]
        if self._goal_test(conf=start_conf, goal_conf=goal_conf, threshold=ext_dist):
            return [[start_conf, goal_conf], None]
        self.roadmap_start.add_node('start', conf=start_conf)
        self.roadmap_goal.add_node('goal', conf=goal_conf)
        last_nid = 'goal'
        tic = time.time()
        for _ in range(max_iter):
            toc = time.time()
            if max_time > 0.0:
                if toc - tic > max_time:
                    print("Too much motion time! Failed to find a path.")
                    return [None, None]
            # Random Sampling
            goal_nid = 'goal'
            goal_conf = self.roadmap_goal.nodes[goal_nid]['conf']
            rand_conf = self._sample_conf(component_name=component_name, rand_rate=rand_rate, default_conf=goal_conf)
            last_nid = self._extend_roadmap(self.roadmap_start,
                                            conf=rand_conf,
                                            ext_dist=ext_dist,
                                            component_name=component_name,
                                            goal_conf=goal_conf,
                                            obstacle_list=obstacle_list,
                                            otherrobot_list=otherrobot_list,
                                            animation=animation)
            if last_nid == 'connection':
                self.roadmap = nx.compose(self.roadmap_start, self.roadmap_goal)
                self.roadmap.add_edge(last_nid, goal_nid)
                break
            else:
                goal_nid = last_nid
                goal_conf = self.roadmap_start.nodes[goal_nid]['conf']
                rand_conf = self._sample_conf(component_name=component_name, rand_rate=rand_rate, default_conf=goal_conf)
                last_nid = self._extend_roadmap(self.roadmap_goal,
                                                conf=rand_conf,
                                                ext_dist=ext_dist,
                                                component_name=component_name,
                                                goal_conf=goal_conf,
                                                obstacle_list=obstacle_list,
                                                otherrobot_list=otherrobot_list,
                                                animation=animation)
                if last_nid == 'connection':
                    self.roadmap = nx.compose(self.roadmap_start, self.roadmap_goal)
                    self.roadmap.add_edge(last_nid, goal_nid)
                    break
        else:
            print("Reach to maximum iteration! Failed to find a path.")
            return [None, None]
        path = self._path_from_roadmap()
        return path
        smoothed_path = self._smooth_path(component_name=component_name,
                                          path=path,
                                          obstacle_list=obstacle_list,
                                          otherrobot_list=otherrobot_list,
                                          granularity=ext_dist,
                                          iterations=smoothing_iterations)
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
        ((0, 12), 3),
        ((-1, 10), 3),
        ((-2, 8), 3)
    ]  # [x,y,size]
    # Set Initial parameters
    robot = xyb.XYBot()
    rrtc = RRTConnect(robot)
    # path = rrtc.plan(seed_jnt_values=np.array([0, 0]), end_conf=np.array([5, 10]), obstacle_list=obstacle_list,
    #                  ext_dist=1, rand_rate=70, max_time=300, hnd_name=None, animation=True)
    # plt.show()
    # nx.draw(rrt.roadmap, with_labels=True, font_weight='bold')
    # plt.show()
    import time
    total_t = 0
    for i in range(100):
        tic = time.time()
        path = rrtc.plan(start_conf=np.array([0, 0]), goal_conf=np.array([5, 10]), obstacle_list=obstacle_list,
                         ext_dist=1, rand_rate=70, max_time=300, component_name='all', animation=True)
        toc = time.time()
        total_t = total_t + toc - tic
        break
    print(total_t)
    # Draw final path
    print(path)
    plt.plot([conf[0] for conf in path], [conf[1] for conf in path], '-k')
    # pathsm = smoother.pathsmoothing(path, rrt, 30)
    # plt.plot([point[0] for point in pathsm], [point[1] for point in pathsm], '-r')
    # plt.pause(0.001)  # Need for Mac
    plt.show()
