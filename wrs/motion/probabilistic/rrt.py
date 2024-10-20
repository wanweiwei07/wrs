import time
import math
import uuid
import random
import scipy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from operator import itemgetter
import wrs.basis.robot_math as rm
import wrs.motion.motion_data as motd
import wrs.modeling.geometric_model as mgm


class RRT(object):
    """
    author: weiwei
    date: 20230807
    """

    def __init__(self, robot):
        self.robot = robot
        self.roadmap = nx.Graph()
        self.start_conf = None
        self.goal_conf = None
        # define data type
        self.toggle_keep = True

    @staticmethod
    def keep_states_decorator(method):
        """
        decorator function for save and restore robot's joint values
        applicable to both single or multi-arm sgl_arm_robots
        :return:
        author: weiwei
        date: 20220404
        """

        def wrapper(self, *args, **kwargs):
            if self.toggle_keep:
                self.robot.backup_state()
                result = method(self, *args, **kwargs)
                self.robot.restore_state()
                return result
            else:
                result = method(self, *args, **kwargs)
                return result

        return wrapper

    def _is_collided(self,
                     conf,
                     obstacle_list=[],
                     other_robot_list=[],
                     toggle_contacts=False):
        """
        The function first examines if joint values of the given conf are in ranges.
        It will promptly return False if any joint value is out of range.
        Or else, it will compute fk and carry out collision checking.
        :param conf:
        :param obstacle_list:
        :param other_robot_list:
        :param toggle_contacts: for debugging collisions at start/goal
        :return:
        author: weiwei
        date: 20220326, 20240314
        """
        if self.robot.are_jnts_in_ranges(jnt_values=conf):
            self.robot.goto_given_conf(jnt_values=conf)
            # # toggle off the following code to consider object pose constraints
            # if len(self.robot.oiee_list)>0:
            #     angle = rm.angle_between_vectors(self.robot.oiee_list[-1].gl_rotmat[:,2], np.array([0,0,1]))
            #     if angle > np.radians(10):
            #         return True
            collision_info = self.robot.is_collided(obstacle_list=obstacle_list, other_robot_list=other_robot_list,
                                                    toggle_contacts=toggle_contacts)
            # if toggle_contacts:
            #     if collision_info[0]:
            #         for pnt in collision_info[1]:
            #             print(pnt)
            #             mgm.gen_sphere(pos=pnt, radius=.01).attach_to(base)
            #         self.robot.gen_meshmodel(toggle_cdprim=True).attach_to(base)
            #         for obs in obstacle_list:
            #             obs.rgb=np.array([1,1,1])
            #             obs.show_cdprim()
            #             obs.attach_to(base)
            #         base.run()
            return collision_info
        else:
            print("The given joint angles are out of joint limits.")
            return (True, []) if toggle_contacts else True

    def _sample_conf(self, rand_rate, default_conf):
        if random.randint(0, 99) < rand_rate:
            return self.robot.rand_conf()
        else:
            return default_conf

    def _get_nearest_nid(self, roadmap, new_conf):
        """
        convert to numpy to accelerate access
        :param roadmap:
        :param new_conf:
        :return:
        author: weiwei
        date: 20210523
        """
        nodes_dict = dict(roadmap.nodes(data="conf"))
        nodes_key_list = list(nodes_dict.keys())  # use python > 3.7, or else there is no guarantee on the order
        nodes_value_list = list(nodes_dict.values())  # attention, correspondence is not guanranteed in python
        # ===============
        # the following code computes euclidean distances. it is decprecated and replaced using cdtree
        # ***** date: 20240304, correspondent: weiwei *****
        # conf_array = np.array(nodes_value_list)
        # diff_conf_array = np.linalg.norm(conf_array - new_conf, axis=1)
        # min_dist_nid = np.argmin(diff_conf_array)
        # return nodes_key_list[min_dist_nid]
        # ===============
        querry_tree = scipy.spatial.cKDTree(nodes_value_list)
        dist_value, indx = querry_tree.query(new_conf, k=1, workers=-1)
        return nodes_key_list[indx]

    def _extend_conf(self, src_conf, end_conf, ext_dist, exact_end=True):
        """
        :param src_conf:
        :param end_conf:
        :param ext_dist:
        :param exact_end:
        :return: a list of 1xn nparray
        """
        len, vec = rm.unit_vector(end_conf - src_conf, toggle_length=True)
        # ===============
        # one step extension: not used because it is slower than full extensions
        # ***** date: 20210523, correspondent: weiwei *****
        # return [src_conf + ext_dist * vec]
        # switch to the following code for ful extensions
        # ===============
        if not exact_end:
            nval = math.ceil(len / ext_dist)
            nval = 1 if nval == 0 else nval  # at least include itself
            conf_array = np.linspace(src_conf, src_conf + nval * ext_dist * vec, nval)
        else:
            nval = math.floor(len / ext_dist)
            nval = 1 if nval == 0 else nval  # at least include itself
            conf_array = np.linspace(src_conf, src_conf + nval * ext_dist * vec, nval)
            conf_array = np.vstack((conf_array, end_conf))
        return list(conf_array)

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
        new_conf_list = self._extend_conf(roadmap.nodes[nearest_nid]["conf"], conf, ext_dist)[1:]
        for new_conf in new_conf_list:
            if self._is_collided(new_conf, obstacle_list, other_robot_list):
                return nearest_nid
            else:
                new_nid = uuid.uuid4()
                roadmap.add_node(new_nid, conf=new_conf)
                roadmap.add_edge(nearest_nid, new_nid)
                nearest_nid = new_nid
                # all_sampled_confs.append([new_node.point, False])
                if animation:
                    self.draw_wspace([roadmap], self.start_conf, self.goal_conf,
                                     obstacle_list, [roadmap.nodes[nearest_nid]["conf"], conf],
                                     new_conf, '^c')
                # check goal
                if self._is_goal_reached(conf=roadmap.nodes[new_nid]["conf"], goal_conf=goal_conf, threshold=ext_dist):
                    roadmap.add_node("goal", conf=goal_conf)
                    roadmap.add_edge(new_nid, "goal")
                    return "goal"
        return nearest_nid

    def _is_goal_reached(self, conf, goal_conf, threshold):
        dist = np.linalg.norm(conf - goal_conf)
        if dist <= threshold:
            # print("Goal reached!")
            return True
        else:
            return False

    def _path_from_roadmap(self):
        nid_path = nx.shortest_path(self.roadmap, source="start", target="goal")
        return list(itemgetter(*nid_path)(self.roadmap.nodes(data="conf")))

    def _smooth_path(self,
                     path,
                     obstacle_list=[],
                     other_robot_list=[],
                     granularity=.2,
                     n_iter=50,
                     animation=False):
        smoothed_path = path
        for _ in range(n_iter):
            if len(smoothed_path) <= 2:
                return smoothed_path
            i = random.randint(0, len(smoothed_path) - 1)
            j = random.randint(0, len(smoothed_path) - 1)
            if abs(i - j) <= 1:
                continue
            if j < i:
                i, j = j, i
            exact_end = True if j == len(smoothed_path) - 1 else False
            shortcut = self._extend_conf(src_conf=smoothed_path[i], end_conf=smoothed_path[j], ext_dist=granularity,
                                         exact_end=exact_end)
            if all(not self._is_collided(conf=conf,
                                         obstacle_list=obstacle_list,
                                         other_robot_list=other_robot_list)
                   for conf in shortcut):
                smoothed_path = smoothed_path[:i] + shortcut + smoothed_path[j + 1:]
            if animation:
                self.draw_wspace([self.roadmap], self.start_conf, self.goal_conf,
                                 obstacle_list, shortcut=shortcut, smoothed_path=smoothed_path)
            if i == 0 and exact_end:  # stop smoothing when shortcut was between start and end
                break
        return smoothed_path

    @keep_states_decorator
    def plan(self,
             start_conf,
             goal_conf,
             obstacle_list=[],
             other_robot_list=[],
             ext_dist=.2,
             rand_rate=70,
             max_n_iter=1000,
             max_time=15.0,
             smoothing_n_iter=50,
             animation=False):
        """
        :return: [path, all_sampled_confs]
        author: weiwei
        date: 20201226
        """
        self.roadmap.clear()
        self.start_conf = start_conf
        self.goal_conf = goal_conf
        # check start_conf and end_conf
        if self._is_collided(start_conf, obstacle_list, other_robot_list):
            print("The start robot configuration is in collision!")
            return None
        if self._is_collided(goal_conf, obstacle_list, other_robot_list):
            print("The goal robot configuration is in collision!")
            return None
        if self._is_goal_reached(conf=start_conf, goal_conf=goal_conf, threshold=ext_dist):
            mot_data = motd.MotionData(self.robot)
            mot_data.extend(jv_list=[start_conf, goal_conf])
            return mot_data
        self.roadmap.add_node("start", conf=start_conf)
        tic = time.time()
        for _ in range(max_n_iter):
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
            if last_nid == "goal":
                path = self._path_from_roadmap()
                smoothed_path = self._smooth_path(path=path,
                                                  obstacle_list=obstacle_list,
                                                  other_robot_list=other_robot_list,
                                                  granularity=ext_dist,
                                                  n_iter=smoothing_n_iter,
                                                  animation=animation)
                mot_data = motd.MotionData(self.robot)
                if getattr(base, "toggle_mesh", True):
                    mot_data.extend(jv_list=smoothed_path)
                else:
                    mot_data.extend(jv_list=smoothed_path, mesh_list=[])
                return mot_data
        else:
            print("Failed to find a path with the given max_n_ter!")
            return None

    @staticmethod
    def draw_wspace(roadmap_list,
                    start_conf,
                    goal_conf,
                    obstacle_list,
                    near_rand_conf_pair=None,
                    new_conf=None,
                    new_conf_mark='^r',
                    shortcut=None,
                    smoothed_path=None,
                    delay_time=.001):
        """
        Draw Graph
        """
        plt.clf()
        ax = plt.gca()
        ax.set_aspect("equal", "box")
        plt.grid(True)
        plt.xlim(-.4, 1.7)
        plt.ylim(-.4, 1.7)
        ax.add_patch(plt.Circle((start_conf[0], start_conf[1]), .05, color='r'))
        ax.add_patch(plt.Circle((goal_conf[0], goal_conf[1]), .05, color='g'))
        for (point, size) in obstacle_list:
            ax.add_patch(plt.Circle((point[0], point[1]), size / 2.0, color='k'))
        colors = "bgrcmykw"
        for i, roadmap in enumerate(roadmap_list):
            for (u, v) in roadmap.edges:
                plt.plot(roadmap.nodes[u]["conf"][0], roadmap.nodes[u]["conf"][1], 'o' + colors[i])
                plt.plot(roadmap.nodes[v]["conf"][0], roadmap.nodes[v]["conf"][1], 'o' + colors[i])
                plt.plot([roadmap.nodes[u]["conf"][0], roadmap.nodes[v]["conf"][0]],
                         [roadmap.nodes[u]["conf"][1], roadmap.nodes[v]["conf"][1]], '-' + colors[i])
        if near_rand_conf_pair is not None:
            plt.plot([near_rand_conf_pair[0][0], near_rand_conf_pair[1][0]],
                     [near_rand_conf_pair[0][1], near_rand_conf_pair[1][1]], "--k")
            ax.add_patch(plt.Circle((near_rand_conf_pair[1][0], near_rand_conf_pair[1][1]), .03, color='grey'))
        if new_conf is not None:
            plt.plot(new_conf[0], new_conf[1], new_conf_mark)
        if smoothed_path is not None:
            plt.plot([conf[0] for conf in smoothed_path], [conf[1] for conf in smoothed_path], linewidth=7,
                     linestyle='-', color='c')
        if shortcut is not None:
            plt.plot([conf[0] for conf in shortcut], [conf[1] for conf in shortcut], linewidth=4, linestyle='--',
                     color='r')
        if not hasattr(RRT, "img_counter"):
            RRT.img_counter = 0
        else:
            RRT.img_counter += 1
        # plt.savefig(str(RRT.img_counter)+'.jpg')
        if delay_time > 0:
            plt.pause(delay_time)
        # plt.waitforbuttonpress()


if __name__ == "__main__":
    import wrs.robot_sim.robots.xybot.xybot as robot
    # ====Search Path with RRT====
    obstacle_list = [
        ((.5, .5), .3),
        ((.3, .6), .3),
        ((.3, .8), .3),
        ((.3, 1.0), .3),
        ((.7, .5), .3),
        ((.9, .5), .3),
        ((1.0, .5), .3)
    ]  # [[x,y],size]
    robot = robot.XYBot()
    rrt = RRT(robot)
    path = rrt.plan(start_conf=np.array([0, 0]), goal_conf=np.array([.6, .9]), obstacle_list=obstacle_list,
                    ext_dist=.1, rand_rate=70, max_time=300, animation=True)
    # Draw final path
    print(path)
    rrt.draw_wspace(roadmap_list=[rrt.roadmap], start_conf=rrt.start_conf, goal_conf=rrt.goal_conf,
                    obstacle_list=obstacle_list)
    plt.plot([conf[0] for conf in path], [conf[1] for conf in path], linewidth=4, color='y')
    # plt.savefig(str(rrtc.img_counter)+'.jpg')
    plt.pause(0.001)
    plt.show()
