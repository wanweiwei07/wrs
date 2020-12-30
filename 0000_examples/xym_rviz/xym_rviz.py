import os
import time
import math
import basis
import numpy as np
from basis import robotmath as rm
import visualization.panda.world as wd
import modeling.geometricmodel as gm
import modeling.collisionmodel as cm
import robotsim.robots.xarm7_shuidi_mobile.xarm7_shuidi_mobile as xav
import motion.probabilistic.rrt_connect as rrtc
import visualization.panda.rpc.rviz_client as rv_client


rvc = rv_client.RVizClient(host="localhost:182001")
# run common_definition at remote end
rvc.load_common_definition('common_definition.py')
# exec at local
exec(rvc.common_definition, globals())
# run common definition at local end
rvc.change_campos_and_lookatpos(np.array([3,0,2]), np.array([0,0,.5]))

global obj
global robot_instance
global robot_jlc_name
global robot_meshmodel_parameters

# # local code
obj.set_pos(np.array([.85, 0, .17]))
obj.set_rgba([.5, .7, .3, 1])
# rvc.add_obj_anime_info(obj)
jlc_name = 'arm'
robot_instance.fk(np.array([0, math.pi * 2 / 3, 0, math.pi, 0, -math.pi / 6, 0]), jlc_name=jlc_name)
rrtc_planner = rrtc.RRTConnect(robot_instance)
path = rrtc_planner.plan(start_conf=np.array([0, math.pi * 2 / 3, 0, math.pi, 0, -math.pi / 6, 0]),
                         goal_conf=np.array([math.pi / 3, math.pi * 1 / 3, 0, math.pi / 2, 0, math.pi / 6, 0]),
                         obstacle_list=[obj],
                         ext_dist=.1,
                         rand_rate=70,
                         maxtime=300,
                         jlc_name=jlc_name)
rvc.add_anime_robot(rmt_robot_instance='robot_instance',
                    loc_robot_jlc_name=robot_jlc_name,
                    loc_robot_meshmodel_parameters=robot_meshmodel_parameters,
                    loc_robot_path=path)
rmt_robot_meshmodel = rvc.add_stationary_robot(rmt_robot_instance='robot_instance', loc_robot_instance=robot_instance)
time.sleep(3)
rvc.delete_anime_robot(rmt_robot_instance='robot_instance')
rvc.delete_stationary_robot(rmt_robot_meshmodel)

