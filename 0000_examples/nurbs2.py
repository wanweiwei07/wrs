from geomdl import construct
from geomdl import fitting
from geomdl.visualization import VisMPL as vis
import numpy as np
import modeling.geometric_model as gm
import visualization.panda.world as wd
import basis.robot_math as rm
import math
from scipy.interpolate import RBFInterpolator

base = wd.World(cam_pos=np.array([.5,.1,.3]), lookat_pos=np.array([0,0,0.05]))
gm.gen_frame().attach_to(base)
tube_model = gm.GeometricModel(initor="./objects/bowl.stl")
tube_model.attach_to(base)
points, pfid = tube_model.sample_surface(radius=.002, nsample=10000)
points_normals = tube_model.objtrm.face_normals[pfid]
sampled_points = []
for id, p in enumerate(points.tolist()):
    if np.dot(np.array([1,0,0]), points_normals[id]) > .3 and p[0]>0:
        gm.gen_sphere(pos=p, radius=.001).attach_to(base)
        sampled_points.append(p)

# x - v
# y - u
rotmat_uv = rm.rotmat_from_euler(0, math.pi/2, 0)
sampled_points = rotmat_uv.dot(np.array(sampled_points).T).T
surface = RBFInterpolator(sampled_points[:, :2], sampled_points[:,2])
xgrid = np.mgrid[1:120, -50:50]*.001
xflat = xgrid.reshape(2,-1).T
zflat = surface(xflat)
interpolated_points = np.column_stack((xflat, zflat))
interpolated_points = rotmat_uv.T.dot(interpolated_points.T).T
for p in interpolated_points:
    gm.gen_sphere(p, rgba=[0,1,0,1], radius=.0005).attach_to(base)
base.run()