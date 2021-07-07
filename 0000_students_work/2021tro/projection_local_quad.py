import numpy as np
import modeling.geometric_model as gm
import modeling.collision_model as cm
import visualization.panda.world as wd
import basis.robot_math as rm
import math
from scipy.spatial import cKDTree
import vision.depth_camera.surface.quadrantic_surface as qs

base = wd.World(cam_pos=np.array([-.3,-.9,.3]), lookat_pos=np.array([0,0,0]))
# gm.gen_frame().attach_to(base)
bowl_model = cm.CollisionModel(initor="./objects/bowl.stl")
bowl_model.set_rgba([.3,.3,.3,.3])
bowl_model.set_rotmat(rm.rotmat_from_euler(math.pi,0,0))
bowl_model.attach_to(base)

pn_direction = np.array([0, 0, -1])

bowl_samples, bowl_sample_normals = bowl_model.sample_surface(toggle_option='normals', radius=.002)
selection = bowl_sample_normals.dot(-pn_direction)>.1
bowl_samples = bowl_samples[selection]
bowl_sample_normals=bowl_sample_normals[selection]
tree = cKDTree(bowl_samples)

pt_direction = rm.orthogonal_vector(pn_direction, toggle_unit=True)
tmp_direction = np.cross(pn_direction, pt_direction)
plane_rotmat = np.column_stack((pt_direction, tmp_direction, pn_direction))
homomat=np.eye(4)
homomat[:3,:3] = plane_rotmat
homomat[:3,3] = np.array([-.07,-.03,.1])
twod_plane = gm.gen_box(np.array([.2, .2, .001]), homomat=homomat, rgba=[1,1,1,.3])
twod_plane.attach_to(base)

circle_radius=.05
line_segs = [[homomat[:3,3], homomat[:3,3]+pt_direction*.05], [homomat[:3,3]+pt_direction*.05, homomat[:3,3]+pt_direction*.05+tmp_direction*.05],
             [homomat[:3,3]+pt_direction*.05+tmp_direction*.05, homomat[:3,3]+tmp_direction*.05], [homomat[:3,3]+tmp_direction*.05, homomat[:3,3]]]
# gm.gen_linesegs(line_segs).attach_to(base)
for sec in line_segs:
    gm.gen_stick(spos=sec[0], epos=sec[1], rgba=[0, 0, 0, 1], thickness=.002, type='round').attach_to(base)
epos = (line_segs[0][1]-line_segs[0][0])*.7+line_segs[0][0]
gm.gen_arrow(spos=line_segs[0][0], epos=epos, thickness=0.004).attach_to(base)
spt = homomat[:3,3]
# gm.gen_stick(spt, spt + pn_direction * 10, rgba=[0,1,0,1]).attach_to(base)
# base.run()
gm.gen_dasharrow(spt, spt-pn_direction*.07, thickness=.004).attach_to(base) # p0
cpt, cnrml = bowl_model.ray_hit(spt, spt + pn_direction * 10000, option='closest')
gm.gen_dashstick(spt, cpt, rgba=[.57,.57,.57,.7], thickness=0.003).attach_to(base)
gm.gen_sphere(pos=cpt, radius=.005).attach_to(base)
gm.gen_dasharrow(cpt, cpt-pn_direction*.07, thickness=.004).attach_to(base) # p0
gm.gen_dasharrow(cpt, cpt+cnrml*.07, thickness=.004).attach_to(base) # p0

angle = rm.angle_between_vectors(-pn_direction, cnrml)
vec = np.cross(-pn_direction, cnrml)
rotmat = rm.rotmat_from_axangle(vec, angle)
new_plane_homomat = np.eye(4)
new_plane_homomat[:3,:3] = rotmat.dot(homomat[:3,:3])
new_plane_homomat[:3,3] = cpt
twod_plane = gm.gen_box(np.array([.2, .2, .001]), homomat=new_plane_homomat, rgba=[1,1,1,.3])
twod_plane.attach_to(base)
new_line_segs = [[cpt, cpt+rotmat.dot(pt_direction)*.05],
                 [cpt+rotmat.dot(pt_direction)*.05, cpt+rotmat.dot(pt_direction)*.05+rotmat.dot(tmp_direction)*.05],
                 [cpt+rotmat.dot(pt_direction)*.05+rotmat.dot(tmp_direction)*.05, cpt+rotmat.dot(tmp_direction)*.05],
                 [cpt+rotmat.dot(tmp_direction)*.05, cpt]]
# gm.gen_linesegs(new_line_segs).attach_to(base)
# for sec in [new_line_segs[0]]:
#     gm.gen_stick(spos=sec[0], epos=sec[1], rgba=[0, 0, 0, 1], thickness=.002, type='round').attach_to(base)
epos = (new_line_segs[0][1]-new_line_segs[0][0])*.7+new_line_segs[0][0]
gm.gen_arrow(spos=new_line_segs[0][0], epos=epos, thickness=0.004).attach_to(base)

t_cpt = cpt
last_normal = cnrml
direction = rotmat.dot(pt_direction)
tmp_direction = rotmat.dot(tmp_direction)
n=5
for tick in range(1, n+1):
    t_npt = cpt+direction*.05/n
    gm.gen_arrow(spos=t_npt, epos=t_npt+last_normal*.025, thickness=0.001, rgba=[1, 1, 0, 1]).attach_to(base)
    nearby_sample_ids = tree.query_ball_point(t_npt, .07)
    nearby_samples = bowl_samples[nearby_sample_ids]
    # gm.GeometricModel(nearby_samples).attach_to(base)
    plane_center, plane_normal = rm.fit_plane(nearby_samples)
    plane_tangential = rm.orthogonal_vector(plane_normal)
    plane_tmp = np.cross(plane_normal, plane_tangential)
    plane_rotmat = np.column_stack((plane_tangential, plane_tmp, plane_normal))
    if pn_direction.dot(plane_normal) > .1:
        plane_rotmat = np.column_stack((plane_tmp, plane_tangential, -plane_normal))
    nearby_samples_on_xy = plane_rotmat.T.dot((nearby_samples-plane_center).T).T
    surface = qs.QuadraticSurface(nearby_samples_on_xy[:, :2], nearby_samples_on_xy[:,2])
    t_npt_on_xy = plane_rotmat.T.dot(t_npt-plane_center)
    projected_t_npt_z_on_xy = surface.get_zdata(np.array([t_npt_on_xy[:2]]))
    projected_t_npt_on_xy = np.array([t_npt_on_xy[0], t_npt_on_xy[1], projected_t_npt_z_on_xy[0]])
    projected_point = plane_rotmat.dot(projected_t_npt_on_xy)+plane_center
    surface_gm = surface.get_gometricmodel([[-.05,.05],[-.05,.05]], rgba=[.5,.7,1,.2])
    surface_gm.set_pos(plane_center)
    surface_gm.set_rotmat(plane_rotmat)
    surface_gm.attach_to(base)
    # homomat = np.eye(4)
    # homomat[:3,:3]=plane_rotmat
    # homomat[:3,3]=plane_center
    # twod_plane = gm.gen_box(np.array([.1, .1, .001]), homomat=homomat, rgba=[.5,.7,1,.2]).attach_to(base)
    # projected_point = rm.project_to_plane(t_npt, plane_center, plane_normal)
    # gm.gen_stick(t_npt, projected_point, thickness=.002).attach_to(base)
    new_normal = rm.unit_vector(t_npt-projected_point)
    if pn_direction.dot(new_normal) > .1:
        new_normal = -new_normal
    gm.gen_arrow(spos=projected_point, epos=projected_point+new_normal*.025, thickness=0.001).attach_to(base)
    angle = rm.angle_between_vectors(last_normal, new_normal)
    vec = rm.unit_vector(np.cross(last_normal, new_normal))
    new_rotmat = rm.rotmat_from_axangle(vec, angle)
    direction = new_rotmat.dot(direction)
    tmp_direction = new_rotmat.dot(tmp_direction)
    # new_line_segs = [[cpt, cpt+direction*(.05-tick*.05/n)],
    #                  [cpt+direction*(.05-tick*.05/n), cpt+direction*(.05-tick*.05/n)+tmp_direction*.05]]
    # gm.gen_linesegs(new_line_segs).attach_to(base)
    gm.gen_stick(spos=cpt, epos=projected_point, rgba=[1,.6,0,1], thickness=.002, type='round').attach_to(base)
    cpt=projected_point
    last_normal = new_normal
    # break

t_cpt = cpt
direction = new_rotmat.dot(tmp_direction)
for tick in range(1, n+1):
    t_npt = cpt+direction*.05/n
    gm.gen_arrow(spos=t_npt, epos=t_npt+last_normal*.025, thickness=0.001, rgba=[1, 1, 0, 1]).attach_to(base)
    nearby_sample_ids = tree.query_ball_point(t_npt, .07)
    nearby_samples = bowl_samples[nearby_sample_ids]
    # gm.GeometricModel(nearby_samples).attach_to(base)
    plane_center, plane_normal = rm.fit_plane(nearby_samples)
    plane_tangential = rm.orthogonal_vector(plane_normal)
    plane_tmp = np.cross(plane_normal, plane_tangential)
    plane_rotmat = np.column_stack((plane_tangential, plane_tmp, plane_normal))
    if pn_direction.dot(plane_normal) > .1:
        plane_rotmat = np.column_stack((plane_tmp, plane_tangential, -plane_normal))
    nearby_samples_on_xy = plane_rotmat.T.dot((nearby_samples-plane_center).T).T
    surface = qs.QuadraticSurface(nearby_samples_on_xy[:, :2], nearby_samples_on_xy[:,2])
    t_npt_on_xy = plane_rotmat.T.dot(t_npt-plane_center)
    projected_t_npt_z_on_xy = surface.get_zdata(np.array([t_npt_on_xy[:2]]))
    projected_t_npt_on_xy = np.array([t_npt_on_xy[0], t_npt_on_xy[1], projected_t_npt_z_on_xy[0]])
    projected_point = plane_rotmat.dot(projected_t_npt_on_xy)+plane_center
    surface_gm = surface.get_gometricmodel([[-.05,.05],[-.05,.05]], rgba=[.5,.7,1,.2])
    surface_gm.set_pos(plane_center)
    surface_gm.set_rotmat(plane_rotmat)
    surface_gm.attach_to(base)
    # homomat = np.eye(4)
    # homomat[:3,:3]=plane_rotmat
    # homomat[:3,3]=plane_center
    # # if tick == 5:
    # gm.gen_box(np.array([.1, .1, .001]), homomat=homomat, rgba=[.5,.7,1,.1]).attach_to(base)
    # projected_point = rm.project_to_plane(t_npt, plane_center, plane_normal)
    # gm.gen_stick(t_npt, projected_point, thickness=.002).attach_to(base)
    new_normal = rm.unit_vector(t_npt-projected_point)
    if pn_direction.dot(new_normal) > .1:
        new_normal = -new_normal
    gm.gen_arrow(spos=projected_point, epos=projected_point+new_normal*.025, thickness=0.001).attach_to(base)
    angle = rm.angle_between_vectors(last_normal, new_normal)
    vec = rm.unit_vector(np.cross(last_normal, new_normal))
    new_rotmat = rm.rotmat_from_axangle(vec, angle)
    # direction = new_rotmat.dot(direction)
    direction = new_rotmat.dot(tmp_direction)
    # new_line_segs = [[cpt, cpt+direction*(.05-tick*.05/n)],
    #                  [cpt+direction*(.05-tick*.05/n), cpt+direction*(.05-tick*.05/n)+tmp_direction*.05]]
    # gm.gen_linesegs(new_line_segs).attach_to(base)
    gm.gen_stick(spos=cpt, epos=projected_point, rgba=[1,.6,0,1], thickness=.002, type='round').attach_to(base)
    cpt=projected_point
    last_normal = new_normal
    # break
#
# t_cpt = cpt
# direction = new_rotmat.dot(-pt_direction)
# for tick in range(1, n+1):
#     t_npt = cpt+direction*.05/n
#     # gm.gen_arrow(spos=cpt, epos=t_npt, thickness=0.001, rgba=[0,1,1,1]).attach_to(base)
#     # gm.gen_arrow(spos=t_npt, epos=t_npt+last_normal*.015, thickness=0.001, rgba=[1,1,0,1]).attach_to(base)
#     nearby_sample_ids = tree.query_ball_point(t_npt, .0015)
#     nearby_samples = bowl_samples[nearby_sample_ids]
#     # gm.GeometricModel(nearby_samples).attach_to(base)
#     plane_center, plane_normal = rm.fit_plane(nearby_samples)
#     plane_tangential = rm.orthogonal_vector(plane_normal)
#     plane_tmp = np.cross(plane_normal, plane_tangential)
#     plane_rotmat = np.column_stack((plane_tangential, plane_tmp, plane_normal))
#     homomat = np.eye(4)
#     homomat[:3,:3]=plane_rotmat
#     homomat[:3,3]=plane_center
#     # twod_plane = gm.gen_box(np.array([.2, .2, .001]), homomat=homomat, rgba=[.5,.7,1,.1]).attach_to(base)
#     projected_point = rm.project_to_plane(t_npt, plane_center, plane_normal)
#     # gm.gen_stick(t_npt, projected_point, thickness=.002).attach_to(base)
#     new_normal = rm.unit_vector(t_npt-projected_point)
#     if pn_direction.dot(new_normal) > .1:
#         new_normal = -new_normal
#     # gm.gen_arrow(spos=projected_point, epos=projected_point+new_normal*.015, thickness=0.001).attach_to(base)
#     angle = rm.angle_between_vectors(last_normal, new_normal)
#     vec = rm.unit_vector(np.cross(last_normal, new_normal))
#     new_rotmat = rm.rotmat_from_axangle(vec, angle)
#     # direction = new_rotmat.dot(direction)
#     direction = new_rotmat.dot(-pt_direction)
#     # new_line_segs = [[cpt, cpt+direction*(.05-tick*.05/n)],
#     #                  [cpt+direction*(.05-tick*.05/n), cpt+direction*(.05-tick*.05/n)+tmp_direction*.05]]
#     # gm.gen_linesegs(new_line_segs).attach_to(base)
#     gm.gen_stick(spos=cpt, epos=projected_point, rgba=[1,.6,0,1], thickness=.002, type='round').attach_to(base)
#     cpt=projected_point
#     last_normal = new_normal
#     # if tick ==2:
#     #     break
#
# t_cpt = cpt
# direction = new_rotmat.dot(-tmp_direction)
# for tick in range(1, n+1):
#     t_npt = cpt+direction*.05/n
#     # gm.gen_arrow(spos=t_npt, epos=t_npt+last_normal*.015, thickness=0.001, rgba=[1, 1, 0, 1]).attach_to(base)
#     nearby_sample_ids = tree.query_ball_point(t_npt, .0015)
#     nearby_samples = bowl_samples[nearby_sample_ids]
#     # gm.GeometricModel(nearby_samples).attach_to(base)
#     plane_center, plane_normal = rm.fit_plane(nearby_samples)
#     plane_tangential = rm.orthogonal_vector(plane_normal)
#     plane_tmp = np.cross(plane_normal, plane_tangential)
#     plane_rotmat = np.column_stack((plane_tangential, plane_tmp, plane_normal))
#     homomat = np.eye(4)
#     homomat[:3,:3]=plane_rotmat
#     homomat[:3,3]=plane_center
#     # twod_plane = gm.gen_box(np.array([.2, .2, .001]), homomat=homomat, rgba=[.5,.7,1,.3]).attach_to(base)
#     projected_point = rm.project_to_plane(t_npt, plane_center, plane_normal)
#     # gm.gen_stick(t_npt, projected_point, thickness=.002).attach_to(base)
#     new_normal = rm.unit_vector(t_npt-projected_point)
#     if pn_direction.dot(new_normal) > .1:
#         new_normal = -new_normal
#     # gm.gen_arrow(spos=projected_point, epos=projected_point+new_normal*.015, thickness=0.001).attach_to(base)
#     angle = rm.angle_between_vectors(last_normal, new_normal)
#     vec = rm.unit_vector(np.cross(last_normal, new_normal))
#     new_rotmat = rm.rotmat_from_axangle(vec, angle)
#     # direction = new_rotmat.dot(direction)
#     direction = new_rotmat.dot(-tmp_direction)
#     # new_line_segs = [[cpt, cpt+direction*(.05-tick*.05/n)],
#     #                  [cpt+direction*(.05-tick*.05/n), cpt+direction*(.05-tick*.05/n)+tmp_direction*.05]]
#     # gm.gen_linesegs(new_line_segs).attach_to(base)
#     gm.gen_stick(spos=cpt, epos=projected_point, rgba=[1,.6,0,1], thickness=.002, type='round').attach_to(base)
#     cpt=projected_point
#     last_normal = new_normal
#     # break


base.run()
