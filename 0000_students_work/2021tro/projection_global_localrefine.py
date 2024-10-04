import numpy as np
import wrs.visualization.panda.world as wd
from wrs import basis as rm, vision as rbfs, modeling as cm
import math
from scipy.spatial import cKDTree

base = wd.World(cam_pos=np.array([-.3,-.7,.42]), lookat_pos=np.array([0,0,0]))
# mgm.gen_frame().attach_to(base)
bowl_model = cm.CollisionModel(initor="./objects/bowl.stl")
bowl_model.set_rgba([.3,.3,.3,.3])
bowl_model.set_rotmat(rm.rotmat_from_euler(math.pi,0,0))
# bowl_model.attach_to(base)

pn_direction = np.array([0, 0, -1])

bowl_samples, bowl_sample_normals = bowl_model.sample_surface(toggle_option='normals', radius=.002)
selection = bowl_sample_normals.dot(-pn_direction)>.1
bowl_samples = bowl_samples[selection]
bowl_sample_normals=bowl_sample_normals[selection]
tree = cKDTree(bowl_samples)
surface = rbfs.RBFSurface(bowl_samples[:, :2], bowl_samples[:,2])
surface.get_gometricmodel(rgba=[.3,.3,.3,.3]).attach_to(base)

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
# mgm.gen_linesegs(line_segs).attach_to(base)
for sec in line_segs:
    gm.gen_stickmodel(spos=sec[0], epos=sec[1], rgba=[0, 0, 0, 1], radius=.002, type='round').attach_to(base)
epos = (line_segs[0][1]-line_segs[0][0])*.7+line_segs[0][0]
gm.gen_arrow(spos=line_segs[0][0], epos=epos, stick_radius=0.004).attach_to(base)
spt = homomat[:3,3]
# mgm.gen_stick(spt, spt + pn_direction * 10, rgba=[0,1,0,1]).attach_to(base)
# base.run()
gm.gen_dashed_arrow(spt, spt - pn_direction * .07, stick_radius=.004).attach_to(base) # p0
cpt, cnrml = bowl_model.ray_hit(spt, spt + pn_direction * 10000, option='closest')
gm.gen_dashed_stick(spt, cpt, rgba=[.57, .57, .57, .7], radius=0.003).attach_to(base)
gm.gen_sphere(pos=cpt, radius=.005).attach_to(base)
gm.gen_dashed_arrow(cpt, cpt - pn_direction * .07, stick_radius=.004).attach_to(base) # p0
gm.gen_dashed_arrow(cpt, cpt + cnrml * .07, stick_radius=.004).attach_to(base) # p0

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
# mgm.gen_linesegs(new_line_segs).attach_to(base)
for sec in new_line_segs:
    gm.gen_stickmodel(spos=sec[0], epos=sec[1], rgba=[0, 0, 0, 1], radius=.002, type='round').attach_to(base)
epos = (new_line_segs[0][1]-new_line_segs[0][0])*.7+new_line_segs[0][0]
gm.gen_arrow(spos=new_line_segs[0][0], epos=epos, stick_radius=0.004).attach_to(base)

last_normal = cnrml
direction = rotmat.dot(pt_direction)
n=3
for tick in range(1, n+1):
    len = .05/n
    tmp_cpt = cpt
    extended_len = 0
    for p in np.linspace(0, len, 1000):
        tmp_t_npt = cpt+direction*p
        tmp_z_surface = surface.get_zdata(np.array([tmp_t_npt[:2]]))
        tmp_projected_point = np.array([tmp_t_npt[0], tmp_t_npt[1], tmp_z_surface[0]])
        tmp_len = np.linalg.norm(tmp_projected_point - tmp_cpt)
        extended_len += tmp_len
        tmp_cpt = tmp_projected_point
        print(tick, extended_len, len)
        if extended_len>len:
            break
    projected_point = tmp_projected_point
    t_npt = tmp_t_npt
    domain_grid = np.meshgrid(np.linspace(-.005, .005, 100, endpoint=True),
                              np.linspace(-.005, .005, 100, endpoint=True))
    domain_0, domain_1 = domain_grid
    domain = np.column_stack((domain_0.ravel()+t_npt[0], domain_1.ravel()+t_npt[1]))
    codomain = surface.get_zdata(domain)
    vertices = np.column_stack((domain, codomain))
    plane_center, plane_normal = rm.fit_plane(vertices)
    new_normal = plane_normal
    if pn_direction.dot(new_normal) > .1:
        new_normal = -new_normal
    angle = rm.angle_between_vectors(last_normal, new_normal)
    vec = rm.unit_vector(np.cross(last_normal, new_normal))
    new_rotmat = rm.rotmat_from_axangle(vec, angle)
    direction = new_rotmat.dot(direction)
    gm.gen_stickmodel(spos=cpt, epos=projected_point, rgba=[1, .6, 0, 1], radius=.002, type='round').attach_to(base)
    cpt=projected_point
    last_normal = new_normal

direction = new_rotmat.dot(tmp_direction)
for tick in range(1, n+1):
    len = .05/n
    tmp_cpt = cpt
    extended_len = 0
    for p in np.linspace(0, len, 1000):
        tmp_t_npt = cpt+direction*p
        tmp_z_surface = surface.get_zdata(np.array([tmp_t_npt[:2]]))
        tmp_projected_point = np.array([tmp_t_npt[0], tmp_t_npt[1], tmp_z_surface[0]])
        tmp_len = np.linalg.norm(tmp_projected_point - tmp_cpt)
        extended_len += tmp_len
        tmp_cpt = tmp_projected_point
        print(tick, extended_len, len)
        if extended_len>len:
            break
    projected_point = tmp_projected_point
    t_npt = tmp_t_npt
    domain_grid = np.meshgrid(np.linspace(-.005, .005, 100, endpoint=True),
                              np.linspace(-.005, .005, 100, endpoint=True))
    domain_0, domain_1 = domain_grid
    domain = np.column_stack((domain_0.ravel()+t_npt[0], domain_1.ravel()+t_npt[1]))
    codomain = surface.get_zdata(domain)
    vertices = np.column_stack((domain, codomain))
    plane_center, plane_normal = rm.fit_plane(vertices)
    new_normal = plane_normal
    if pn_direction.dot(new_normal) > .1:
        new_normal = -new_normal
    angle = rm.angle_between_vectors(last_normal, new_normal)
    vec = rm.unit_vector(np.cross(last_normal, new_normal))
    new_rotmat = rm.rotmat_from_axangle(vec, angle)
    direction = new_rotmat.dot(tmp_direction)
    gm.gen_stickmodel(spos=cpt, epos=projected_point, rgba=[1, .6, 0, 1], radius=.002, type='round').attach_to(base)
    cpt=projected_point
    last_normal = new_normal

direction = new_rotmat.dot(-pt_direction)
for tick in range(1, n+1):
    len = .05/n
    tmp_cpt = cpt
    extended_len = 0
    for p in np.linspace(0, len, 1000):
        tmp_t_npt = cpt+direction*p
        tmp_z_surface = surface.get_zdata(np.array([tmp_t_npt[:2]]))
        tmp_projected_point = np.array([tmp_t_npt[0], tmp_t_npt[1], tmp_z_surface[0]])
        tmp_len = np.linalg.norm(tmp_projected_point - tmp_cpt)
        extended_len += tmp_len
        tmp_cpt = tmp_projected_point
        print(tick, extended_len, len)
        if extended_len>len:
            break
    projected_point = tmp_projected_point
    t_npt = tmp_t_npt
    domain_grid = np.meshgrid(np.linspace(-.005, .005, 100, endpoint=True),
                              np.linspace(-.005, .005, 100, endpoint=True))
    domain_0, domain_1 = domain_grid
    domain = np.column_stack((domain_0.ravel()+t_npt[0], domain_1.ravel()+t_npt[1]))
    codomain = surface.get_zdata(domain)
    vertices = np.column_stack((domain, codomain))
    plane_center, plane_normal = rm.fit_plane(vertices)
    new_normal = plane_normal
    if pn_direction.dot(new_normal) > .1:
        new_normal = -new_normal
    angle = rm.angle_between_vectors(last_normal, new_normal)
    vec = rm.unit_vector(np.cross(last_normal, new_normal))
    new_rotmat = rm.rotmat_from_axangle(vec, angle)
    direction = new_rotmat.dot(-pt_direction)
    gm.gen_stickmodel(spos=cpt, epos=projected_point, rgba=[1, .6, 0, 1], radius=.002, type='round').attach_to(base)
    cpt=projected_point
    last_normal = new_normal

direction = new_rotmat.dot(-tmp_direction)
for tick in range(1, n+1):
    len = .05/n
    tmp_cpt = cpt
    extended_len = 0
    for p in np.linspace(0, len, 1000):
        tmp_t_npt = cpt+direction*p
        tmp_z_surface = surface.get_zdata(np.array([tmp_t_npt[:2]]))
        tmp_projected_point = np.array([tmp_t_npt[0], tmp_t_npt[1], tmp_z_surface[0]])
        tmp_len = np.linalg.norm(tmp_projected_point - tmp_cpt)
        extended_len += tmp_len
        tmp_cpt = tmp_projected_point
        print(tick, extended_len, len)
        if extended_len>len:
            break
    projected_point = tmp_projected_point
    t_npt = tmp_t_npt
    domain_grid = np.meshgrid(np.linspace(-.005, .005, 100, endpoint=True),
                              np.linspace(-.005, .005, 100, endpoint=True))
    domain_0, domain_1 = domain_grid
    domain = np.column_stack((domain_0.ravel()+t_npt[0], domain_1.ravel()+t_npt[1]))
    codomain = surface.get_zdata(domain)
    vertices = np.column_stack((domain, codomain))
    plane_center, plane_normal = rm.fit_plane(vertices)
    new_normal = plane_normal
    if pn_direction.dot(new_normal) > .1:
        new_normal = -new_normal
    angle = rm.angle_between_vectors(last_normal, new_normal)
    vec = rm.unit_vector(np.cross(last_normal, new_normal))
    new_rotmat = rm.rotmat_from_axangle(vec, angle)
    direction = new_rotmat.dot(-tmp_direction)
    gm.gen_stickmodel(spos=cpt, epos=projected_point, rgba=[1, .6, 0, 1], radius=.002, type='round').attach_to(base)
    cpt=projected_point
    last_normal = new_normal

base.run()
