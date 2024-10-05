from wrs import wd, rm, mgm

option = 'a'

if option == 'a':
    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0], toggle_debug=True)
    rotmat = rm.rotmat_from_euler(rm.pi / 3, -rm.pi / 6, rm.pi / 3)
    ax, angle = rm.axangle_between_rotmat(rm.np.eye(3), rotmat)
    rotmat_start = rm.rotmat_from_axangle(ax, -rm.pi / 6)
    cross_vec = rotmat_start.dot(rotmat[:, 0])
    rotmat_goal = rm.rotmat_from_axangle(ax, rm.pi / 3)
    mgm.gen_arrow(epos=ax * .3, rgb=rm.const.gray, stick_radius=.0025, alpha=.9).attach_to(base)
    mgm.gen_frame(ax_length=.2).attach_to(base)
    mgm.gen_arrow(epos=cross_vec * .3, rgb=rm.const.yellow, alpha=1).attach_to(base)
    nxt_vec_uvw = rotmat_goal.dot(cross_vec)
    mgm.gen_dashed_arrow(epos=nxt_vec_uvw * .3, rgb=rm.const.yellow, alpha=1).attach_to(base)
    radius, _ = rm.unit_vector(cross_vec * .3 - cross_vec.dot(ax) * ax * .3, toggle_length=True)
    mgm.gen_arrow(spos=cross_vec.dot(ax) * ax * .3, epos=cross_vec * .3, rgb=rm.const.orange, alpha=1).attach_to(base)
    mgm.gen_dashed_arrow(spos=cross_vec.dot(ax) * ax * .3, epos=nxt_vec_uvw * .3, rgb=rm.const.orange,
                         alpha=1).attach_to(base)
    mgm.gen_arrow(epos=ax * np.sqrt(.3 ** 2 - radius ** 2), stick_radius=.003, rgb=rm.const.black, alpha=1).attach_to(base)
    mgm.gen_sphere(radius=.005, pos=ax * np.sqrt(.3 ** 2 - radius ** 2), rgb=rm.const.black, alpha=1).attach_to(base)
    mgm.gen_sphere(radius=.005, pos=cross_vec * .3, rgb=rm.const.black, alpha=1).attach_to(base)
    mgm.gen_sphere(radius=.005, pos=nxt_vec_uvw * .3, rgb=rm.const.black, alpha=1).attach_to(base)
    # rectangle
    epos_vec = rm.unit_vector(cross_vec * .3 - cross_vec.dot(ax) * ax * .3)
    mgm.gen_stick(spos=cross_vec.dot(ax) * ax * .3 - ax * .03,
                  epos=cross_vec.dot(ax) * ax * .3 - ax * .03 + epos_vec * .03,
                  rgb=rm.const.black, alpha=1,
                  radius=.001).attach_to(base)
    mgm.gen_stick(spos=cross_vec.dot(ax) * ax * .3 - ax * .03 + epos_vec * .03,
                  epos=cross_vec.dot(ax) * ax * .3 - ax * .03 + epos_vec * .03 + ax * .03 - ax * .005,
                  rgb=rm.const.black, alpha=1,
                  radius=.001).attach_to(base)
    epos_vec = rm.unit_vector(nxt_vec_uvw * .3 - nxt_vec_uvw.dot(ax) * ax * .3)
    mgm.gen_stick(spos=nxt_vec_uvw.dot(ax) * ax * .3 - ax * .03,
                  epos=nxt_vec_uvw.dot(ax) * ax * .3 - ax * .03 + epos_vec * .03,
                  rgb=rm.const.black, alpha=1,
                  radius=.001).attach_to(base)
    mgm.gen_stick(spos=nxt_vec_uvw.dot(ax) * ax * .3 - ax * .03 + epos_vec * .03,
                  epos=nxt_vec_uvw.dot(ax) * ax * .3 - ax * .03 + epos_vec * .03 + ax * .03 - ax * .0045,
                  rgb=rm.const.black, alpha=1,
                  radius=.001).attach_to(base)
    # rot traj
    mgm.gen_torus(ax,
                  starting_vector=nxt_vec_uvw * .3 - nxt_vec_uvw.dot(ax) * ax * .3,
                  portion=5 / 6,
                  center=cross_vec.dot(ax) * ax * .3,
                  major_radius=radius,
                  minor_radius=.0015,
                  n_sec_major=64,
                  rgb=rm.const.white, alpha=1).attach_to(base)
    mgm.gen_circarrow(ax,
                      starting_vector=cross_vec * .3 - cross_vec.dot(ax) * ax * .3,
                      portion=1 / 6,
                      center=cross_vec.dot(ax) * ax * .3,
                      major_radius=radius,
                      n_sec_major=64,
                      minor_radius=.0015,
                      rgb=rm.const.orange, alpha=1).attach_to(base)
if option == 'b':
    rotmat = rm.rotmat_from_euler(rm.pi / 3, -rm.pi / 6, rm.pi / 3)
    ax, angle = rm.axangle_between_rotmat(np.eye(3), rotmat)
    base = wd.World(cam_pos=ax * 2, lookat_pos=[0, 0, 0])
    rotmat_start = rm.rotmat_from_axangle(ax, -rm.pi / 6)
    cross_vec = rotmat_start.dot(rotmat[:, 0])
    rotmat_goal = rm.rotmat_from_axangle(ax, rm.pi / 3)
    nxt_vec_uvw = rotmat_goal.dot(cross_vec)
    radius, _ = rm.unit_vector(cross_vec * .3 - cross_vec.dot(ax) * ax * .3, toggle_length=True)
    mgm.gen_arrow(spos=cross_vec.dot(ax) * ax * .3, epos=cross_vec * .3, rgb=rm.const.orange, alpha=1).attach_to(base)
    mgm.gen_dashed_arrow(spos=cross_vec.dot(ax) * ax * .3, epos=nxt_vec_uvw * .3, rgb=rm.const.orange,
                         alpha=1).attach_to(base)
    mgm.gen_torus(ax,
                  starting_vector=nxt_vec_uvw * .3 - nxt_vec_uvw.dot(ax) * ax * .3,
                  portion=5 / 6,
                  center=cross_vec.dot(ax) * ax * .3,
                  major_radius=radius,
                  minor_radius=.0015,
                  n_sec_major=64,
                  rgb=rm.const.white, alpha=1).attach_to(base)
    mgm.gen_circarrow(ax,
                      starting_vector=cross_vec * .3 - cross_vec.dot(ax) * ax * .3,
                      portion=1 / 6,
                      center=cross_vec.dot(ax) * ax * .3,
                      major_radius=radius,
                      n_sec_major=64,
                      minor_radius=.0015,
                      rgb=rm.const.orange, alpha=1).attach_to(base)
    mgm.gen_sphere(radius=.005, pos=ax * np.sqrt(.3 ** 2 - radius ** 2), rgb=rm.const.black, alpha=1).attach_to(base)
    mgm.gen_sphere(radius=.005, pos=cross_vec * .3, rgb=rm.const.black, alpha=1).attach_to(base)
    mgm.gen_sphere(radius=.005, pos=nxt_vec_uvw * .3, rgb=rm.const.black, alpha=1).attach_to(base)
if option == 'c':
    rotmat = rm.rotmat_from_euler(rm.pi / 3, -rm.pi / 6, rm.pi / 3)
    ax, angle = rm.axangle_between_rotmat(np.eye(3), rotmat)
    base = wd.World(cam_pos=ax * 2, lookat_pos=[0, 0, 0])
    rotmat_start = rm.rotmat_from_axangle(ax, -rm.pi / 6)
    cross_vec = rotmat_start.dot(rotmat[:, 0])
    rotmat_goal = rm.rotmat_from_axangle(ax, rm.pi / 3)
    nxt_vec_uvw = rotmat_goal.dot(cross_vec)
    radius, _ = rm.unit_vector(cross_vec * .3 - cross_vec.dot(ax) * ax * .3, toggle_length=True)
    mgm.gen_dashed_arrow(spos=cross_vec.dot(ax) * ax * .3, epos=nxt_vec_uvw * .3, rgb=rm.const.orange,
                         alpha=1).attach_to(base)
    mgm.gen_torus(ax,
                  starting_vector=nxt_vec_uvw * .3 - nxt_vec_uvw.dot(ax) * ax * .3,
                  portion=5 / 6,
                  center=cross_vec.dot(ax) * ax * .3,
                  major_radius=radius,
                  minor_radius=.0015,
                  n_sec_major=64,
                  rgb=rm.const.white, alpha=1).attach_to(base)
    mgm.gen_circarrow(ax,
                      starting_vector=cross_vec * .3 - cross_vec.dot(ax) * ax * .3,
                      portion=1 / 6,
                      center=cross_vec.dot(ax) * ax * .3,
                      major_radius=radius,
                      n_sec_major=64,
                      minor_radius=.0015,
                      rgb=rm.const.orange, alpha=1).attach_to(base)
    mgm.gen_sphere(radius=.005, pos=ax * np.sqrt(.3 ** 2 - radius ** 2), rgb=rm.const.black, alpha=1).attach_to(base)
    mgm.gen_sphere(radius=.005, pos=cross_vec * .3, rgb=rm.const.black, alpha=1).attach_to(base)
    mgm.gen_sphere(radius=.005, pos=nxt_vec_uvw * .3, rgb=rm.const.black, alpha=1).attach_to(base)
    # projections
    mgm.gen_dashed_arrow(spos=ax * np.sqrt(.3 ** 2 - radius ** 2),
                         epos=ax * np.sqrt(.3 ** 2 - radius ** 2) + np.cross(ax, cross_vec * .3) * rm.sin(rm.pi / 3),
                         rgb=np.array([1, 0, 1]), alpha=.5).attach_to(base)
    mgm.gen_dashed_arrow(spos=ax * np.sqrt(.3 ** 2 - radius ** 2),
                         epos=ax * np.sqrt(.3 ** 2 - radius ** 2) + (
                                 cross_vec * .3 - cross_vec.dot(ax) * ax * .3) * rm.cos(rm.pi / 3),
                         rgb=np.array([0, 1, 1]), alpha=.5).attach_to(base)
    # lines
    mgm.gen_dashed_stick(spos=nxt_vec_uvw * .3,
                         epos=ax * np.sqrt(.3 ** 2 - radius ** 2) + np.cross(ax, cross_vec * .3) * rm.sin(rm.pi / 3),
                         radius=.001, rgb=rm.const.black, alpha=1, len_solid=.01, len_interval=.007).attach_to(base)
    mgm.gen_dashed_stick(spos=nxt_vec_uvw * .3,
                         epos=ax * np.sqrt(.3 ** 2 - radius ** 2) + (
                                 cross_vec * .3 - cross_vec.dot(ax) * ax * .3) * rm.cos(rm.pi / 3),
                         radius=.001, rgb=rm.const.black, alpha=1, len_solid=.01, len_interval=.007).attach_to(base)
base.run()
