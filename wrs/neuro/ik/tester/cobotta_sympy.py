import sympy
from sympy import Matrix
from sympy.utilities import lambdify
import numpy as np
from wrs import robot_sim as cbt_s, modeling as gm
import wrs.visualization.panda.world as world
import time

rbt_s = cbt_s.Cobotta()
x, y, z, r00, r01, r02, r10, r11, r12, r20, r21, r22 = sympy.symbols("x y z r00 r01 r02 r10 r11 r12 r20 r21 r22")
diff = -(rbt_s.manipulator_dict['arm'].jnts[6]['loc_pos'][1] + rbt_s.manipulator_dict['arm']._loc_flange_pos[1])
Matrix([x,y,z])+Matrix([r02, r12, r22])*diff
Matrix([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])*diff

Matrix([x,y,z])-


def sym_axangle(ax, angle):
    st = sympy.sin(angle)
    ct = sympy.cos(angle)
    # ax = ax.T[0, :]
    _rotmat = [
        [ct + ax[0] ** 2 * (1 - ct), ax[0] * ax[1] * (1 - ct) - ax[2] * st, ax[2] * ax[0] * (1 - ct) + ax[1] * st],
        [ax[0] * ax[1] * (1 - ct) + ax[2] * st, ct + ax[1] ** 2 * (1 - ct), ax[1] * ax[2] * (1 - ct) - ax[0] * st],
        [ax[2] * ax[0] * (1 - ct) - ax[1] * st, ax[1] * ax[2] * (1 - ct) + ax[0] * st, ct + ax[2] ** 2 * (1 - ct)]]
    return sympy.Matrix(_rotmat)


base = world.World(cam_pos=np.array([1.5, 1, .7]))
gm.gen_frame(alpha=.3).attach_to(base)

q1, q2, q3, q4, q5, q6 = sympy.symbols("q1 q2 q3 q4 q5 q6")
x, y, z, r00, r01, r02, r10, r11, r12, r20, r21, r22 = sympy.symbols("x y z r00 r01 r02 r10 r11 r12 r20 r21 r22")

rbt_s = cbt_s.Cobotta()

print("Computing joint 1...")
# rotmat
rotmat0 = Matrix(rbt_s.manipulator_dict['arm'].jnts[0]['gl_rotmatq'])
ax1 = rotmat0 * Matrix(rbt_s.manipulator_dict['arm'].jnts[1]['loc_motionax'])
rotmat1 = sym_axangle(ax1, q1)
accumulated_rotmat1 = rotmat1
# pos
pos0 = Matrix(rbt_s.manipulator_dict['arm'].jnts[0]['gl_posq'])
pos1 = pos0 + rotmat0 * Matrix(rbt_s.manipulator_dict['arm'].jnts[1]['loc_pos'])

print("Computing joint 2...")
# rotmat
ax2 = accumulated_rotmat1 * Matrix(rbt_s.manipulator_dict['arm'].jnts[2]['loc_motionax'])
rotmat2 = sym_axangle(ax2, q2)
accumulated_rotmat2 = rotmat2 * accumulated_rotmat1
# pos
pos2 = pos1 + accumulated_rotmat1 * Matrix(rbt_s.manipulator_dict['arm'].jnts[2]['loc_pos'])

print("Computing joint 3...")
# rotmat
ax3 = accumulated_rotmat2 * Matrix(rbt_s.manipulator_dict['arm'].jnts[3]['loc_motionax'])
rotmat3 = sym_axangle(ax3, q3)
accumulated_rotmat3 = rotmat3 * accumulated_rotmat2
# pos
pos3 = pos2 + accumulated_rotmat2 * Matrix(rbt_s.manipulator_dict['arm'].jnts[3]['loc_pos'])

print("Computing joint 4...")
# rotmat
ax4 = accumulated_rotmat3 * Matrix(rbt_s.manipulator_dict['arm'].jnts[4]['loc_motionax'])
rotmat4 = sym_axangle(ax4, q4)
accumulated_rotmat4 = rotmat4 * accumulated_rotmat3
# pos
pos4 = pos3 + accumulated_rotmat3 * Matrix(rbt_s.manipulator_dict['arm'].jnts[4]['loc_pos'])

print("Computing joint 5...")
# rotmat
ax5 = accumulated_rotmat4 * Matrix(rbt_s.manipulator_dict['arm'].jnts[5]['loc_motionax'])
rotmat5 = sym_axangle(ax5, q5)
accumulated_rotmat5 = rotmat5 * accumulated_rotmat4
# pos
pos5 = pos4 + accumulated_rotmat4 * Matrix(rbt_s.manipulator_dict['arm'].jnts[5]['loc_pos'])

print("Computing joint 6...")
# rotmat
ax6 = accumulated_rotmat5 * Matrix(rbt_s.manipulator_dict['arm'].jnts[6]['loc_motionax'])
rotmat6 = sym_axangle(ax6, q6)
accumulated_rotmat6 = rotmat6 * accumulated_rotmat5
# pos
pos6 = pos5 + accumulated_rotmat5 * Matrix(rbt_s.manipulator_dict['arm'].jnts[6]['loc_pos'])

total_pos = pos6
total_rotmat = accumulated_rotmat6

# eq0 = total_rotmat[0, 0] - r00
# eq1 = total_rotmat[0, 1] - r01
# eq2 = total_rotmat[0, 2] - r02
# eq3 = total_rotmat[1, 0] - r10
# eq4 = total_rotmat[1, 1] - r11
# eq5 = total_rotmat[1, 2] - r12
# eq6 = total_rotmat[2, 0] - r20
# eq7 = total_rotmat[2, 1] - r21
# eq8 = total_rotmat[2, 2] - r22
# eq9 = pos6[0]-x
# eq10 = pos6[1]-y
# eq11 = pos6[2]-z
# print("Solving equation...")
# values = sympy.solve([eq0, eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11], [q1, q2, q3, q4, q5, q6])
# print(values)
# base.run()
# pickle.dump(total_rotmat, "cbt_fk.pkl")
# base.run()
#
# sympy_total_rotmat = pickle.load("cbt_fk.pkl")

fk = lambdify((q1, q2, q3, q4, q5, q6), [total_pos, total_rotmat], 'numpy')
jnt_values=[.5]*6
tic = time.time_ns()
print(fk(*jnt_values))
toc = time.time_ns()
print(f"sympy eval cost {toc - tic}")

base.run()

jnt_values = np.array([.5] * 6)
tic = time.time_ns()
total_pos = total_pos.subs([(q1, jnt_values[0]),
                            (q2, jnt_values[1]),
                            (q3, jnt_values[2]),
                            (q4, jnt_values[3]),
                            (q5, jnt_values[4]),
                            (q6, jnt_values[5])])
resultant_pos = np.array(total_pos.tolist()).ravel().astype(np.float64)
resultant_rotmat = total_rotmat.subs([(q1, jnt_values[0]),
                                      (q2, jnt_values[1]),
                                      (q3, jnt_values[2]),
                                      (q4, jnt_values[3]),
                                      (q5, jnt_values[4]),
                                      (q6, jnt_values[5])])
resultant_rotmat = np.array(resultant_rotmat.tolist()).astype(np.float64)
toc = time.time_ns()
print(f"sympy eval cost {toc - tic}")
tic = time.time_ns()
rbt_s.fk(jnt_values=jnt_values)
actual_pos, actual_rotmat = rbt_s.get_gl_tcp(manipulator_name='arm')
toc = time.time_ns()
print(f"direct fk cost {toc - tic}")
#
# print(resultant_rotmat, actual_rotmat)
print(resultant_pos, resultant_rotmat)
gm.gen_frame(pos=resultant_pos, rotmat=resultant_rotmat).attach_to(base)
gm.gen_myc_frame(pos=actual_pos, rotmat=actual_rotmat).attach_to(base)

rbt_s.gen_meshmodel(toggle_tcp_frame=True, toggle_jnt_frames=True, rgba=[.3, .3, .3, .3]).attach_to(base)

pos2 = pos2.subs([(q1, jnt_values[0]),
                  (q2, jnt_values[1])])
resultant_pos2 = np.array(pos2.tolist()).ravel().astype(np.float64)
gm.gen_sphere(pos=resultant_pos2).attach_to(base)

pos3 = pos3.subs([(q1, jnt_values[0]),
                  (q2, jnt_values[1]),
                  (q3, jnt_values[2])])
resultant_pos3 = np.array(pos3.tolist()).ravel().astype(np.float64)
gm.gen_sphere(pos=resultant_pos3).attach_to(base)

pos4 = pos4.subs([(q1, jnt_values[0]),
                  (q2, jnt_values[1]),
                  (q3, jnt_values[2]),
                  (q4, jnt_values[2])])
resultant_pos4 = np.array(pos4.tolist()).ravel().astype(np.float64)
gm.gen_sphere(pos=resultant_pos4).attach_to(base)
base.run()

# sympy_resultant_rotmat = sympy_rotmat1.subs([(q1, jnt_values[0])])
# resultant_rotmat = np.array(sympy_resultant_rotmat.tolist()).astype(np.float64)
# mgm.gen_mycframe(pos=rbt_s.manipulator_dict['arm'].joints[1]['gl_posq'], rotmat=resultant_rotmat).attach_to(base)
#
# sympy_resultant_rotmat = (sympy_rotmat2 * sympy_rotmat1).subs([(q1, jnt_values[0]), (q2, jnt_values[1])])
# resultant_rotmat = np.array(sympy_resultant_rotmat.tolist()).astype(np.float64)
# mgm.gen_mycframe(pos=rbt_s.manipulator_dict['arm'].joints[2]['gl_posq'], rotmat=resultant_rotmat).attach_to(base)
#
# sympy_resultant_rotmat = (sympy_rotmat3 * sympy_rotmat2 * sympy_rotmat1).subs(
#     [(q1, jnt_values[0]), (q2, jnt_values[1]), (q3, jnt_values[3])])
# resultant_rotmat = np.array(sympy_resultant_rotmat.tolist()).astype(np.float64)
# mgm.gen_mycframe(pos=rbt_s.manipulator_dict['arm'].joints[3]['gl_posq'], rotmat=resultant_rotmat).attach_to(base)
# base.run()
