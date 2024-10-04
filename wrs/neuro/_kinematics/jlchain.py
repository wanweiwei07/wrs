import torch
import wrs.neuro._kinematics.math_utils as nkm
from wrs import basis as bc, robot_sim as rkc, robot_sim as rkmg, neuro as nkjl, modeling as mgm


# TODO delay finalize
# TODO joint gl -> flange

class JLChain(object):
    """
    Joint Link Chain, no branches allowed
    Usage:
    1. Create a JLChain instance with a given n_dof and update its parameters for particular definition
    2. Define multiple instances of this class to compose a complicated structure
    3. Use mimic for underactuated or coupled mechanism
    """

    def __init__(self,
                 name="auto",
                 pos=torch.zeros(3),
                 rotmat=torch.eye(3),
                 n_dof=6):
        """
        conf -- configuration: target joint values
        :param name:
        :param pos:
        :param rotmat:
        :param home: number of joints
        :param name:
        """
        self.name = name
        self.n_dof = n_dof
        self.home = torch.zeros(self.n_dof, dtype=torch.float32)  # self.n_dof joints plus one anchor
        # initialize anchor
        self.anchor = nkjl.Anchor(name=f"{name}_anchor", pos=pos, rotmat=rotmat)
        # initialize joints and links
        self.jnts = [nkjl.Joint(name=f"{name}_j{i}") for i in range(self.n_dof)]
        self._jnt_ranges = self._get_jnt_ranges()
        # default flange joint id, loc_xxx are considered described in it
        self._flange_jnt_id = self.n_dof - 1
        # default flange for cascade connection
        self._loc_flange_pos = torch.zeros(3)
        self._loc_flange_rotmat = torch.eye(3)
        self._gl_flange_pos = self._loc_flange_pos
        self._gl_flange_rotmat = self._loc_flange_rotmat
        # finalizing tag
        self._is_finalized = False
        # iksolver
        self._ik_solver = None

    @staticmethod
    def assert_finalize_decorator(method):
        def wrapper(self, *args, **kwargs):
            if self._is_finalized:
                return method(self, *args, **kwargs)
            else:
                raise ValueError("JLChain is not finalized.")

        return wrapper

    @property
    def jnt_ranges(self):
        return self._jnt_ranges

    @property
    def flange_jnt_id(self):
        return self._flange_jnt_id

    @flange_jnt_id.setter
    def flange_jnt_id(self, value):
        self._flange_jnt_id = value

    @property
    def loc_flange_pos(self):
        return self._loc_flange_pos

    @property
    def loc_flange_rotmat(self):
        return self._loc_flange_rotmat

    @property
    def loc_flange_homomat(self):
        return nkm.homomat_from_posrot(pos=self._loc_flange_pos, rotmat=self._loc_flange_rotmat)

    @property
    @assert_finalize_decorator
    def gl_flange_pos(self):
        return self._gl_flange_pos

    @property
    @assert_finalize_decorator
    def gl_flange_rotmat(self):
        return self._gl_flange_rotmat

    @property
    def pos(self):
        return self.anchor.pos

    @property
    def rotmat(self):
        return self.anchor.rotmat

    def _get_jnt_ranges(self):
        """
        get jnt ranges
        :return: tensor with shape (n_dof, 2)
        date: 20180602, 20200704osaka
        author: weiwei
        """
        jnt_limits = []
        for i in range(self.n_dof):
            jnt_limits.append(self.jnts[i].motion_range)
        return torch.stack(jnt_limits)

    def fk(self, jnt_values, update=False):
        """
        :param jnt_values: a 1xn ndarray where each element indicates the value of a joint (in radian or meter)
        :param update if True, update internal values
        :return: True (succ), False (failure)
        author: weiwei
        date: 20161202, 20201009osaka, 20230823
        """
        if not update:
            homomat = self.anchor.gl_flange_homomat_list[0]
            jnt_pos = torch.zeros((self.n_dof, 3))
            jnt_motion_ax = torch.zeros((self.n_dof, 3))
            for i in range(self.flange_jnt_id + 1):
                jnt_pos[i, :] = homomat[:3, 3] + homomat[:3, :3] @ self.jnts[i].loc_pos
                homomat = homomat @ self.jnts[i].get_motion_homomat(motion_value=jnt_values[i])
                jnt_motion_ax[i, :] = homomat[:3, :3] @ self.jnts[i].loc_motion_ax
            gl_flange_homomat = homomat @ self.loc_flange_homomat
            gl_flange_pos = gl_flange_homomat[:3, 3]
            gl_flange_rotmat = gl_flange_homomat[:3, :3]
            return (gl_flange_pos, gl_flange_rotmat)
        else:
            # self.anchor.update_pose()
            pos = self.anchor.gl_flange_pose_list[0][0]
            rotmat = self.anchor.gl_flange_pose_list[0][1]
            for i in range(self.n_dof):
                self.jnts[i].update_globals(pos=pos, rotmat=rotmat, motion_value=jnt_values[i])
                pos = self.jnts[i].gl_pos_q
                rotmat = self.jnts[i].gl_rotmat_q
            self._gl_flange_pos, self._gl_flange_rotmat = self._compute_gl_flange()
            return (self._gl_flange_pos, self._gl_flange_rotmat)

    def fix_to(self, pos, rotmat, jnt_values=None):
        self.anchor.pos = pos
        self.anchor.rotmat = rotmat
        if jnt_values is None:
            return self.goto_given_conf(jnt_values=self.get_jnt_values())
        else:
            return self.goto_given_conf(jnt_values=jnt_values)

    def finalize(self):
        """
        :return: (pos, rotmat)
        author: weiwei
        date: 20201126, 20231111
        """
        self._jnt_ranges = self._get_jnt_ranges()
        gl_flange_pose = self.go_home()
        self._is_finalized = True
        return gl_flange_pose

    def set_flange(self, loc_flange_pos=None, loc_flange_rotmat=None):
        if loc_flange_pos is not None:
            self._loc_flange_pos = loc_flange_pos
        if loc_flange_rotmat is not None:
            self._loc_flange_rotmat = loc_flange_rotmat
        self._is_finalized = False

    def _compute_gl_flange(self):
        if self.n_dof >= 1:
            gl_pos = self.jnts[self.flange_jnt_id].gl_pos_q + self.jnts[
                self.flange_jnt_id].gl_rotmat_q @ self._loc_flange_pos
            gl_rotmat = self.jnts[self.flange_jnt_id].gl_rotmat_q @ self._loc_flange_rotmat
        else:
            pos = self.anchor.gl_flange_pose_list[0][0]
            rotmat = self.anchor.gl_flange_pose_list[0][1]
            gl_pos = pos + rotmat @ self._loc_flange_pos
            gl_rotmat = rotmat @ self._loc_flange_rotmat
        return (gl_pos, gl_rotmat)

    @assert_finalize_decorator
    def cvt_pose_in_flange_to_gl(self, loc_pos=torch.zeros(3), loc_rotmat=torch.eye(3)):
        """
        given a loc pos and rotmat in the flange frame, convert it to global frame
        :param loc_pos: 1x3 tensor in the flange frame
        :param loc_rotmat: 3x3 tensor in the flange frame
        :param
        :return:
        author: weiwei
        date: 202403032
        """
        tmp_gl_pos = self._gl_flange_pos + self._gl_flange_rotmat @ loc_pos
        tmp_gl_rotmat = self._gl_flange_rotmat @ loc_rotmat
        return (tmp_gl_pos, tmp_gl_rotmat)

    @assert_finalize_decorator
    def cvt_gl_to_flange(self, gl_pos, gl_rotmat):
        """
        given a global pos and rotmat, get its relative pos and rotmat to the flange frame
        :param gl_pos: 1x3 tensor
        :param gl_rotmat: 3x3 tensor
        :return:
        author: weiwei
        date: 20190312
        """
        return nkm.rel_pose(self.gl_flange_pos, self.gl_flange_rotmat, gl_pos, gl_rotmat)

    def cvt_pose_in_flange_to_functional(self, pos_in_flange, rotmat_in_flange):
        """
        convert a pose in the flange frame to the functional joint frame
        :param pos_in_flange:
        :param rotmat_in_flange:
        :return:
        author: weiwei
        date: 20240301
        """
        tmp_loc_pos = self._loc_flange_pos + self._loc_flange_rotmat @ pos_in_flange
        tmp_loc_rotmat = self._loc_flange_rotmat @ rotmat_in_flange
        return (tmp_loc_pos, tmp_loc_rotmat)

    def are_jnts_in_ranges(self, jnt_values):
        """
        check if the given jnt_values are in range
        :param jnt_values: tensor
        :return:
        author: weiwei
        date: 20220326toyonaka
        """
        if len(jnt_values) != self.n_dof:
            raise ValueError(f"The given joint values do not match n_dof: {len(jnt_values)} vs. {self.n_dof}")
        below_lower_bound = jnt_values < self.jnt_ranges[:, 0]
        above_upper_bound = jnt_values > self.jnt_ranges[:, 1]
        if torch.any(below_lower_bound) or torch.any(above_upper_bound):
            print("Joints are out of ranges!")
            return False
        else:
            return True

    def goto_given_conf(self, jnt_values):
        """
        move to the given configuration
        :param jnt_values: tensor
        :return: null
        author: weiwei
        date: 20230927osaka
        """
        return self.fk(jnt_values=jnt_values, update=True)

    def go_home(self):
        """
        move to home configuration
        :return: null
        author: weiwei
        date: 20161211osaka
        """
        return self.goto_given_conf(jnt_values=self.home)

    def go_zero(self):
        """
        move to zero configuration
        :return: null
        author: weiwei
        date: 20161211osaka
        """
        return self.goto_given_conf(jnt_values=torch.zeros(self.n_dof, dtype=torch.float32))

    def get_jnt_values(self):
        """
        get the current joint values
        :return: jnt_values: a 1xn ndarray
        author: weiwei
        date: 20161205tsukuba
        """
        jnt_values = torch.zeros(self.n_dof)
        for i in range(self.n_dof):
            jnt_values[i] = self.jnts[i].motion_value
        return jnt_values

    def rand_conf(self):
        """
        generate a random configuration
        author: weiwei
        date: 20200326
        """
        return torch.rand(self.n_dof) * (self.jnt_ranges[:, 1] - self.jnt_ranges[:, 0]) + self.jnt_ranges[:, 0]

    @assert_finalize_decorator
    def ik(self,
           tgt_pos,
           tgt_rotmat,
           seed_jnt_values=None,
           toggle_dbg=False):
        """
        :param tgt_pos: 1x3 tensor
        :param tgt_rotmat: 3x3 tensor
        :param seed_jnt_values: the starting configuration used in the numerical iteration
        :return:
        """
        raise NotImplementedError

    def gen_stickmodel(self,
                       stick_rgba=bc.lnk_stick_rgba,
                       toggle_jnt_frames=False,
                       toggle_flange_frame=True,
                       name='jlc_stick_model',
                       jnt_radius=rkc.JNT_RADIUS,
                       lnk_radius=rkc.LNK_STICK_RADIUS):
        with torch.no_grad():
            m_col = rkmg.gen_jlc_stick(jlc=self,
                                       stick_rgba=stick_rgba,
                                       toggle_jnt_frames=toggle_jnt_frames,
                                       toggle_flange_frame=toggle_flange_frame,
                                       name=name,
                                       jnt_radius=jnt_radius,
                                       lnk_radius=lnk_radius)
        return m_col

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_flange_frame=False,
                      toggle_jnt_frames=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False,
                      name='jlc_mesh_model'):
        with torch.no_grad():
            m_col = rkmg.gen_jlc_mesh(jlc=self,
                                      rgb=rgb,
                                      alpha=alpha,
                                      toggle_flange_frame=toggle_flange_frame,
                                      toggle_jnt_frames=toggle_jnt_frames,
                                      toggle_cdprim=toggle_cdprim,
                                      toggle_cdmesh=toggle_cdmesh,
                                      name=name)
        return m_col


if __name__ == "__main__":
    import time
    from tqdm import tqdm
    import wrs.visualization.panda.world as wd
    import wrs.robot_sim._kinematics.constant as rkc
    from torch.optim import LBFGS

    base = wd.World(cam_pos=[1.25, .75, .75], lookat_pos=[0, 0, .3])
    mgm.gen_frame().attach_to(base)

    jlc = JLChain(n_dof=6)
    jlc.jnts[0].loc_pos = torch.tensor([0.0, 0.0, 0.0])
    jlc.jnts[0].loc_motion_ax = torch.tensor([0.0, 0.0, 1.0])
    jlc.jnts[0].motion_range = torch.tensor([-torch.pi / 2.0, torch.pi / 2.0])
    # jlc.jnts[1].change_type(rkc.JntType.PRISMATIC)
    jlc.jnts[1].loc_pos = torch.tensor([0.0, 0.0, .05])
    jlc.jnts[1].loc_motion_ax = torch.tensor([0.0, 1.0, 0.0])
    jlc.jnts[1].motion_range = torch.tensor([-torch.pi / 2.0, torch.pi / 2.0])
    jlc.jnts[2].loc_pos = torch.tensor([0.0, 0.0, 0.2])
    jlc.jnts[2].loc_motion_ax = torch.tensor([0.0, 1.0, 0.0])
    jlc.jnts[2].motion_range = torch.tensor([-torch.pi, torch.pi])
    jlc.jnts[3].loc_pos = torch.tensor([0.0, 0.0, 0.2])
    jlc.jnts[3].loc_motion_ax = torch.tensor([0.0, 0.0, 1.0])
    jlc.jnts[3].motion_range = torch.tensor([-torch.pi / 2.0, torch.pi / 2.0])
    jlc.jnts[4].loc_pos = torch.tensor([0.0, 0.0, 0.1])
    jlc.jnts[4].loc_motion_ax = torch.tensor([0.0, 1.0, 0.0])
    jlc.jnts[4].motion_range = torch.tensor([-torch.pi / 2.0, torch.pi / 2.0])
    jlc.jnts[5].loc_pos = torch.tensor([0.0, 0.0, 0.05])
    jlc.jnts[5].loc_motion_ax = torch.tensor([0.0, 0.0, 1.0])
    jlc.jnts[5].motion_range = torch.tensor([-torch.pi / 2.0, torch.pi / 2.0])
    jlc._loc_flange_pos = torch.tensor([0.0, 0.0, 0.01])
    jlc._loc_flange_pos = torch.tensor([0.1, 0.1, 0.1])
    # jlc.finalize(ik_solver=None)
    result = jlc.finalize()
    jlc.gen_stickmodel(stick_rgba=bc.navy_blue, toggle_jnt_frames=True, toggle_flange_frame=True).attach_to(base)
    # rand feasible goal
    # random_jnts = jlc.rand_conf()
    # tgt_pose = jlc.fk(jnt_values=random_jnts, update=False)
    # with torch.no_grad():
    #     mgm.gen_frame(pos=tgt_pose[0], rotmat=tgt_pose[1], ax_length=.2).attach_to(base)
    #
    success = 0
    total_time = 10
    time_list = []
    for i in tqdm(range(total_time)):
        jnt_values = torch.zeros(6, requires_grad=True)
        optimizer = LBFGS([jnt_values], lr=1)
        random_jnts = jlc.rand_conf()
        tgt_pose = jlc.fk(jnt_values=random_jnts, update=False)
        def closure():
            optimizer.zero_grad()
            cur_pose = jlc.fk(jnt_values=jnt_values, update=False)
            loss = nkm.diff_between_posrot(*(cur_pose + tgt_pose))
            loss.backward()
            return loss
        tic=time.time()
        for i in range(5):
            loss = optimizer.step(closure)
            if loss < 1e-6:
                toc=time.time()
                print("time cost is: ",  toc-tic)
                print(f"iteration stoped after {i} iterations")
                print(f"loss is: {loss}")
                success += 1
                time_list.append(toc-tic)
                break
    with torch.no_grad():
        mgm.gen_frame(pos=tgt_pose[0], rotmat=tgt_pose[1], ax_length=.2).attach_to(base)
        jlc.fk(jnt_values=jnt_values, update=True)
        jlc.gen_stickmodel(stick_rgba=bc.navy_blue, toggle_flange_frame=True,
                           toggle_jnt_frames=True).attach_to(base)
    print("success rate is: ", success/total_time)
    print("average time cost is: ", sum(time_list)/len(time_list))
    base.run()

    # optimizer = Adam([jnt_values], lr=.1)
    # for i in range(100):
    #     cur_pose = jlc.goto_given_conf(jnt_values)
    #     loss = nkm.diff_between_posrot(*(cur_pose + tgt_pose))
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     print(loss)
    #     if i % 20 == 0:
    #         with torch.no_grad():
    #             mgm.gen_frame(pos=cur_pose[0], rotmat=cur_pose[1]).attach_to(base)
    #             jlc.gen_stickmodel(stick_rgba=bc.cool_map(i/100), toggle_jnt_frames=True, toggle_flange_frame=True).attach_to(
    #                 base)
    # base.run()

    seed_jnt_values = jlc.get_jnt_values()

    success = 0
    num_win = 0
    opt_win = 0
    time_list = []
    tgt_list = []
    for i in tqdm(range(100), desc="ik"):
        random_jnts = jlc.rand_conf()
        tgt_pos, tgt_rotmat = jlc.fk(jnt_values=random_jnts, update=False, toggle_jacobian=False)
        tic = time.time()
        joint_values_with_dbg_info = jlc.ik(tgt_pos=tgt_pos,
                                            tgt_rotmat=tgt_rotmat,
                                            toggle_dbg=False)
        toc = time.time()
        time_list.append(toc - tic)
        print(time_list[-1])
        if joint_values_with_dbg_info is not None:
            success += 1
            if joint_values_with_dbg_info[0] == 'o':
                opt_win += 1
            elif joint_values_with_dbg_info[0] == 'n':
                num_win += 1
            mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
            jlc.fk(jnt_values=joint_values_with_dbg_info, update=True, toggle_jacobian=False)
            jlc.gen_stickmodel(stick_rgba=rm.bc.navy_blue, toggle_flange_frame=True,
                               toggle_jnt_frames=True).attach_to(base)
            base.run()
        else:
            tgt_list.append((tgt_pos, tgt_rotmat))
    print(f'success: {success}')
    print(f'num_win: {num_win}, opt_win: {opt_win}')
    print('average', np.mean(time_list))
    print('max', np.max(time_list))
    print('min', np.min(time_list))
    base.run()
