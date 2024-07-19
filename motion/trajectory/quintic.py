import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

class QuinticSpline(object):

    def __init__(self, x, y):
        bc_type = [
            ((1, [0.0, 0.0, 0.0]), (2, [0.0, 0.0, 0.0])),
            ((1, [0.0, 0.0, 0.0]), (2, [0.0, 0.0, 0.0]))
        ]
        self.spl = make_interp_spline(x, y, k=5, bc_type=bc_type)

    def __call__(self, x):
        return self.spl(x)

    def derivative(self, n=1):
        return self.spl.derivative(nu=n)


if __name__ == '__main__':

    n = 10  # 时间步长数
    n_jnts = 3  # 关节数
    np.random.seed(0)
    joint_values = np.random.rand(n, n_jnts)
    time_steps = np.arange(n)
    quintic_spline = QuinticSpline(time_steps, joint_values)

    # bc_type = [
    #     ((1, [0.0, 0.0, 0.0]), (2, [0.0, 0.0, 0.0])),
    #     ((1, [0.0, 0.0, 0.0]), (2, [0.0, 0.0, 0.0]))
    # ]
    # # quintic_spline = QuinticSpline(time_steps, joint_values)
    # quintic_spline = make_interp_spline(time_steps, joint_values, k=5, bc_type=bc_type)
    # 绘制结果
    # 生成插值数据
    time_dense = np.linspace(0, n - 1, 100)
    # 绘制结果
    fig, axs = plt.subplots(n_jnts, 3, figsize=(18, 18))

    joint_dense = quintic_spline(time_dense)
    print(joint_dense)
    joint_vel = quintic_spline.derivative()(time_dense)
    joint_acc = quintic_spline.derivative(2)(time_dense)

    for j in range(n_jnts):
        axs[j, 0].plot(time_steps, joint_values[:, j], 'ro', label=f'Joint {j + 1} data points')
        axs[j, 0].plot(time_dense, joint_dense[:, j], 'b-', label=f'Joint {j + 1} quintic spline')
        axs[j, 0].set_xlabel('Time')
        axs[j, 0].set_ylabel(f'Joint {j + 1} value')
        axs[j, 0].legend()
        axs[j, 0].set_title(f'Quintic Spline Interpolation for Joint {j + 1}')

        axs[j, 1].plot(time_dense, joint_vel[:, j], 'g-', label=f'Joint {j + 1} velocity')
        axs[j, 1].set_xlabel('Time')
        axs[j, 1].set_ylabel(f'Joint {j + 1} velocity')
        axs[j, 1].legend()
        axs[j, 1].set_title(f'Velocity for Joint {j + 1}')

        axs[j, 2].plot(time_dense, joint_acc[:, j], 'm-', label=f'Joint {j + 1} acceleration')
        axs[j, 2].set_xlabel('Time')
        axs[j, 2].set_ylabel(f'Joint {j + 1} acceleration')
        axs[j, 2].legend()
        axs[j, 2].set_title(f'Acceleration for Joint {j + 1}')

    plt.tight_layout()
    plt.show()
