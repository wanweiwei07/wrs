import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 给定的路径点配置
path_1d = np.array([0, 1, 4, 4, 5, 6, 8, 10])
m = len(path_1d)

# 最大速度和最大加速度限制
max_vel = 2
max_acc = 1

# 目标函数：最小化总时间
def objective(t):
    return np.sum(t)

# 打印调试信息
def debug_print(message, value):
    print(f"{message}: {value}")

# 初始和结束速度为0的约束条件
def initial_velocity_constraint(t, path_1d):
    x = np.cumsum(np.insert(t, 0, 0))
    debug_print("Initial x", x)
    coeffs = np.polyfit(x, path_1d, 5)
    poly = np.poly1d(coeffs)
    d_poly = np.polyder(poly, 1)
    return d_poly(x[0])

def final_velocity_constraint(t, path_1d):
    x = np.cumsum(np.insert(t, 0, 0))
    debug_print("Final x", x)
    coeffs = np.polyfit(x, path_1d, 5)
    poly = np.poly1d(coeffs)
    d_poly = np.polyder(poly, 1)
    return d_poly(x[-1])

# 多项式通过路径点的约束
def through_path_points_constraint(t, path_1d):
    x = np.cumsum(np.insert(t, 0, 0))
    debug_print("Path points x", x)
    coeffs = np.polyfit(x, path_1d, 5)
    poly = np.poly1d(coeffs)
    return poly(x) - path_1d

# 优化问题
def optimize_trajectory(path_1d):
    m = len(path_1d)

    # 初始猜测的时间间隔
    t_initial = np.ones(m - 1)
    debug_print("Initial t", t_initial)

    constraints = [
        {'type': 'eq', 'fun': lambda t: initial_velocity_constraint(t, path_1d)},
        {'type': 'eq', 'fun': lambda t: final_velocity_constraint(t, path_1d)},
        {'type': 'eq', 'fun': lambda t: through_path_points_constraint(t, path_1d)}
    ]

    result = minimize(objective, t_initial, constraints=constraints, method='SLSQP')

    if not result.success:
        raise ValueError("Optimization failed: " + result.message)

    t_optimal = result.x
    debug_print("Optimal t", t_optimal)
    x_optimal = np.cumsum(np.insert(t_optimal, 0, 0))
    debug_print("Optimal x", x_optimal)
    coeffs_optimal = np.polyfit(x_optimal, path_1d, 5)
    return coeffs_optimal, x_optimal

# 优化路径
coeffs_optimal, x_optimal = optimize_trajectory(path_1d)

# 生成多项式插值
optimized_poly = np.poly1d(coeffs_optimal)
x_dense = np.linspace(min(x_optimal), max(x_optimal), 100)
y_dense = optimized_poly(x_dense)
velocities = np.polyder(optimized_poly, 1)(x_dense)
accelerations = np.polyder(optimized_poly, 2)(x_dense)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(x_dense, y_dense, 'b-', label='Position')
plt.plot(x_optimal, path_1d, 'ro', label='Data points')
plt.xlabel('x')
plt.ylabel('Position')
plt.legend()
plt.title('Optimized Quintic Polynomial Interpolation')

plt.subplot(3, 1, 2)
plt.plot(x_dense, velocities, 'g-', label='Velocity')
plt.axhline(y=max_vel, color='r', linestyle='--', label='Max Velocity')
plt.axhline(y=-max_vel, color='r', linestyle='--')
plt.xlabel('x')
plt.ylabel('Velocity')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(x_dense, accelerations, 'r-', label='Acceleration')
plt.axhline(y=max_acc, color='b', linestyle='--', label='Max Acceleration')
plt.axhline(y=-max_acc, color='b', linestyle='--')
plt.xlabel('x')
plt.ylabel('Acceleration')
plt.legend()

plt.tight_layout()
plt.show()
