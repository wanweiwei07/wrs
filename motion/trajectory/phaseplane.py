import numpy as np
import matplotlib.pyplot as plt

# 示例经过点
waypoints = np.array([
    [0, 0],
    [1, 1],
    [2, 4],
    [3, 9],
    [4, 16]
])

max_velocity = 1.0
max_acceleration = 0.5


def forward_pass(path_points, max_velocity, max_acceleration):
    num_points = len(path_points)
    velocities = np.zeros(num_points)
    velocities[0] = 0

    for i in range(1, num_points):
        dist = np.linalg.norm(path_points[i] - path_points[i - 1])
        velocities[i] = min(max_velocity, np.sqrt(velocities[i - 1] ** 2 + 2 * max_acceleration * dist))

    return velocities


def backward_pass(path_points, velocities, max_velocity, max_acceleration):
    num_points = len(path_points)
    velocities[-1] = 0

    for i in range(num_points - 2, -1, -1):
        dist = np.linalg.norm(path_points[i + 1] - path_points[i])
        velocities[i] = min(velocities[i], np.sqrt(velocities[i + 1] ** 2 + 2 * max_acceleration * dist))

    return velocities


path_points = waypoints
velocities = forward_pass(path_points, max_velocity, max_acceleration)
velocities = backward_pass(path_points, velocities, max_velocity, max_acceleration)

# 生成时间最优轨迹
time_intervals = np.zeros(len(path_points) - 1)
for i in range(len(path_points) - 1):
    dist = np.linalg.norm(path_points[i + 1] - path_points[i])
    avg_velocity = (velocities[i] + velocities[i + 1]) / 2
    time_intervals[i] = dist / avg_velocity

# 累积时间
time = np.zeros(len(path_points))
for i in range(1, len(path_points)):
    time[i] = time[i - 1] + time_intervals[i - 1]

# Plot the results using matplotlib
plt.figure(figsize=(12, 6))

# Plot path points
plt.subplot(1, 2, 1)
plt.plot(path_points[:, 0], path_points[:, 1], marker='o', label='Path')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Path Points')
plt.grid(True)
plt.legend()

# Plot velocity profile
plt.subplot(1, 2, 2)
plt.plot(time, velocities, marker='o', label='Velocity')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.title('Velocity Profile')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
