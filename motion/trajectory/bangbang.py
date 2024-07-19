import numpy as np
import matplotlib.pyplot as plt


def s_curve_control(path, v_max, a_max, j_max, dt=0.01):
    """
    Generate an S-curve control trajectory with multiple passing points.

    :param path: List of positions to pass through [p0, p1, ..., pn]
    :param v_max: Maximum velocity
    :param a_max: Maximum acceleration
    :param j_max: Maximum jerk
    :param dt: Time step for the simulation
    :return: times, positions, velocities, accelerations
    """
    path = np.array(path)
    times = [0]
    positions = [path[0]]
    velocities = [0]
    accelerations = [0]
    jerks = [0]

    for i in range(len(path) - 1):
        delta_q = path[i + 1] - path[i]
        t_j = a_max / j_max  # Time to reach max acceleration (jerk phase)
        t_a = v_max / a_max - t_j  # Time to reach max velocity from zero acceleration
        t_v = (delta_q - (a_max * t_j ** 2 + 0.5 * a_max * t_a ** 2)) / v_max  # Constant velocity phase

        if t_v < 0:
            # Adjust to ensure valid time segments
            t_a = np.sqrt(delta_q / (0.5 * j_max))
            t_j = t_a
            t_v = 0

        t_total = 2 * (t_j + t_a) + t_v
        t_segment = np.arange(0, t_total, dt)

        for t in t_segment:
            if t < t_j:
                j = j_max
                a = j * t
                v = 0.5 * j * t ** 2
                q = path[i] + (1 / 6) * j * t ** 3
            elif t < t_j + t_a:
                j = 0
                a = a_max
                v = a * (t - t_j) + 0.5 * j_max * t_j ** 2
                q = path[i] + (1 / 6) * j_max * t_j ** 3 + 0.5 * a * (t - t_j) ** 2 + 0.5 * j_max * t_j ** 2 * (t - t_j)
            elif t < t_j + t_a + t_j:
                j = -j_max
                a = a_max - j_max * (t - (t_j + t_a))
                v = v_max - 0.5 * j_max * (t_j + t_a - t) ** 2
                q = path[i] + v_max * (t - t_a - t_j) - (1 / 6) * j_max * (t_j + t_a - t) ** 3
            elif t < t_total - (t_j + t_a + t_j):
                j = 0
                a = 0
                v = v_max
                q = path[i] + 0.5 * j_max * t_j ** 3 + 0.5 * a_max * t_a ** 2 + v * (t - 2 * t_j - t_a)
            elif t < t_total - (t_j + t_a):
                j = -j_max
                a = -j_max * (t - (t_total - t_j - t_a - t_j))
                v = v_max - 0.5 * j_max * (t - (t_total - t_j - t_a - t_j)) ** 2
                q = path[i + 1] - (1 / 6) * j_max * (t_total - t - t_j - t_a) ** 3
            elif t < t_total - t_j:
                j = 0
                a = -a_max
                v = -a * (t_total - t_j - t - t_a)
                q = path[i + 1] - 0.5 * a_max * (t_total - t - t_j - t_a) ** 2
            else:
                j = j_max
                a = -a_max + j_max * (t_total - t)
                v = -0.5 * j_max * (t_total - t) ** 2
                q = path[i + 1] - (1 / 6) * j_max * (t_total - t) ** 3

            times.append(times[-1] + dt)
            positions.append(q)
            velocities.append(v)
            accelerations.append(a)
            jerks.append(j)

    return times, positions, velocities, accelerations, jerks


# Example usage
path_points = [0, 1, 3, 6, 7]
v_max = 2
a_max = 1
j_max = 0.5

times, positions, velocities, accelerations, jerks = s_curve_control(path_points, v_max, a_max, j_max)

# Plotting the results
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(times, positions, label='Position')
plt.title('S-Curve Control with Multiple Passing Points')
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(times, velocities, label='Velocity')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(times, accelerations, label='Acceleration')
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(times, jerks, label='Jerk')
plt.xlabel('Time')
plt.ylabel('Jerk')
plt.legend()

plt.tight_layout()
plt.show()
