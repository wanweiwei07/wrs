import numpy as np
import matplotlib.pyplot as plt

# 数据
data_array = np.array([
    [58.4, 58.1, 60.6, 61.4, 57.7],
    [72.1, 71.9, 67.8, 69.8, 70.7],
    [71.5, 68.6, 70.0, 71.0, 68.6],
    [70.5, 72.4, 71.5, 69.6, 68.5],
    [68.5, 69.4, 71.3, 69.2, 70.1],
])

positions = [10, 20, 30, 40, 50]  # mm
n_positions, n_repeats = data_array.shape

# bar 参数
bar_width = 0.15
x = np.arange(n_positions)

colors = ["skyblue", "salmon", "lightgreen", "orange", "violet"]

fig, ax = plt.subplots(figsize=(10, 5))
bars_all = []
scale = 0.9  # bar高度缩短比例

# === 修改部分：按行绘制，每行是一个 group ===
for i in range(n_positions):  # 遍历每个位置
    for j in range(n_repeats):  # 遍历该位置的重复实验
        bars = ax.bar(x[i] + j*bar_width - (n_repeats-1)/2*bar_width,
                      data_array[i, j] * scale,
                      width=bar_width,
                      color=colors[j % len(colors)],
                      edgecolor="black")
        bars_all.append((bars, data_array[i, j]))

# === 标注函数（单个值版本） ===
def annotate_bar(bar, val, group_heights):
    min_gap = 1 # y轴数值间隔（非像素），根据需要可调
    base_offset=.8
    bar_top = bar.get_height()
    xpos = round(bar.get_x() + bar.get_width()/2, 3)
    target_y = bar_top+base_offset
    while any(abs(target_y - h) < min_gap for h in group_heights[-2:]):
        target_y += min_gap
    ax.annotate(f'{val:.1f}',
                xy=(xpos, bar_top),
                xytext=(xpos, target_y), textcoords="data",
                ha='center', va='bottom', fontsize=14,
                arrowprops=dict(arrowstyle="-", color="black", lw=0.8))
    group_heights.append(target_y)

# === 给每个位置单独标注，避免跨位置冲突 ===
for i in range(n_positions):
    group_heights = []
    for j in range(n_repeats):
        bar = ax.patches[i*n_repeats + j]  # 找到对应 bar
        val = data_array[i, j]
        annotate_bar(bar, val, group_heights)

# === 坐标轴样式 ===
ax.set_xticks(x)
ax.set_xticklabels([f"{p} mm" for p in positions], fontsize=14, color="black")
ax.tick_params(axis="y", labelsize=14, colors="black")
ax.set_xlabel("Finger Position", fontsize=14)
ax.set_ylabel("Max Grasp Force (N)", fontsize=14)

# 自动调整 y 范围
ymin = 50
ymax = 75
ax.set_ylim(ymin, ymax)

plt.tight_layout()
plt.savefig("grouped_force_arrow_byrow.png", dpi=300, bbox_inches="tight")
plt.close()
