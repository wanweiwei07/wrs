import numpy as np
import matplotlib.pyplot as plt
he
# 数据
data_main = {
    '1st_column': [23.88801, 36.13535, 49.71259, 62.71275, 75.83295, 88.74965, 102.1793, 115.7686, 129.0042, 143.0163, 154.2564],
    '2nd_column': [24.36376, 37.29031, 50.77564, 63.69682, 76.85656, 90.21013, 103.6176, 117.0839, 130.6391, 143.8036, None],
    '3rd_column': [23.88801, 37.08801, 50.28801, 63.48801, 76.68801, 89.88801, 103.088, 116.288, 129.488, 142.688, 155.888]
}

open_data = np.array(data_main['1st_column'])
theory_data = np.array(data_main['3rd_column'])

# 误差
open_error = open_data - theory_data
close_error = np.array([c - t if c is not None else 0
                        for c, t in zip(data_main['2nd_column'], theory_data)])

# 百分比
percent = np.linspace(0, 100, len(theory_data))
theory_min = theory_data[0]
xtick_vals = theory_data - theory_min

# 颜色
colors = ["skyblue", "salmon"]

# 参数
bar_width = 0.375
subplot_height = 4.5
subplot_width = len(percent) * bar_width * 2 + 1.0

fig, ax = plt.subplots(figsize=(subplot_width, subplot_height))

x = np.arange(len(percent))

# === 使用之前的比例拉伸高度 ===
scale_factor = 6
bars1 = ax.bar(x - bar_width/2, open_error * scale_factor,
               width=bar_width, color=colors[0], edgecolor="black", label="Open Error")
bars2 = ax.bar(x + bar_width/2, close_error * scale_factor,
               width=bar_width, color=colors[1], edgecolor="black", label="Close Error")

# === 标注函数 ===
all_labels_y = []

def annotate_bars(bars, values):
    for bar, val in zip(bars, values):
        if val is None:
            continue
        scaled_val = val * scale_factor
        if val >= 0:
            ypos = bar.get_height()
            ax.annotate(f'{val:.2f}',
                        xy=(bar.get_x() + bar.get_width()/2, ypos),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=14)
            all_labels_y.append(ypos + 3)
        else:
            ypos = bar.get_height()
            ax.annotate(f'{val:.2f}',
                        xy=(bar.get_x() + bar.get_width()/2, ypos),
                        xytext=(0, -3), textcoords="offset points",
                        ha='center', va='top', fontsize=14)
            all_labels_y.append(ypos - 3)

annotate_bars(bars1, open_error)
annotate_bars(bars2, close_error)

# === 坐标轴样式 ===
xticklabels = [f"{v:.1f} ({int(p)}%)" for v, p in zip(xtick_vals, percent)]
ax.set_xticks(x)
ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=14, color="black")

ax.tick_params(axis="y", labelsize=14, colors="black")
ax.set_xlabel("Theoretical Distance Progress (mm, %)", fontsize=14)
ax.set_ylabel("Error (measured - theoretical)", fontsize=14)

# 图例放到上侧
ax.legend(fontsize=14, loc='lower center',
          bbox_to_anchor=(0.5, 1.02), ncol=2, borderaxespad=0)

# 自动调整 y 范围，保证所有标注在框内并稍微增加边距
all_values = (open_error * scale_factor).tolist() + (close_error * scale_factor).tolist() + all_labels_y
ymin = min(all_values)
ymax = max(all_values)
y_range = ymax - ymin
ax.set_ylim(ymin - 0.05 * y_range, ymax + 0.05 * y_range)

plt.tight_layout()
plt.savefig("backslash.png", dpi=300, bbox_inches="tight")
plt.close()
