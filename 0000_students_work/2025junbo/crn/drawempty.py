import pandas as pd
import matplotlib.pyplot as plt
import math
import os

# ===== 参数 =====
fontsize = 14
base_offset = int(fontsize * 0)    # 初始偏移
min_gap = int(fontsize * 1.5)        # 最小间隔
bump = int(fontsize * 0.8)           # 冲突时每次上移

plt.rcParams.update({"font.size": fontsize})

# 输出文件夹
out_dir = "plots"
os.makedirs(out_dir, exist_ok=True)

# 读取 Excel
file_path = "grasp_analysis_results_corner.xlsx"
df = pd.read_excel(file_path, header=None)

# 名称映射
name_to_label = {
    "cobotta_gripper": "C",
    "robotiq_gripper_85": "r",
    "robotiq_gripper_140": "R",
    "wrs_gripper_2": "W",
    "wrs_gripper_4": "P"
}
x_order = list(name_to_label.keys())

# 分组
groups = list(df.groupby(df.iloc[:, 1]))  # 第二列是分组

# 颜色列表
colors = ["steelblue", "salmon", "seagreen", "orange", "orchid", "gray"]

# 尺寸参数
bar_unit = 0.1    # 每个 bar 占用宽度 (inch)
subplot_height = 3.5

for group_name, group_data in groups:
    if group_name == "key":
        # --- 每个 group 内部做线性归一化 ---
        col5 = group_data.iloc[:, 4]  # 第五列 = 基数
        min_val, max_val = col5.min(), col5.max()
        group_data = group_data.copy()
        if max_val > min_val:
            group_data["alpha"] = 0.2 + 0.8 * (col5 - min_val) / (max_val - min_val)
            group_data["alpha"] = group_data["alpha"].clip(0.2, 1.0)
        else:
            group_data["alpha"] = 1.0

        # 透视表：行=gripper(第0列)，列=condition(第2列)
        pivot_val = group_data.pivot_table(
            index=0, columns=2, values=6, aggfunc="first"
        ).reindex(x_order)
        pivot_alpha = group_data.pivot_table(
            index=0, columns=2, values="alpha", aggfunc="first"
        ).reindex(x_order)

        n_cond = len(pivot_val.columns)
        bars_per_subplot = len(x_order) * max(1, n_cond)
        subplot_width = bars_per_subplot * bar_unit + 1.0

        fig, ax = plt.subplots(figsize=(subplot_width, subplot_height))

        x = range(len(x_order))
        bar_width = 0.8 / n_cond if n_cond > 0 else 0.8

        # 记录每个 x 位置上已放置的文字高度（target_y）
        y_last = {}

        max_bar_height = 0

        for i, cond in enumerate(pivot_val.columns):
            y_values = pivot_val[cond].fillna(0).tolist()
            alphas = pivot_alpha[cond].fillna(1).tolist()

            for xi, (val, alp) in enumerate(zip(y_values, alphas)):
                bar = ax.bar(
                    xi + i * bar_width, val,
                    width=bar_width,
                    color=colors[i % len(colors)],
                    edgecolor="black",
                    alpha=alp
                )[0]

                max_bar_height = max(max_bar_height, val)

                if val != 0:
                    label = f"{int(round(val))}" if abs(val - round(val)) < 1e-6 else f"{val:.1f}"

                    if xi not in y_last:
                        y_last[xi] = []

                    offset = base_offset
                    target_y = val + offset

                    # 避免和已有标注重叠
                    while any(abs(target_y - y0) < min_gap for y0 in y_last[xi]):
                        offset += bump
                        target_y = val + offset

                    y_last[xi].append(target_y)

                    # 如果 offset==0 → 不画连线
                    arrowprops = None if offset == 0 else dict(arrowstyle="-", lw=0.8)

                    ax.annotate(label,
                                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                xytext=(0, offset),
                                textcoords="offset points",
                                ha="center", va="bottom",
                                fontsize=fontsize,
                                arrowprops=arrowprops)

        # X 轴
        ax.set_xticks([xi + (n_cond-1)/2*bar_width for xi in x])
        ax.set_xticklabels([name_to_label[k] for k in x_order], fontsize=fontsize, color="white")

        # 去掉 Y 轴
        ax.set_yticks([])
        ax.set_ylabel("")

        # 动态 ylim，给标注留空间
        ax.set_ylim(0, max_bar_height + fontsize * 3)
        ax.set_facecolor("mistyrose")
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, color="red", lw=3)
        ax.plot([0, 1], [1, 0], transform=ax.transAxes, color="red", lw=3)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"group_empty.png")
        plt.savefig(out_path, dpi=300)
        plt.close(fig)

print(f"✅ 所有分组图已保存到 {out_dir}/ (字体{fontsize}pt, 动态连线，最小线长=0)")
