import pandas as pd
import matplotlib.pyplot as plt
import math
import os

# 输出文件夹
out_dir = "plots"
os.makedirs(out_dir, exist_ok=True)

# 读取 Excel（如果是 CSV 就换成 read_csv）
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

# 分组（第二列是分组）
groups = list(df.groupby(df.iloc[:, 1]))

# 颜色列表
colors = ["skyblue", "salmon", "lightgreen", "orange", "violet", "gray"]

# 参数
base_offset = 7   # 默认偏移量
min_gap = 14        # 文字之间至少间隔 8 pt
bar_unit = 0.35    # 每个 bar 占用宽度 (inch)
subplot_height = 3.0

for group_name, group_data in groups:
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

    # === 第一步：画 bar，收集信息 ===
    bars_info = []
    for i, cond in enumerate(pivot_val.columns):
        y_values = pivot_val[cond].fillna(0).tolist()
        alphas   = pivot_alpha[cond].fillna(1).tolist()

        for xi, (val, alp) in enumerate(zip(y_values, alphas)):
            bar = ax.bar(
                xi + i * bar_width, val,
                width=bar_width,
                color=colors[i % len(colors)],
                edgecolor="black",
                alpha=alp
            )[0]
            # if val != 0:
            bars_info.append((xi, i, val, bar))

    # === 第二步：统一加 label ===
    for xi in set(b[0] for b in bars_info):
        tick_bars = [b for b in bars_info if b[0] == xi]
        tick_bars.sort(key=lambda x: x[2])  # 按高度从低到高

        used_positions = []
        for _, i, val, bar in tick_bars:
            target_y = val + base_offset

            # 避免和已有标注冲突
            for y0 in used_positions:
                if abs(target_y - y0) < min_gap:
                    target_y = y0 + min_gap

            used_positions.append(target_y)

            label = f"{int(round(val))}" if abs(val - round(val)) < 1e-6 else f"{val:.1f}"

            ax.annotate(label,
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, target_y - val),
                        textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=14,
                        arrowprops=dict(arrowstyle="-", lw=0.5))

    # X 轴
    ax.tick_params(axis="x", labelsize=14, colors="black")
    ax.set_xticks([xi + (n_cond-1)/2*bar_width for xi in x])
    ax.set_xticklabels([name_to_label[k] for k in x_order])

    # 去掉 Y 轴
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_ylim(0, 100)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"group_{group_name}.png")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

print(f"✅ 所有分组图已保存到 {out_dir}/")
