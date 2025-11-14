import pandas as pd
import matplotlib.pyplot as plt
import math

# 读取 Excel
file_path = "grasp_analysis_results_standard.xlsx"
df = pd.read_excel(file_path, header=None)

# 列索引
col1 = df.iloc[:, 0]   # 第一列（夹爪名称）
col2 = df.iloc[:, 1]   # 第二列（分组）
col7 = df.iloc[:, 6]   # 第七列（数值）

# 名称映射
name_to_label = {
    "cobotta_gripper": "C",
    "robotiq_gripper_85": "r",
    "robotiq_gripper_140": "R",
    "wrs_gripper_2": "W",
    "wrs_gripper_4": "P"
}
x_order = [
    "cobotta_gripper",
    "robotiq_gripper_85",
    "robotiq_gripper_140",
    "wrs_gripper_2",
    "wrs_gripper_4"
]

# 按第二列分组
groups = df.groupby(col2)

# subplot 排布（每行 6 个）
n_groups = len(groups)
n_cols = 6
n_rows = math.ceil(n_groups / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
axes = axes.flatten()

for idx, (group_name, group_data) in enumerate(groups):
    ax = axes[idx]
    col = idx % n_cols

    # 按固定顺序取数值
    y_values = []
    for key in x_order:
        subset = group_data[group_data.iloc[:, 0] == key]
        if not subset.empty:
            y_values.append(subset.iloc[0, 6])
        else:
            y_values.append(0)

    # 画柱状图
    bars = ax.bar([name_to_label[k] for k in x_order], y_values,
                  color="skyblue", edgecolor="black")
    ax.set_ylim(0, 100)
    ax.set_title(f"Group {group_name}", fontsize=9)

    # 在每个 bar 上标数值
    for bar, value in zip(bars, y_values):
        if value == 0:
            continue  # 跳过 0，不显示
        elif abs(value - round(value)) < 1e-6:
            label = f"{int(round(value))}"  # 整数
        else:
            label = f"{value:.1f}"          # 一位小数

        if value >= 90:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 5,
                    label, ha="center", va="top", fontsize=7, color="white")
        else:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    label, ha="center", va="bottom", fontsize=7)

    # 只保留每行第一个 subplot 的 y 轴
    if col != 0:
        ax.set_yticks([])
        ax.set_ylabel("")

# 删除多余 subplot
for j in range(n_groups, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
