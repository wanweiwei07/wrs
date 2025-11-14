import pandas as pd
import matplotlib.pyplot as plt
import os

# 读取 Excel
file_path = "grasp_analysis_results_standard.csv"
# df = pd.read_excel(file_path, header=None)
df = pd.read_csv(file_path, header=None)

# 名称映射（横轴标签）
name_to_label = {
    "cobotta_gripper": "C",
    "robotiq_gripper_85": "r",
    "robotiq_gripper_140": "R",
    "wrs_gripper_2": "W",
    "wrs_gripper_4": "P"
}
x_order = list(name_to_label.keys())

# 输出目录
out_dir = "plots"
os.makedirs(out_dir, exist_ok=True)

# 按第二列分组
groups = list(df.groupby(df.iloc[:, 1]))

# 指定显示纵轴刻度的图编号（从 1 开始数）
show_yaxis = {}

for idx, (group_name, group_data) in enumerate(groups, start=1):
    # 按固定顺序取数值
    y_values = []
    x_labels = []
    for key in x_order:
        subset = group_data[group_data.iloc[:, 0] == key]
        if not subset.empty:
            y_values.append(subset.iloc[0, 6])
        else:
            y_values.append(0)
        x_labels.append(name_to_label[key])

    fig, ax = plt.subplots(figsize=(3, 2.5))

    # 🌟 判断是否 P 最大
    if y_values[-1] == max(y_values):  # y_values[-1] 对应 wrs_gripper_4 (P)
        ax.set_facecolor("honeydew")  # 浅绿色背景 (可以改为 "lightgreen")
    bars = ax.bar(x_labels, y_values, color="skyblue", edgecolor="black")
    ax.set_ylim(0, 100)

    # 横轴字体大小
    ax.tick_params(axis="x", labelsize=12)

    # 控制纵轴刻度
    if idx in show_yaxis:
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.tick_params(axis="y", labelsize=10)
    else:
        ax.set_yticks([])

    # 去掉标题
    ax.set_title("")
    ax.set_ylabel("")

    # 在每个 bar 上标数值
    for bar, value in zip(bars, y_values):
        if abs(value - round(value)) < 1e-6:
            label = f"{int(round(value))}"
        else:
            label = f"{value:.1f}"

        if value >= 90:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 5,
                    label, ha="center", va="top", fontsize=14, color="black")
        else:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    label, ha="center", va="bottom", fontsize=14)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"group{idx}_{group_name}.png")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

print(f"✅ 所有图已保存到文件夹: {out_dir}，横轴为 C r R W P")
