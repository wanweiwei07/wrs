import os
from PIL import Image

# 输入文件夹
in_dir = "plots"
# 输出文件
out_path = "all_groups_custom.png"

# 行定义（每一行放多少张图）
layout = [3, 2, 2, 4, 3, 2, 4, 3]
row_height = 300  # 每行高度（px）

# 读取所有图片路径
files = sorted([os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.endswith(".png")])
if not files:
    raise RuntimeError(f"没有找到任何 PNG 图片，请确认 {in_dir}/ 里有图。")

# 检查数量
total_needed = sum(layout)
if total_needed > len(files):
    raise RuntimeError(f"需要 {total_needed} 张图，但只找到 {len(files)} 张！")

# 打开并缩放图片
images = []
for f in files:
    img = Image.open(f)
    w, h = img.size
    new_w = int(w * row_height / h)
    img_resized = img.resize((new_w, row_height), Image.LANCZOS)
    images.append(img_resized)

# 拼接排布
rows = []
start = 0
for count in layout:
    rows.append(images[start:start+count])
    start += count

# 每行宽度
row_widths = [sum(img.size[0] for img in row) for row in rows]

# 画布大小
canvas_width = max(row_widths)
canvas_height = len(rows) * row_height
merged = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))

# 拼接
y_offset = 0
for row_imgs in rows:
    x_offset = 0
    for img in row_imgs:
        merged.paste(img, (x_offset, y_offset))
        x_offset += img.size[0]
    y_offset += row_height

# 保存
merged.save(out_path, dpi=(300, 300))
print(f"✅ 拼接完成: {out_path}")
