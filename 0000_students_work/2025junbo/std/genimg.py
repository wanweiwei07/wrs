import os
from PIL import Image
import math
import re

# 输入目录（之前保存的小图）
in_dir = "plots"
# 输出文件
out_path = "merged.png"

# 提取文件里的编号，例如 group12_xxx.png → 12
def extract_idx(filename):
    match = re.search(r"group(\d+)_", filename)
    return int(match.group(1)) if match else 0

# 读取所有图片路径，并按照 idx 排序
files = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.endswith(".png")]
files = sorted(files, key=lambda f: extract_idx(os.path.basename(f)))

if not files:
    raise RuntimeError("没有找到任何 PNG 图片，请确认 plots/ 文件夹里有图。")

# 打开所有图片
images = [Image.open(f) for f in files]

# 每行 6 个
n_cols = 6
n_imgs = len(images)
n_rows = math.ceil(n_imgs / n_cols)

# 假设所有图片大小相同
w, h = images[0].size

# 创建大画布
merged = Image.new("RGB", (n_cols * w, n_rows * h), color=(255, 255, 255))

# 把小图贴到大图
for idx, img in enumerate(images):
    row = idx // n_cols
    col = idx % n_cols
    merged.paste(img, (col * w, row * h))

# 保存
merged.save(out_path, dpi=(300, 300))
print(f"✅ 拼接完成，大图已保存: {out_path}")
