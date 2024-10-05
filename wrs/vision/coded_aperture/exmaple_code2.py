import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from wrs.vision.coded_aperture.code_aperture import mura

# 生成MURA编码
code = mura(rank=5)
aperture = code.aperture.T

# 加载图像并转换为numpy数组
image = Image.open('test.jpg').convert('RGB')
image_array = np.array(image)

# 将图像转换为张量
image_tensor = torch.tensor(image_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

# 定义卷积函数
def convolve_tensor(image_tensor, aperture_tensor):
    convolved_tensor = F.conv2d(image_tensor, aperture_tensor, padding=aperture.shape[0] // 2, groups=3)
    return convolved_tensor

# 对每个颜色通道进行卷积
aperture_tensor = torch.tensor(aperture, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
aperture_tensor = aperture_tensor.repeat(3, 1, 1, 1)  # Repeat for each RGB channel
convolved_image_tensor = F.conv2d(image_tensor, aperture_tensor, padding=aperture.shape[0] // 2, groups=3)

# 定义损失函数（均方误差）
def loss_function(reconstructed_image, original_convolved_image):
    return F.mse_loss(convolve_tensor(reconstructed_image, aperture_tensor), original_convolved_image)

# 初始化重建图像（可以使用全零图像）
reconstructed_image = torch.zeros_like(image_tensor, requires_grad=True)

# 使用梯度下降法优化
optimizer = torch.optim.Adam([reconstructed_image], lr=0.2)
num_iterations = 100

for i in range(num_iterations):
    optimizer.zero_grad()
    loss = loss_function(reconstructed_image, convolved_image_tensor)
    loss.backward()
    optimizer.step()

    if (i + 1) % 10 == 0:
        print(f"Iteration {i+1}/{num_iterations}, Loss: {loss.item()}")

# 将重建图像转换为numpy数组
reconstructed_image_np = reconstructed_image.detach().squeeze(0).permute(1, 2, 0).numpy() * 255.0
reconstructed_image_np = np.clip(reconstructed_image_np, 0, 255).astype(np.uint8)

# 显示原始图像、卷积图像和重建图像
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image_array)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Convolved Image')
plt.imshow(convolved_image_tensor.squeeze(0).permute(1, 2, 0).detach().numpy())
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Reconstructed Image')
plt.imshow(reconstructed_image_np)
plt.axis('off')

plt.show()
