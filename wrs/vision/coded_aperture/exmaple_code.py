from wrs.vision.coded_aperture.code_aperture import mura
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d, wiener
from skimage.restoration import richardson_lucy
from numpy.fft import fft2, ifft2
from scipy.ndimage import zoom

# 生成 MURA 编码
code = mura(rank=5)
aperture = code.aperture.T

# 加载图像并转换为 numpy 数组
image = Image.open('test.jpg').convert('RGB')
image_array = np.array(image)

# 定义卷积函数
def convolve_channel(channel, aperture):
    return convolve2d(channel, aperture, mode='same', boundary='wrap')

# 对每个颜色通道进行卷积并规范化
def convolve_and_normalize(channel, aperture):
    convolved = convolve_channel(channel, aperture)
    # normalized = 255 * (convolved - np.min(convolved)) / (np.max(convolved) - np.min(convolved))
    # return normalized.astype(np.uint8)
    return convolved

# 对每个颜色通道进行卷积
convolved_r = convolve_and_normalize(image_array[:, :, 0], aperture)
convolved_g = convolve_and_normalize(image_array[:, :, 1], aperture)
convolved_b = convolve_and_normalize(image_array[:, :, 2], aperture)

# 合并卷积后的通道
convolved_image = np.stack((convolved_r, convolved_g, convolved_b), axis=2)

# 定义反卷积函数（使用 Richardson-Lucy 算法）
def richardson_lucy_deconvolve(channel, psf, iterations=30):
    deconvolved = richardson_lucy(channel, psf, num_iter=iterations)
    deconvolved = 255 * (deconvolved - np.min(deconvolved)) / (np.max(deconvolved) - np.min(deconvolved))
    return deconvolved.astype(np.uint8)

# 定义反卷积函数（使用 Wiener 滤波）
def wiener_deconvolve(channel, psf, noise_variance=0.1):
    deconvolved = wiener(channel, mysize=psf.shape, noise=noise_variance)
    deconvolved = 255 * (deconvolved - np.min(deconvolved)) / (np.max(deconvolved) - np.min(deconvolved))
    return deconvolved.astype(np.uint8)

# 定义反卷积函数（使用 FFT）
def fft_deconvolve(channel, psf, epsilon=1e-2):
    # 获取图像和 PSF 的尺寸
    image_shape = channel.shape
    psf_shape = psf.shape
    # 零填充 PSF 到与图像相同的尺寸
    psf_padded = np.zeros(image_shape)
    psf_padded[:psf_shape[0], :psf_shape[1]] = psf
    psf_padded = np.roll(psf_padded, -psf_shape[0] // 2, axis=0)
    psf_padded = np.roll(psf_padded, -psf_shape[1] // 2, axis=1)

    # 计算 FFT
    image_fft = fft2(channel)
    psf_fft = fft2(psf_padded, s=image_shape)

    # 避免除以零，并添加正则化项
    psf_fft = np.where(psf_fft == 0, epsilon, psf_fft)
    psf_inv_fft = np.conj(psf_fft) / (np.abs(psf_fft) ** 2 + epsilon)

    # 进行反卷积
    deconvolved_fft = image_fft * psf_inv_fft
    deconvolved = np.abs(ifft2(deconvolved_fft))

    # 规范化结果
    # deconvolved = 255 * (deconvolved - np.min(deconvolved)) / (np.max(deconvolved) - np.min(deconvolved))
    return deconvolved.astype(np.uint8)

# def enhance_contrast(image):
#     image_pil = Image.fromarray(image)
#     enhancer = ImageEnhance.Contrast(image_pil)
#     image_enhanced = enhancer.enhance(2)  # 调整对比度因子，可以根据需要调整
#     return np.array(image_enhanced)

code.gen_decoder()
code.gen_psf()

# 对每个颜色通道进行反卷积（使用 Richardson-Lucy 算法）
deconvolved_r_rl = richardson_lucy_deconvolve(convolved_r, code.decoder.T, iterations=10)
deconvolved_g_rl = richardson_lucy_deconvolve(convolved_g, code.decoder.T, iterations=10)
deconvolved_b_rl = richardson_lucy_deconvolve(convolved_b, code.decoder.T, iterations=10)
deconvolved_image_rl = np.stack((deconvolved_r_rl, deconvolved_g_rl, deconvolved_b_rl), axis=2)

# 对每个颜色通道进行反卷积（使用 Wiener 滤波）
deconvolved_r_wiener = wiener_deconvolve(convolved_r, code.decoder.T, noise_variance=0.1)
deconvolved_g_wiener = wiener_deconvolve(convolved_g, code.decoder.T, noise_variance=0.1)
deconvolved_b_wiener = wiener_deconvolve(convolved_b, code.decoder.T, noise_variance=0.1)
deconvolved_image_wiener = np.stack((deconvolved_r_wiener, deconvolved_g_wiener, deconvolved_b_wiener), axis=2)

# 对每个颜色通道进行反卷积（使用 FFT）
deconvolved_r_fft = fft_deconvolve(convolved_r, code.decoder.T)
deconvolved_g_fft = fft_deconvolve(convolved_g, code.decoder.T)
deconvolved_b_fft = fft_deconvolve(convolved_b, code.decoder.T)
deconvolved_image_fft = np.stack((deconvolved_r_fft, deconvolved_g_fft, deconvolved_b_fft), axis=2)

# 显示结果
plt.figure(figsize=(24, 6))

plt.subplot(2, 5, 1)
plt.title('Original Image')
plt.imshow(image_array)
plt.axis('off')

plt.subplot(2, 5, 2)
plt.title('Aperture')
zoom_factor = 100
zoomed_aperture = zoom(aperture, zoom_factor, order=0)
plt.imshow(zoomed_aperture, cmap='gray')
plt.axis('off')

plt.subplot(2, 5, 3)
plt.title('Convolved Image')
plt.imshow(convolved_image)
plt.axis('off')

plt.subplot(2, 5, 4)
plt.title('Deconvolved (Richardson-Lucy)')
plt.imshow(deconvolved_image_rl)
plt.axis('off')

plt.subplot(2, 5, 5)
plt.title('Deconvolved (Wiener)')
plt.imshow(deconvolved_image_wiener)
plt.axis('off')

plt.subplot(2, 5, 6)
plt.title('Deconvolved (FFT)')
plt.imshow(deconvolved_image_fft)
plt.axis('off')

plt.show()
