from vision.coded_aperture.code_aperture import mura
from PIL import Image
import numpy as np
from scipy.ndimage import convolve, zoom
import matplotlib.pyplot as plt
from scipy.signal import convolve2d, wiener
from skimage import restoration

code = mura(rank=5)
aperture = code.aperture.T

image = Image.open('test.jpg')
image_array = np.array(image)

def convolve_and_normalize(channel, aperture):
    convolved = convolve2d(channel, aperture, mode='same', boundary='fill', fillvalue=0)
    normalized = 255 * (convolved - np.min(convolved)) / (np.max(convolved) - np.min(convolved))
    return normalized.astype(np.uint8)

convolved_r = convolve_and_normalize(image_array[:, :, 0], aperture)
convolved_g = convolve_and_normalize(image_array[:, :, 1], aperture)
convolved_b = convolve_and_normalize(image_array[:, :, 2], aperture)

convolved_image = np.stack((convolved_r, convolved_g, convolved_b), axis=2)

# # reverse
# def deconvolve(image, decoder):
#     kernel_ft = np.fft.fft2(decoder, s=image.shape)
#     image_ft = np.fft.fft2(image)
#     # Avoid division by zero
#     kernel_ft = np.where(kernel_ft == 0, 1e-10, kernel_ft)
#     deconvolved_ft = image_ft / kernel_ft
#     deconvolved = np.fft.ifft2(deconvolved_ft)
#     return np.abs(deconvolved).astype(np.uint8)
#     return deconvolve

def wiener_deconvolve(channel, aperture, noise_variance=0.01):
    # Apply Wiener filter
    deconvolved = wiener(channel, mysize=aperture.shape, noise=noise_variance)
    # Normalize the deconvolved image
    deconvolved = 255 * (deconvolved - np.min(deconvolved)) / (np.max(deconvolved) - np.min(deconvolved))
    return deconvolved.astype(np.uint8)

# Deconvolution using Richardson-Lucy algorithm
def richardson_lucy_deconvolve(channel, aperture, iterations=30):
    # Apply Richardson-Lucy deconvolution
    deconvolved = restoration.richardson_lucy(channel, aperture, num_iter=iterations)
    # Normalize the deconvolved image
    deconvolved = 255 * (deconvolved - np.min(deconvolved)) / (np.max(deconvolved) - np.min(deconvolved))
    return deconvolved.astype(np.uint8)

code.gen_decoder(method='matched')
decoder = code.decoder.T

convolved_r = richardson_lucy_deconvolve(convolved_image[:, :, 0], decoder)
convolved_g = richardson_lucy_deconvolve(convolved_image[:, :, 1], decoder)
convolved_b = richardson_lucy_deconvolve(convolved_image[:, :, 2], decoder)

decode_convolved_image = np.stack((convolved_r, convolved_g, convolved_b), axis=2)

plt.figure()

plt.subplot(1, 4, 1)
plt.title('Original Image')
plt.imshow(image_array)
plt.axis('off')

zoom_factor = 100
zoomed_aperture = zoom(aperture, zoom_factor, order=0)
plt.subplot(1, 4, 3)
plt.title('Aperture')
plt.imshow(zoomed_aperture)
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title('Convolved Image')
plt.imshow(convolved_image, interpolation='none')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title('Decoded Image')
plt.imshow(decode_convolved_image)
plt.axis('off')

plt.show()