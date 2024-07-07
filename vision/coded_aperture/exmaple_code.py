from vision.coded_aperture.code_aperture import mura
from PIL import Image
import numpy as np
from scipy.ndimage import convolve, zoom
import matplotlib.pyplot as plt

code = mura(rank=5)
aperture = code.aperture.T

image = Image.open('test.jpg')
image_array = np.array(image)

convolved_r = convolve(image_array[:, :, 0], aperture, mode='constant', cval=0.0)
convolved_g = convolve(image_array[:, :, 1], aperture, mode='constant', cval=0.0)
convolved_b = convolve(image_array[:, :, 2], aperture, mode='constant', cval=0.0)

convolved_image = np.stack((convolved_r, convolved_g, convolved_b), axis=2)

# reverse
def deconvolve(image, decoder):
    kernel_ft = np.fft.fft2(decoder, s=image.shape)
    image_ft = np.fft.fft2(image)
    # Avoid division by zero
    kernel_ft = np.where(kernel_ft == 0, 1e-10, kernel_ft)
    deconvolved_ft = image_ft / kernel_ft
    deconvolved = np.fft.ifft2(deconvolved_ft)
    return np.abs(deconvolved).astype(np.uint8)

code.gen_decoder(method='matched')
decoder = code.decoder.T

convolved_r = deconvolve(convolved_image[:, :, 0], decoder)
convolved_g = deconvolve(convolved_image[:, :, 1], decoder)
convolved_b = deconvolve(convolved_image[:, :, 2], decoder)

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