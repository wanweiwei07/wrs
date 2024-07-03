import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def generate_mura_mask(N):
    """Generate a MURA mask of size NxN."""
    p = (N+1) // 2
    mask = np.ones((N, N), dtype=int)
    for x in range(N):
        for y in range(N):
            mask[x, y] = ((x * y) % p) % 2
    return mask

def encode_image(image, mask):
    """Encode the image using the MURA mask."""
    encoded = convolve2d(image, mask, mode='same', boundary='wrap')
    return encoded

def decode_image(encoded_image, mask):
    """Decode the encoded image using the MURA mask."""
    decoded = convolve2d(encoded_image, mask[::-1, ::-1], mode='same', boundary='wrap')
    return decoded

def main():
    # Image size and MURA mask size
    image_size = 320
    mask_size = 31  # Set the mask size to the image size

    # Generate a synthetic image
    image = np.zeros((image_size, image_size))
    image[80:240, 80:240] = 1  # A simple square as an example


    # Generate the MURA mask
    mask = generate_mura_mask(mask_size)

    # Encode the image
    encoded_image = encode_image(image, mask)

    # Decode the image
    decoded_image = decode_image(encoded_image, mask)

    # Normalize the decoded image
    decoded_image /= np.max(decoded_image)

    # Plot the results
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('MURA Mask')
    axes[2].imshow(encoded_image, cmap='gray')
    axes[2].set_title('Encoded Image')
    axes[3].imshow(decoded_image, cmap='gray')
    axes[3].set_title('Decoded Image')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
