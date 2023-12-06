import numpy as np
import imageio
from PIL import Image

def non_local_means_denoising(image, h=10, patch_size=3, search_window=21):
    pad_width = search_window // 2
    padded_image = np.pad(image, pad_width, mode='reflect')
    denoised_image = np.zeros_like(image)
    h_squared = h * h + 1e-6  # Adding a small constant to avoid division by zero

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            i1 = i + pad_width
            j1 = j + pad_width
            W1 = padded_image[i1 - pad_width:i1 + pad_width + 1, j1 - pad_width:j1 + pad_width + 1]

            average = 0.0
            s_weights = 0.0

            for k in range(max(i1 - pad_width, pad_width), min(i1 + pad_width, padded_image.shape[0] - pad_width)):
                for l in range(max(j1 - pad_width, pad_width), min(j1 + pad_width, padded_image.shape[1] - pad_width)):
                    if k == i1 and l == j1:
                        continue

                    W2 = padded_image[k - pad_width:k + pad_width + 1, l - pad_width:l + pad_width + 1]
                    d = np.sum((W1 - W2) ** 2)

                    weight = np.exp(-d / h_squared)
                    s_weights += weight
                    average += weight * padded_image[k, l]

            if s_weights > 0:
                average /= s_weights

            denoised_image[i, j] = np.clip(average, 0, 255)  # Clip values to valid range

    return denoised_image