import matplotlib.pyplot as plt
import numpy as np
import imageio
from PIL import Image

def plot_image_histogram(url):
    # Load the image
    image = imageio.imread(url, pilmode='L')  # 'L' mode for grayscale
    image_pil = Image.fromarray(image)
    
    # Convert the image to numpy array and flatten it
    image_data = np.array(image_pil).flatten()
    
    # Plot the histogram
    plt.hist(image_data, bins=256, range=[0,256], density=True, color='gray', alpha=0.75)
    plt.title("Image Brightness Histogram")
    plt.xlabel("Brightness Level")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

url = "VOCdevkit\\VOC2007\\JPEGImages\\1938.jpg"
plot_image_histogram(url)