import os
import argparse
import imageio
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

from YCbCr_brightness import adjust_brightness_ycbcr
from pseudo_hdr import pseudo_hdr_effect

# Function to ensure the output directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to construct output filenames
def construct_filename(output_dir, original_name, suffix, ext=".jpeg"):
    base_name = os.path.basename(original_name)
    name, _ = os.path.splitext(base_name)
    new_filename = f"{name}_{suffix}{ext}"
    return os.path.join(output_dir, new_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Processing Script")

    parser.add_argument("--output", metavar="dir", required=True, help="Output directory to save processed images")
    parser.add_argument("--image", metavar="path", required=True, help="Path to the input image to process")

    options = parser.parse_args()

    # Ensure the output directory exists
    ensure_dir(options.output)

    # Read the input image
    input_image_path = options.image
    data = imageio.imread(input_image_path, pilmode="RGB")

    # Apply pseudo HDR effect
    data = pseudo_hdr_effect(data)
    hdr_filename = construct_filename(options.output, input_image_path, 'after_pseudo_HDR')
    imageio.imwrite(hdr_filename, data, quality=100)

    # Convert image for OpenCV processing
    data = cv2.cvtColor(np.array(data), cv2.COLOR_RGB2BGR)

    # Apply Non-Local Means Denoising
    gpu_data = cv2.cuda_GpuMat()
    gpu_data.upload(data)

    h_luminance = 3
    photo_render = 10
    search_window = 21
    block_size = 7
  
    # Create a destination GpuMat object
    photo_render_dst = cv2.cuda_GpuMat(gpu_data.size(), gpu_data.type())
        
    # Perform denoising on the GPU
    photo_render_dst = cv2.cuda.fastNlMeansDenoisingColored(gpu_data, h_luminance, photo_render, search_window=search_window, block_size=block_size)
    data = photo_render_dst.download()
    
    # Convert back to RGB for saving
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    
    NLM_filename = construct_filename(options.output, input_image_path, 'after_NLM')
    imageio.imwrite(hdr_filename, data, quality=100)

    # Convert to PIL Image for further processing
    data = Image.fromarray(data)

    # Enhance edges
    data = data.filter(ImageFilter.EDGE_ENHANCE)
    edge_enhanced_filename = construct_filename(options.output, input_image_path, 'after_EDGE_ENHANCEMENT')
    data.save(edge_enhanced_filename, 'JPEG', quality=100)

    # YCbCr adjustment
    data = adjust_brightness_ycbcr(data)
    ycbcr_filename = construct_filename(options.output, input_image_path, 'after_YCbCr')
    data.save(ycbcr_filename, 'JPEG', quality=100)

    # Sharpen the image
    data = data.filter(ImageFilter.SHARPEN)
    sharpened_filename = construct_filename(options.output, input_image_path, 'after_sharpen')
    data.save(sharpened_filename, 'JPEG', quality=100)

    print(f"Processing complete. Images saved to {options.output}")
