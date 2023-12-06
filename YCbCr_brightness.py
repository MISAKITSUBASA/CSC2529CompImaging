from PIL import Image, ImageStat
import numpy as np

def adjust_brightness_ycbcr(image):
    """
    Adjusts the brightness of an image more gently based on its histogram, aiming to preserve details.
    image: A PIL.Image object in RGB format.
    Returns: Adjusted PIL.Image object in RGB format.
    """
    # Convert image to YCbCr and split the channels
    ycbcr_image = image.convert('YCbCr')
    y, cb, cr = ycbcr_image.split()
    
    # Calculate the histogram of the Y channel
    y_histogram = y.histogram()
    
    # Calculate the current average brightness
    stat = ImageStat.Stat(y)
    current_brightness = stat.mean[0]

    # Calculate the target brightness. Aim for a gentler adjustment.
    midpoint = np.argmax(y_histogram)
    target_brightness = 128  # Midpoint of the brightness range (0-255)

    # Adjust the target brightness based on the histogram's peak, but more gently
    if midpoint < 128:
        target_brightness += np.sqrt(128 - midpoint)
    elif midpoint > 128:
        target_brightness -= np.sqrt(midpoint - 128)

    # Scale the Y channel to adjust brightness, with a limit to prevent over-adjustment
    y_np = np.array(y, dtype=np.float32)
    adjustment_factor = min(max(target_brightness / current_brightness, 0.5), 1.5)
    y_np *= adjustment_factor
    y_np = np.clip(y_np, 0, 255).astype(np.uint8)
    y = Image.fromarray(y_np, mode='L')
    
    # Merge the channels back together and convert back to RGB
    adjusted_image = Image.merge('YCbCr', (y, cb, cr)).convert('RGB')
    
    return adjusted_image


# # Method 2:
# import numpy as np
# import imageio
# from PIL import Image, ImageStat

# def adjust_brightness_ycbcr(image):
#     """
#     Normalizes the luminance of an image to have a specific target mean and standard deviation.
#     image: A PIL.Image object in RGB format.
#     Returns: Adjusted PIL.Image object in RGB format.
#     """
#     # Convert image to YCbCr
#     ycbcr_image = image.convert('YCbCr')
#     y, cb, cr = ycbcr_image.split()
    
#     # Calculate the current mean and standard deviation of the luminance
#     y_np = np.array(y, dtype=np.float32)
#     mean_y = np.mean(y_np)
#     std_y = np.std(y_np)
    
#     # Define target mean and standard deviation for luminance
#     target_mean_y = 128.0  # Neutral brightness
#     target_std_y = 64.0    # Target a moderate standard deviation for contrast
    
#     # Normalize the luminance channel
#     y_np = (y_np - mean_y) / std_y * target_std_y + target_mean_y
    
#     # Clip the values to be in the 8-bit range and convert back to image
#     y_np = np.clip(y_np, 0, 255).astype(np.uint8)
#     y = Image.fromarray(y_np, mode='L')
    
#     # Merge the channels back together and convert back to RGB
#     adjusted_image = Image.merge('YCbCr', (y, cb, cr)).convert('RGB')
    
#     return adjusted_image



