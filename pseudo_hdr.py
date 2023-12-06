import cv2

def pseudo_hdr_effect(image):
    """
    Apply a pseudo HDR effect to an image using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    """
    # Convert image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Apply CLAHE to the V channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    
    # Convert back to RGB
    image_hdr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return image_hdr
