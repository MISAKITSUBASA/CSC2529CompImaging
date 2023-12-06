from PIL import ImageFilter

def edge_enhancement(image):
    """
    Enhance edges in an image.
    """
    return image.filter(ImageFilter.EDGE_ENHANCE)