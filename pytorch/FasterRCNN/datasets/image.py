#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# pytorch/FasterRCNN/datasets/image.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# Image loading and pre-processing.
#
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import imageio
from PIL import Image
import numpy as np
from typing import List
import numpy as np
from skimage.filters import gaussian
from YCbCr_brightness import adjust_brightness_ycbcr
import cv2
from pseudo_hdr import pseudo_hdr_effect
import imageio
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from NLM import non_local_means_denoising
class ChannelOrder(Enum):
  RGB = "RGB"
  BGR = "BGR"

@dataclass
class PreprocessingParams:
  """
  Image preprocessing parameters. Channel order may be either ChannelOrder.RGB or ChannelOrder.BGR.
  Scaling factor is applied first, followed by standardization with supplied means and standard
  deviations supplied in the order specified by channel_order.
  """
  channel_order: ChannelOrder
  scaling: float
  means: List[float]
  stds: List[float]


def _compute_scale_factor(original_width, original_height, min_dimension_pixels):
  if not min_dimension_pixels:
    return 1.0
  if original_width > original_height:
    scale_factor = min_dimension_pixels / original_height
  else:
    scale_factor = min_dimension_pixels / original_width
  return scale_factor

def _preprocess_vgg16(image_data, preprocessing):
  if preprocessing.channel_order == ChannelOrder.RGB:
    pass                                        # already in RGB order
  elif preprocessing.channel_order == ChannelOrder.BGR:
    image_data = image_data[:, :, ::-1]         # RGB -> BGR
  else:
    raise ValueError("Invalid ChannelOrder value: %s" % str(preprocessing.channel_order))
  image_data[:, :, 0] *= preprocessing.scaling
  image_data[:, :, 1] *= preprocessing.scaling
  image_data[:, :, 2] *= preprocessing.scaling
  image_data[:, :, 0] = (image_data[:, :, 0] - preprocessing.means[0]) / preprocessing.stds[0]
  image_data[:, :, 1] = (image_data[:, :, 1] - preprocessing.means[1]) / preprocessing.stds[1]
  image_data[:, :, 2] = (image_data[:, :, 2] - preprocessing.means[2]) / preprocessing.stds[2]
  image_data = image_data.transpose([2, 0, 1])  # (height,width,3) -> (3,height,width)
  return image_data.copy()                      # copy required to eliminate negative stride (which Torch doesn't like)


# TODO: None local mean:
  
def load_image(url, preprocessing, min_dimension_pixels = None, horizontal_flip = False):
  """
  Loads and preprocesses an image for use with the Faster R-CNN model.
  This involves standardizing image pixels to ImageNet dataset-level
  statistics and ensuring channel order matches what the model's
  backbone (feature extractor) expects. The image can be resized so
  that the minimum dimension is a defined size, as recommended by
  Faster R-CNN.

  Parameters
  ----------
  url : str
    URL (local or remote file) to load.
  preprocessing : PreprocessingParams
    Image pre-processing parameters governing channel order and normalization.
  min_dimension_pixels : int
    If not None, specifies the size in pixels of the smaller side of the image.
    The other side is scaled proportionally.
  horizontal_flip : bool
    Whether to flip the image horizontally.

  Returns
  -------
  np.ndarray, PIL.Image, float, Tuple[int, int, int]
    Image pixels as float32, shaped as (channels, height, width); an image
    object suitable for drawing and visualization; scaling factor applied to
    the image dimensions; and the original image shape.
  """
  # # edited by us
  # if not cv2.cuda.getCudaEnabledDeviceCount():
  #       raise RuntimeError("CUDA device not available. Ensure you have a CUDA-enabled GPU and OpenCV is built with CUDA support.")
  
  data = imageio.imread(url, pilmode = "RGB")
  
  # apply pseudo hdr to the image
  data = pseudo_hdr_effect(data)
  data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
  
  # # Upload to GPU
  gpu_data = cv2.cuda_GpuMat()
  gpu_data.upload(data)
  
  # print(type(data),"data class")
  
  # apply denoise
  # CPU:
  # data = cv2.fastNlMeansDenoisingColored(data, None, h=3, hColor=10, templateWindowSize=7, searchWindowSize=21)
  # GPU:
  h_luminance = 3
  photo_render = 10
  search_window = 21
  block_size = 7
  
  # Create a destination GpuMat object
  photo_render_dst = cv2.cuda_GpuMat(gpu_data.size(), gpu_data.type())
    
  # Perform denoising on the GPU
  photo_render_dst = cv2.cuda.fastNlMeansDenoisingColored(gpu_data, h_luminance, photo_render, search_window=search_window, block_size=block_size)
  data = photo_render_dst.download()
  
  # # Apply color balance
  # data = apply_color_balance(data)
    
  data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
  
  # own write NLM
  # data = non_local_means_denoising(data)
  
  # Convert to PIL Image for further processing
  image = Image.fromarray(data, mode = "RGB")
  
  # Enhance edges
  image = image.filter(ImageFilter.EDGE_ENHANCE)
  
  # nlm -> image array
  image = adjust_brightness_ycbcr(image)
  
  image = image.filter(ImageFilter.SHARPEN)
  
  original_width, original_height = image.width, image.height
  if horizontal_flip:
    image = image.transpose(method = Image.FLIP_LEFT_RIGHT)
    
  if min_dimension_pixels is not None:
    scale_factor = _compute_scale_factor(original_width = image.width, original_height = image.height, min_dimension_pixels = min_dimension_pixels)
    width = int(image.width * scale_factor)
    height = int(image.height * scale_factor)
    image = image.resize((width, height), resample = Image.BILINEAR)
  else:
    scale_factor = 1.0
    
  
  image_data = np.array(image).astype(np.float32)
  image_data = _preprocess_vgg16(image_data = image_data, preprocessing = preprocessing)
  
  return image_data, image, scale_factor, (image_data.shape[0], original_height, original_width)

def apply_color_balance(image):
    """
    Apply automatic color balance using OpenCV's white balance algorithm.
    """
    wb = cv2.xphoto.createSimpleWB()
    balanced_image = wb.balanceWhite(image)
    return balanced_image
  
  
def load_image_API(image, preprocessing, min_dimension_pixels = None, horizontal_flip = False):
  # # edited by us
  if not cv2.cuda.getCudaEnabledDeviceCount():
        raise RuntimeError("CUDA device not available. Ensure you have a CUDA-enabled GPU and OpenCV is built with CUDA support.")
  
  data = image
  # apply pseudo hdr to the image
  data = pseudo_hdr_effect(data)
  hdred_image_data = data
  hdred_image = Image.fromarray(hdred_image_data, mode = "RGB")
  
  data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
  
  # # Upload to GPU
  gpu_data = cv2.cuda_GpuMat()
  gpu_data.upload(data)
  
  # print(type(data),"data class")
  
  # apply denoise
  # CPU:
  # data = cv2.fastNlMeansDenoisingColored(data, None, h=3, hColor=10, templateWindowSize=7, searchWindowSize=21)
  # GPU:
  h_luminance = 3
  photo_render = 10
  search_window = 21
  block_size = 7
  
  # Create a destination GpuMat object
  photo_render_dst = cv2.cuda_GpuMat(gpu_data.size(), gpu_data.type())
    
  # Perform denoising on the GPU
  photo_render_dst = cv2.cuda.fastNlMeansDenoisingColored(gpu_data, h_luminance, photo_render, search_window=search_window, block_size=block_size)
  data = photo_render_dst.download()
  
  # # Apply color balance
  # data = apply_color_balance(data)
    
  data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
  
  # own write NLM
  # data = non_local_means_denoising(data)
  
  # Convert to PIL Image for further processing
  image = Image.fromarray(data, mode = "RGB")
  
  original_width, original_height = image.width, image.height
  if horizontal_flip:
    image = image.transpose(method = Image.FLIP_LEFT_RIGHT)
    
  if min_dimension_pixels is not None:
    scale_factor = _compute_scale_factor(original_width = image.width, original_height = image.height, min_dimension_pixels = min_dimension_pixels)
    width = int(image.width * scale_factor)
    height = int(image.height * scale_factor)
    image = image.resize((width, height), resample = Image.BILINEAR)
    hdred_image = hdred_image.resize((width, height), resample = Image.BILINEAR)
  else:
    scale_factor = 1.0
    
  
  image_data = np.array(image).astype(np.float32)
  image_data = _preprocess_vgg16(image_data = image_data, preprocessing = preprocessing)
  
  return hdred_image, image_data, image, scale_factor, (image_data.shape[0], original_height, original_width)
