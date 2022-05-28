# Project Image Filtering and Hybrid Images Stencil Code
# Based on previous and current work
# by James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale
from numpy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift
'''
def my_imfilter(image: np.ndarray, filter: np.ndarray):
  """
  Your function should meet the requirements laid out on the project webpage.
  Apply a filter to an image. Return the filtered image.
  Inputs:
  - image -> numpy nd-array of dim (m, n, c) for RGB images or numpy nd-array of dim (m, n) for gray scale images
  - filter -> numpy nd-array of odd dim (k, l)
  Returns
  - filtered_image -> numpy nd-array of dim (m, n, c) or numpy nd-array of dim (m, n)
  Errors if:
  - filter has any even dimension -> raise an Exception with a suitable error message.
  """
  filtered_image = np.asarray([0])

  ##################
  # Your code here #
  raise NotImplementedError('my_imfilter function in helpers.py needs to be implemented')
  ##################

  return filtered_image
'''
def image_pad(image: np.ndarray , paddingx,paddingy, pad ='zero'):
  # padding with zero
  # paddingx = 0.5*(R-1)
  # paddingy = 0.5*(L-1)
  if pad == 'zero':
    imagePadded = np.pad(image , [( paddingx , paddingx ),( paddingy , paddingy )] , constant_values = 0 )
  elif pad == 'reflect':
    imagePadded = np.pad(image , [( paddingx , paddingx ),( paddingy , paddingy )] , mode = 'reflect' )
  return imagePadded


def conv_channel(imagePadded: np.ndarray, filter: np.ndarray):  # convolve one channel
  col = imagePadded.shape[1] - filter.shape[1] + 1
  row = imagePadded.shape[0] - filter.shape[0] + 1  # restore the original dim
  conv_result = np.zeros((row, col))

  R = filter.shape[0]  # number of rows
  L = filter.shape[1]  # number of col

  for y in range(col):
    # checking for the range of y -----> should not be necessary since padding is done
    for x in range(row):
      conv_result[x, y] = (filter * imagePadded[x: x + R, y: y + L]).sum()
  return conv_result




def conv_fft(imageP: np.ndarray, filter: np.ndarray, N, M):  # convolve one channel
  # fft2 for the kernel

  R = filter.shape[0]  # number of rows
  L = filter.shape[1]  # number of col
  filter_p = np.zeros((N, M))
  filter_p[0:R, 0:L] = filter
  kernel_freq = fft2(filter_p)
  paddingx = int(0.5 * (R - 1))
  paddingy = int(0.5 * (L - 1))

  # fft2 for the image
  image_p = np.zeros((N, M))
  image_p[0:imageP.shape[0], 0:imageP.shape[1]] = imageP
  image_freq = fft2(image_p)
  # muliply both elementwise multiplication
  conv = fftshift(image_freq) * fftshift(kernel_freq)
  # ifft2 to the product
  conv = ifftshift(conv)
  result = np.real(ifft2(conv))
  ############################################################################################################
  result = result.astype(np.float32)
  #result = result.astype(np.uint8)
  # print(result)
  #############################################################################################################
  return result[paddingx:imageP.shape[0] + paddingx, paddingy:imageP.shape[1] + paddingy]


def my_imfilter(image: np.ndarray, filter: np.ndarray, pad='zero', conv='normal'):
  R = filter.shape[0]  # number of rows
  L = filter.shape[1]  # number of col

  assert filter.shape[0] % 2 == 1 and filter.shape[1] % 2 == 1, 'Filter Dimensions have to be odd vaules'

  # flip the Kernel ----> rotate by 180
  filter = np.flip(filter)  # flip vertically and horizontally

  # Pad the input image with zeros.
  paddingx = int(0.5 * (R - 1))
  paddingy = int(0.5 * (L - 1))

  # Support grayscale and color images:
  # could a dim (m,n,) cause a problem ------>>????????????????????/

  filtered_image = np.empty(shape=image.shape)  # creating a np array with the same shape as the input image

  if len(image.shape) < 3:  # grayScale
    if conv == 'fft':
      #print('fft')
      imageconv = conv_fft(image[:, :], filter, image.shape[0] + R - 1, image.shape[1] + L - 1)
      filtered_image[:, :] = imageconv
    else:
      imagePadded = image_pad(image[:, :], paddingx, paddingy, pad)
      imageconv = conv_channel(imagePadded, filter)
      filtered_image[:, :] = imageconv
  else:  # colored images
    for i in range(image.shape[2]):  # for each channel
      if conv == 'fft':
        #print("fft")
        imageconv = conv_fft(image[:, :, i], filter, image.shape[0] + R - 1, image.shape[1] + L - 1)
        filtered_image[:, :, i] = imageconv
      else:
        imagePadded = image_pad(image[:, :, i], paddingx, paddingy, pad)
        imageconv = conv_channel(imagePadded, filter)
        filtered_image[:, :, i] = imageconv

  return filtered_image

def create_gaussian_filter(cutoff_frequency):
  ksize = (cutoff_frequency * 4 + 1, cutoff_frequency * 4 + 1)
  L, W = ksize  # get the length and width of the ksize
  sigma = cutoff_frequency
  arr = np.linspace(-(L - 1) / 2, (L - 1) / 2, L)
  x, y = np.meshgrid(arr, arr)
  kernel = np.exp(-0.5 * (np.square(x) + np.square(y)) / np.square(sigma))
  kernel = kernel / np.sum(kernel)
  return kernel

def gen_hybrid_image(image1: np.ndarray, image2: np.ndarray, cutoff_frequency: float):
  """
   Inputs:
   - image1 -> The image from which to take the low frequencies.
   - image2 -> The image from which to take the high frequencies.
   - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian
                         blur that will remove high frequencies.

   Task:
   - Use my_imfilter to create 'low_frequencies' and 'high_frequencies'.
   - Combine them to create 'hybrid_image'.
  """

  assert image1.shape == image2.shape


  # Steps:
  # (1) Remove the high frequencies from image1 by blurring it. The amount of
  #     blur that works best will vary with different image pairs
  # generate a gaussian kernel with mean=0 and sigma = cutoff_frequency,
  # Just a heads up but think how you can generate 2D gaussian kernel from 1D gaussian kernel

  # Your code here:
  # Gaussian filter:
  low_kernel = create_gaussian_filter(cutoff_frequency)
  low_frequencies = my_imfilter(image1, low_kernel);# Replace with your implementation
  #print('low pass filter', low_frequencies)


  # (2) Remove the low frequencies from image2. The easiest way to do this is to
  #     subtract a blurred version of image2 from the original version of image2.
  #     This will give you an image centered at zero with negative values.
  # Your code here #
  # Gaussian filter:
  high_kernel = create_gaussian_filter(cutoff_frequency+2)
  high_frequencies = image2 - my_imfilter(image2, high_kernel);

  # cliping the high pass image to  get kernel from range 0 to 1 so it can be saved on the disk
  high_frequencies = np.clip(high_frequencies+0.5, 0.0,1.0)
  #print('high pass filter', high_frequencies)

  # (3) Combine the high frequencies and low frequencies
  # Your code here #
  hybrid_image = low_frequencies + high_frequencies # Replace with your implementation



  # (4) At this point, you need to be aware that values larger than 1.0
  # or less than 0.0 may cause issues in the functions in Python for saving
  # images to disk. These are called in proj1_part2 after the call to 
  # gen_hybrid_image().
  # One option is to clip (also called clamp) all values below 0.0 to 0.0, 
  # and all values larger than 1.0 to 1.0.
  hybrid_image = np.clip(hybrid_image-0.5,0.0,1.0)

  # (5) As a good software development practice you may add some checks (assertions) for the shapes
  # and ranges of your results. This can be performed as test for the code during development or even
  # at production!
  # subtracting 0.5 from the high freq image to (substitue) the adding 0.5 in proj1_part2.py
  high_frequencies =high_frequencies-0.5


  return low_frequencies, high_frequencies, hybrid_image

def vis_hybrid_image(hybrid_image: np.ndarray):
  """
  Visualize a hybrid image by progressively downsampling the image and
  concatenating all of the images together.
  """
  scales = 5
  scale_factor = 0.5
  padding = 5
  original_height = hybrid_image.shape[0]
  num_colors = 1 if hybrid_image.ndim == 2 else 3

  output = np.copy(hybrid_image)
  cur_image = np.copy(hybrid_image)
  for scale in range(2, scales+1):
    # add padding
    output = np.hstack((output, np.ones((original_height, padding, num_colors),
                                        dtype=np.float32)))
    # downsample image
    cur_image = rescale(cur_image,scale_factor, mode='reflect',multichannel=True)
    # pad the top to append to the output
    pad = np.ones((original_height-cur_image.shape[0], cur_image.shape[1], num_colors), dtype=np.float32)
    tmp = np.vstack((pad, cur_image))
    output = np.hstack((output, tmp))
  return output

def load_image(path):
  return img_as_float32(io.imread(path))

def load_image_gray(path):
  return img_as_float32(io.imread(path, as_gray=True))

def save_image(path, im):
  return io.imsave(path, img_as_ubyte(im.copy()))
