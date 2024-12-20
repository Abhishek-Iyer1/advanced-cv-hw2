import numpy as np
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import affine_transform
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine
import cv2

def SubtractDominantMotion(image1, image2, M, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    
    # put your implementation here
    mask = np.ones(image1.shape, dtype=bool)

    ################### TODO Implement Substract Dominant Motion ###################
    M = M[:2]
    # Apply the affine transformation to the image (image1 @ M)
    image1_warped = cv2.warpAffine(image1, M, (image1.shape[1], image1.shape[0]))

    # Figure out ROI as some pixels will be out of bounds

    # Get the difference between the two images
    diff = image2 - image1_warped

    # Threshold the difference to get the mask
    mask = np.abs(diff) > tolerance

    # Dilate the mask to fill in the holes
    mask = binary_dilation(mask, structure=np.ones((5, 5)))

    # Erode the mask to remove noise
    mask = binary_erosion(mask, structure=np.ones((3, 3)))

    return mask.astype(bool)
