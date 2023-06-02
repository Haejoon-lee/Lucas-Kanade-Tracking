import numpy as np
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import affine_transform
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
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

    ################### TODO Implement Substract Dominent Motion ###################
    M = LucasKanadeAffine(image1, image2, threshold, num_iters)
    image2_wp = affine_transform(image2, M, output_shape = image1.shape)
    image2_wp = binary_dilation(binary_erosion(image2_wp))  #for better results
    
    abs_diff = np.abs(image1 - image2_wp)
    mask = (abs_diff > tolerance)

    return mask.astype(bool)