import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform
import cv2

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """
    ################### TODO Implement Inverse Composition Affine ###################

    # put your implementation here
    M0 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    M0  = np.vstack([M0, [0, 0, 1]]) # 3 x 3

    # Calculate the gradient of the It1
    Itx = cv2.Sobel(It, cv2.CV_64F, 1, 0, ksize=3)
    Ity = cv2.Sobel(It, cv2.CV_64F, 0, 1, ksize=3)

    # Create splines to interpolate the values of the image at the new location
    rows, cols = np.arange(It.shape[0]), np.arange(It.shape[1])

    It1_spline = RectBivariateSpline(rows, cols, It1)

    x1, y1, x2, y2 = 0, 0, It.shape[1], It.shape[0]
    x = np.arange(x1, x2)
    y = np.arange(y1, y2)
    x_meshgrid, y_meshgrid = np.meshgrid(x, y)


    # Jacobian = 2 x 6 x N^2
    jacobian = np.array([[x_meshgrid.flatten(), y_meshgrid.flatten(), np.ones_like(x_meshgrid.flatten()), np.zeros_like(x_meshgrid.flatten()), np.zeros_like(x_meshgrid.flatten()), np.zeros_like(x_meshgrid.flatten())],
                         [np.zeros_like(x_meshgrid.flatten()), np.zeros_like(x_meshgrid.flatten()), np.zeros_like(x_meshgrid.flatten()), x_meshgrid.flatten(), y_meshgrid.flatten(), np.ones_like(x_meshgrid.flatten())]])
    
    jacobian = np.transpose(jacobian, (2, 0, 1)) # N^2 x 2 x 6
    nabla_T = np.array([Itx.flatten(), Ity.flatten()]).T

    nabla_T = np.expand_dims(nabla_T, axis=-1)
    nabla_T = np.transpose(nabla_T, (0, 2, 1)) # N^2 x 1 x 2

    A = (nabla_T @ jacobian).reshape((-1, 6)) # N^2 x 6
    H = A.T @ A
    H_inv = np.linalg.inv(H)


    for i in range(num_iters):
        homogeneous_coords = np.array([[x_meshgrid.flatten(), y_meshgrid.flatten(), np.ones_like(x_meshgrid.flatten())]]).reshape((3, -1)) # 3 x N^2
        homogeneous_warped = M0 @ homogeneous_coords
        dehomogeneous_warped = homogeneous_warped[:2] / homogeneous_warped[2]
        x_warped_meshgrid, y_warped_meshgrid = dehomogeneous_warped[0].reshape(It.shape), dehomogeneous_warped[1].reshape(It.shape)

        It1_warped = It1_spline.ev(y_warped_meshgrid, x_warped_meshgrid)
        I_diff = It1_warped - It
        b = I_diff.flatten().reshape((-1, 1))
        dp = H_inv @ A.T @ b

        M_delta = np.array([[1 + dp[0].item(), dp[1].item(), dp[2].item()], 
                            [dp[3].item(), 1 + dp[4].item(), dp[5].item()],
                            [0, 0, 1]])
        
        M0 = M0 @ np.linalg.inv(M_delta)

        if np.linalg.norm(dp)**2 < threshold:
            break
                    
    return M0
