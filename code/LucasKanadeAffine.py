import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ################### TODO Implement Lucas Kanade Affine ###################
    # prev_delta_p = np.zeros((2, 1))
    # You have a template image and the current image where current image is the next frame
    # To find b, we need to subtraact the current image from the template image and then query the template image at the new location (x')
    # To find A, we need to calculate the gradient of the current image in x and y and then query the current image at the new location (x')
    # We need to iterate over the number of iterations and update the p until the change in p is less than the threshold
    # You can use scipy.interpolate.RectBivariateSpline to interpolate the values of the image at the new location (x')

    M = np.vstack([M, [0, 0, 1]]) # 3 x 3

    x1, y1, x2, y2 = 0, 0, It.shape[1], It.shape[0]
    p1 = 0; p2 = 0; p3 = 0; p4 = 0; p5 = 0; p6 = 0
    # Get the gradient of the current image
    It1x = cv2.Sobel(It1, cv2.CV_64F, 1, 0, ksize=3)
    It1y = cv2.Sobel(It1, cv2.CV_64F, 0, 1, ksize=3)

    # Create splines in order to get interpolated values
    rows, cols = np.arange(It.shape[0]), np.arange(It.shape[1])

    It_spline = RectBivariateSpline(rows, cols, It)
    It1_spline = RectBivariateSpline(rows, cols, It1)
    It1x_spline = RectBivariateSpline(rows, cols, It1x)
    It1y_spline = RectBivariateSpline(rows, cols, It1y)
    
    x = np.arange(x1, x2)
    y = np.arange(y1, y2)
    x_meshgrid, y_meshgrid = np.meshgrid(x, y) # N X N (257 x 257)
    # Jacobian = 2 x 6 x N^2 (Later we will reshape to N^2 x 2 x 6, think of it as N^2 matrices of 2 x 6)
    jacobian_vectorized = np.array([[x_meshgrid.flatten(), y_meshgrid.flatten(), np.ones_like(x_meshgrid.flatten()), np.zeros_like(x_meshgrid.flatten()), np.zeros_like(x_meshgrid.flatten()), np.zeros_like(x_meshgrid.flatten())],
                                    [np.zeros_like(x_meshgrid.flatten()), np.zeros_like(x_meshgrid.flatten()), np.zeros_like(x_meshgrid.flatten()), x_meshgrid.flatten(), y_meshgrid.flatten(), np.ones_like(x_meshgrid.flatten())]]).T # N^2 x 6 x 2
    
    for i in tqdm(range(num_iters)):
        homogeneous_coords = np.array([[x_meshgrid.flatten(), y_meshgrid.flatten(), np.ones_like(x_meshgrid.flatten())]]).reshape((3, -1)) # 3 x N^2
        homogeneous_warped = M @ homogeneous_coords # 3 x N^2
        dehomogeneous_warped = homogeneous_warped[:2] / homogeneous_warped[2] # 2 x N^2
        
        # Check if the points are within the image
        valid_mask = np.array([[(dehomogeneous_warped[0] >= 0) & (dehomogeneous_warped[0] < It.shape[1])], [(dehomogeneous_warped[1] >= 0) & (dehomogeneous_warped[1] < It.shape[0])]])
        valid_mask = valid_mask.squeeze(axis=1)    # (valid_row_idx, valid_col_idx) x numpxl
        dehomogeneous_warped[~valid_mask] = 0 # 2 x N^2

        #Visualize the resultss
        x_warped, y_warped = dehomogeneous_warped[0], dehomogeneous_warped[1] # 1 X N^2

        # Get the interpolated values
        It_roi = It_spline.ev(y_meshgrid.flatten(), x_meshgrid.flatten()) # N x N
        It1_warped = It1_spline.ev(y_warped, x_warped) # N x N
        It1x_warped = It1x_spline.ev(y_warped, x_warped) # N x N
        It1y_warped = It1y_spline.ev(y_warped, x_warped) # N x N

        # Get the difference between the two images 
        I_diff = It_roi - It1_warped # N x N

        # Construct our equation for delta_p
        # delta_p = H^(-1) * A^T * b, where A = [dW/dp1, dW/dp2, dW/dp3, dW/dp4, dW/dp5, dW/dp6]

        # Nabla I = N^2 x 2 (Later we will reshape to N^2 x 1 x 2, think of it as N^2 matrices of 1 x 2)
        Nabla_I = np.array([It1x_warped.flatten(), It1y_warped.flatten()]).T
        
        jacobian_vectorized_valid = np.transpose(jacobian_vectorized, (0, 2, 1)) # N^2 x 2 x 6
        Nabla_I = Nabla_I.reshape((-1, 1, 2)) # N^2 x 1 x 2

        # Now we can compute the A (N^2 matrices of 1 x 6), reshaped to N^2 x 6 for convenience
        A = Nabla_I @ jacobian_vectorized_valid # N^2 x 1 x 6
        A = A.reshape((-1, 6)) # N^2 x 6

        b = I_diff.flatten().reshape((-1, 1))  # N^2 x 1

        # Compute the Hessian and the Gauss Newton update step
        H = A.T @ A # 6 x 6
        # (6 x 6) x (6 x N^2) x (N^2 x 1) = 6 x 1
        delta_p = (np.linalg.inv(H) @ A.T @ b).reshape(-1, 1) # 6 x 1

        # Update M
        p1 = p1 + delta_p[0].item()
        p2 = p2 + delta_p[1].item()
        p3 = p3 + delta_p[2].item()
        p4 = p4 + delta_p[3].item()
        p5 = p5 + delta_p[4].item()
        p6 = p6 + delta_p[5].item()
        
        M = np.array([[1 + p1, p2, p3], [p4, 1 + p5, p6], [0, 0, 1]])
        if np.linalg.norm(delta_p)**2 < threshold:
            break

    return M
