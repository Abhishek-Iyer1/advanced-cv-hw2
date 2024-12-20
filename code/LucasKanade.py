import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2
from tqdm import tqdm

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros((2, 1))):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
    p = p0.copy().reshape((-1, 1))
    # You have a template image and the current image where current image is the next frame
    # To find b, we need to subtraact the current image from the template image and then query the template image at the new location (x')
    # To find A, we need to calculate the gradient of the current image in x and y and then query the current image at the new location (x')
    # We need to iterate over the number of iterations and update the p until the change in p is less than the threshold
    x1, y1, x2, y2 = rect

    # Get the gradient of the current image
    It1x = cv2.Sobel(It1, cv2.CV_64F, 1, 0, ksize=3)
    It1y = cv2.Sobel(It1, cv2.CV_64F, 0, 1, ksize=3)

    # Create splines in order to get interpolated values
    assert It.shape == It1.shape == It1x.shape == It1y.shape, "All images should be of the same shape"
    rows, cols = np.arange(It.shape[0]), np.arange(It.shape[1])

    It_spline = RectBivariateSpline(rows, cols, It)
    It1_spline = RectBivariateSpline(rows, cols, It1)
    It1x_spline = RectBivariateSpline(rows, cols, It1x)
    It1y_spline = RectBivariateSpline(rows, cols, It1y)

    for i in tqdm(range(num_iters)):
        x = np.arange(x1, x2 + 1)
        y = np.arange(y1, y2 + 1)
        x_warped = x + p[0]
        y_warped = y + p[1]
        x_meshgrid, y_meshgrid = np.meshgrid(x, y)
        x_warped_meshgrid, y_warped_meshgrid = np.meshgrid(x_warped, y_warped)

        # Get the interpolated values
        It_roi = It_spline.ev(y_meshgrid, x_meshgrid)
        It1_warped = It1_spline.ev(y_warped_meshgrid, x_warped_meshgrid)
        It1x_warped = It1x_spline.ev(y_warped_meshgrid, x_warped_meshgrid)
        It1y_warped = It1y_spline.ev(y_warped_meshgrid, x_warped_meshgrid)

        # Get the difference between the two images 
        I_diff = It_roi - It1_warped

        A = np.array([It1x_warped.flatten(), It1y_warped.flatten()]).T # N x 2
        b = I_diff.flatten().reshape((-1, 1)) # N x 1

        # Compute the Hessian and the Gauss Newton update step
        H = A.T @ A # 2 x 2
        delta_p = np.linalg.pinv(H) @ A.T @ b # 2 x 1

        # Update transform parameters, p
        p += delta_p

        if (np.linalg.norm(delta_p)**2 < threshold):
            break

    return p
