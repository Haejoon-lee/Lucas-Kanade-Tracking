import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2

# def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
#     """
#     :param It: template image
#     :param It1: Current image
#     :param rect: Current position of the car (top left, bot right coordinates)
#     :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
#     :param num_iters: number of iterations of the optimization
#     :param p0: Initial movement vector [dp_x0, dp_y0]
#     :return: p: movement vector [dp_x, dp_y]
#     """
	
#     # Put your implementation here
#     # set up the threshold
#     ################### TODO Implement Lucas Kanade ###################
    
#     return p


def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
	
    # Put your implementation here
    # set up the threshold
    ################### TODO Implement Lucas Kanade ###################
    H_It, W_It = It.shape
    
    x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3] #writen in image coordinates, not matrix
    H_rect, W_rect = y2 - y1, x2 - x1 

    It1_y, It1_x = np.gradient(It1) #Gradient of current img

    x_axis_rect = np.linspace(x1, x2, int(W_rect))
    y_axis_rect = np.linspace(y1, y2, int(H_rect))
    X_grid_rect, Y_grid_rect = np.meshgrid(x_axis_rect, y_axis_rect)
    
    x = np.arange(0, W_It, 1)
    y = np.arange(0, H_It, 1)   
    spline_It = RectBivariateSpline(y, x, It) #Spline interpolation over a rectangular mesh
    spline_It1 = RectBivariateSpline(y, x, It1)
    spline_It1_x = RectBivariateSpline(y, x, It1_x)
    spline_It1_y = RectBivariateSpline(y, x, It1_y)
    template_rect = spline_It.ev(Y_grid_rect, X_grid_rect) #Bring template_rect from It

    # Iterations for finding optimal dp
    p = p0
    dp = [[1000], [1000]] #Big dp for while loop
    counter = 1
    while np.square(dp).sum() > threshold and counter <= num_iters:

        # translate the rectangle
        x1_tr, y1_tr, x2_tr, y2_tr = x1+p[0], y1+p[1], x2+p[0], y2+p[1]

        x_axis_rect_tr = np.linspace(x1_tr, x2_tr, int(W_rect))
        y_axis_rect_tr = np.linspace(y1_tr, y2_tr, int(H_rect))
        X_grid_rect_tr, Y_grid_rect_tr = np.meshgrid(x_axis_rect_tr, y_axis_rect_tr)

        spline_It1_tr = spline_It1.ev(Y_grid_rect_tr, X_grid_rect_tr)

        #A, b
        It1_rect_tr_x = spline_It1_x.ev(Y_grid_rect_tr, X_grid_rect_tr)
        It1_rect_tr_y = spline_It1_y.ev(Y_grid_rect_tr, X_grid_rect_tr)
        A = np.vstack((It1_rect_tr_x.ravel(),It1_rect_tr_y.ravel())).T

        b = (template_rect - spline_It1_tr).reshape(-1, 1)
        b = b.reshape(-1,1) #Columnize

        #Solve argmin|Ax-b|^2 for finding dp
        dp = np.linalg.inv(A.T@A) @ (A.T) @ b

        #update parameters
        p[0] += dp[0,0]
        p[1] += dp[1,0]

        counter += 1
        
    return p

def LucasKanade_difftemp(It, It1, rect_temp, rect_cur, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect_cur: Current position of the car (top left, bot right coordinates)
    :param rect_temp: Template rec position in the template image (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
	
    # Put your implementation here
    # set up the threshold
    ################### TODO Implement Lucas Kanade ###################
    H_It, W_It = It.shape
    H_It1, W_It1 = It1.shape
    
    x1, y1, x2, y2 = rect_cur[0], rect_cur[1], rect_cur[2], rect_cur[3] #writen in image coordinates, not matrix
    xt1, yt1, xt2, yt2 = rect_temp[0], rect_temp[1], rect_temp[2], rect_temp[3] #writen in image coordinates, not matrix

    H_rect, W_rect = y2 - y1, x2 - x1
    H_rect_temp, W_rect_temp = yt2 - yt1, xt2 - xt1

    It1_y, It1_x = np.gradient(It1) #Gradient of current img

#     x_axis_rect = np.linspace(x1, x2, int(W_rect))
#     y_axis_rect = np.linspace(y1, y2, int(H_rect))
#     X_grid_rect, Y_grid_rect = np.meshgrid(x_axis_rect, y_axis_rect)
    
    x_axis_rect_temp = np.linspace(xt1, xt2, int(round(W_rect_temp)))
    y_axis_rect_temp = np.linspace(yt1, yt2, int(round(H_rect_temp)))
    X_grid_rect_temp, Y_grid_rect_temp = np.meshgrid(x_axis_rect_temp, y_axis_rect_temp)
    

    xt = np.arange(0, W_It, 1)
    yt = np.arange(0, H_It, 1)
    x = np.arange(0, W_It1, 1)
    y = np.arange(0, H_It1, 1)
    spline_It = RectBivariateSpline(yt, xt, It) #Spline interpolation over a rectangular mesh
    spline_It1 = RectBivariateSpline(y, x, It1)
    spline_It1_x = RectBivariateSpline(y, x, It1_x)
    spline_It1_y = RectBivariateSpline(y, x, It1_y)
    
    template_rect = spline_It.ev(Y_grid_rect_temp, X_grid_rect_temp) #Bring template_rect from It

    # Iterations for finding optimal dp
    p = p0
    dp = [[1000], [1000]] #Big dp for while loop
    counter = 1
    while np.square(dp).sum() > threshold and counter <= num_iters:

        # translate the rectangle
        x1_tr, y1_tr, x2_tr, y2_tr = x1+p[0], y1+p[1], x2+p[0], y2+p[1]

        x_axis_rect_tr = np.linspace(x1_tr, x2_tr, int(round(W_rect)))
        y_axis_rect_tr = np.linspace(y1_tr, y2_tr, int(round(H_rect)))
        X_grid_rect_tr, Y_grid_rect_tr = np.meshgrid(x_axis_rect_tr, y_axis_rect_tr)

        spline_It1_tr = spline_It1.ev(Y_grid_rect_tr, X_grid_rect_tr)

        #A, b
        It1_rect_tr_x = spline_It1_x.ev(Y_grid_rect_tr, X_grid_rect_tr)
        It1_rect_tr_y = spline_It1_y.ev(Y_grid_rect_tr, X_grid_rect_tr)
        A = np.vstack((It1_rect_tr_x.ravel(),It1_rect_tr_y.ravel())).T

        b = (template_rect - spline_It1_tr).reshape(-1, 1)
        b = b.reshape(-1,1) #Columnize

        #Solve argmin|Ax-b|^2 for finding dp
        dp = np.linalg.inv(A.T@A) @ (A.T) @ b

        #update parameters
        p[0] += dp[0,0]
        p[1] += dp[1,0]

        counter += 1
        
    return p

