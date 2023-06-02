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

    # put your implementation here
    M0 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    ################### TODO Implement Inverse Composition Affine ###################
    H_It, W_It = It.shape
    
    cor_leftop = np.array([[0, 0, 1]])
    cor_rightbot = np.array([[H_It, W_It, 1]])

    x = np.arange(0, W_It, 1)
    y = np.arange(0, H_It, 1)   
    spline_It = RectBivariateSpline(y, x, It) #Spline interpolation over a rectangular mesh
    spline_It1 = RectBivariateSpline(y, x, It1)

    x_axis = np.linspace(cor_leftop[0,0], cor_rightbot[0,0], int(round(W_It)))
    y_axis = np.linspace(cor_leftop[0,1], cor_rightbot[0,1], int(round(H_It)))
    X_grid, Y_grid = np.meshgrid(x_axis, y_axis)
    template = spline_It.ev(Y_grid, X_grid)

    It_y, It_x = np.gradient(It) #Gradient of template img
    Del_It = np.vstack((It_x.ravel(),It_y.ravel())).T

    M = M0
    M_sq = np.vstack((M, [0, 0, 1]))
    
    #Precompute A here
    A = np.zeros((H_It*W_It, 6))        
    for i in range(H_It):
        for j in range(W_It):
            #I is (1,2) for each pixel
            #Jacobiani is (2,6)for each pixel
            Del_It_point = np.array([Del_It[i*W_It+j]]).reshape(1,2)
            jacob_point = np.array([[j, i, 1, 0, 0, 0], [0, 0, 0, j, i, 1]])
            A[i*W_It+j] = Del_It_point @ jacob_point
    
    H = np.linalg.inv(A.T@A)
    Precomp = H @ A.T
    # Iterations for finding optimal dp
    dp = [[1000], [1000]] #Big dp for while loop
    counter = 1
    while np.square(dp).sum() > threshold and counter <= num_iters:
        
        # Warp Current Image
        cor_leftop_wp = M@cor_leftop.T
        cor_rightbot_wp = M@cor_rightbot.T
        
        x_axis_wp = np.linspace(cor_leftop_wp[0,0], cor_rightbot_wp[0,0], int(round(W_It)))
        y_axis_wp = np.linspace(cor_leftop_wp[1,0], cor_rightbot_wp[1,0], int(round(H_It)))
        X_grid_wp, Y_grid_wp = np.meshgrid(x_axis_wp, y_axis_wp)
        It1_wp = spline_It1.ev(Y_grid_wp, X_grid_wp)
        
        #Get b
        b = (It1_wp - template).reshape(-1,1)

        #Solve argmin|Ax-b|^2 for finding dp
        dp = Precomp @ b
        
        #Updating
        dM = np.array([[1+dp[0,0], dp[1,0], dp[2,0]], [dp[3,0], 1+dp[4,0], dp[5,0]], [0, 0, 1]])  #dM should be square to be inversed
        # M_sq = np.vstack((M, [0, 0, 1]))
        M_sq = M_sq @ np.linalg.inv(dM)
        M = M_sq[:2, :]
        counter += 1
        
    return M