import cv2 
import numpy as np

def filteredGradient(im, sigma):
    # Computes the smoothed horizontal and vertical gradient images for a given
    # input image and standard deviation. The convolution operation should use
    # the default border handling provided by cv2.
    #
    # im: 2D float32 array with shape (height, width). The input image.
    # sigma: double. The standard deviation of the gaussian blur kernel.

    # Returns:
    # Fx: 2D double array with shape (height, width). The horizontal
    #     gradients.
    # Fy: 2D double array with shape (height, width). The vertical
    #     gradients.
    
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255
    half_width = 3*sigma
    ax = np.linspace(-half_width-1, half_width+1, half_width*2+1)
    gauss = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(ax**2 / (2 * sigma**2)))
    dgx = np.expand_dims(gauss[1:] - gauss[0:-1],axis=1)
    dgy = dgx.T
    Fx = cv2.filter2D(im, -1, dgx)
    Fy = cv2.filter2D(im, -1, dgy)
    
    return Fx, Fy


def edgeStrengthAndOrientation(Fx, Fy):
    # Given horizontal and vertical gradients for an image, computes the edge
    # strength and orientation images.
    #
    # Fx: 2D double array with shape (height, width). The horizontal gradients.
    # Fy: 2D double array with shape (height, width). The vertical gradients.

    # Returns:
    # F: 2D double array with shape (height, width). The edge strength
    #        image.
    # D: 2D double array with shape (height, width). The edge orientation
    #        image.
    
    F = np.sqrt(Fx**2 + Fy**2)
    D = np.arctan(Fy, Fx)
    return F, D


def suppression(F, D):
    # Runs nonmaximum suppression to create a thinned edge image.
    #
    # F: 2D double array with shape (height, width). The edge strength values
    #    for the input image.
    # D: 2D double array with shape (height, width). The edge orientation
    #    values for the input image.

    # Returns:
    # I: 2D double array with shape (height, width). The output thinned
    #        edge image.
    D2 = np.round(D / np.pi * 4).astype(int)
    orientations = np.arange(-1, 1)
    I = np.copy(F)

    x, y = np.indices(D2.shape)
    for i in orientations:
        for j in orientations:
            mask = (D2 == D2[x + i, y + j]) & (F < F[x + i, y + j])
            I[mask] = 0
    
    return I


def hysteresisThresholding(I, D, tL, tH):
    # Runs hysteresis thresholding on the input image.

    # I: 2D double array with shape (height, width). The input's edge image
    #    after thinning with nonmaximum suppression.
    # D: 2D double array with shape (height, width). The edge orientation
    #    image.
    # tL: double. The low threshold for detection.
    # tH: double. The high threshold for detection.

    # Returns:
    # edgeMap: 2D binary array with shape (height, width). Output edge map,
    #          where edges are 1 and other pixels are 0. 

    return edgeMap

def cannyEdgeDetection(im, sigma, tL, tH):
    # Runs the canny edge detector on the input image. This function should
    # not duplicate your implementations of the edge detector components. It
    # should just call the provided helper functions, which you fill in.
    #
    # IMPORTANT: We have broken up the code this way so that you can get
    # better partial credit if there is a bug in the implementation. Make sure
    # that all of the work the algorithm does is in the proper helper
    # functions, and do not change any of the provided interfaces. You
    # shouldn't need to create any new .py files, unless they are for testing
    # these provided functions.
    # 
    # im: 2D double array with shape (height, width). The input image.
    # sigma: double. The standard deviation of the gaussian blur kernel.
    # tL: double. The low threshold for detection.
    # tH: double. The high threshold for detection.

    # Returns:
    # edgeMap: 2D binary image of shape (height, width). Output edge map,
    #          where edges are 1 and other pixels are 0.


    # TODO: Implement me!

    return edgeMap
