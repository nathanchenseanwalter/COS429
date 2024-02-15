"""
Author: Donsuk Lee (donlee90@stanford.edu)
Modified by: Vivien Nguyen (vivienn@princeton.edu)
Date created: 09/2017
Last modified: 09/02/2021
Python Version: 3.5+
"""

import numpy as np
from skimage import filters
from skimage.feature import corner_peaks
from skimage.util.shape import view_as_blocks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve

from utils import pad, unpad, get_output_space, warp_image


def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter
        
    Returns:
        response: Harris response image of shape (H, W)
    """

    H, W = img.shape
    window = np.ones((window_size, window_size))

    response = np.zeros((H, W))

    # 1. Compute x and y derivatives (I_x, I_y) of an image
    dx = filters.sobel_v(img)
    dy = filters.sobel_h(img)

    windowed_dx2 = convolve(dx * dx, window, mode='constant')
    windowed_dy2 = convolve(dy * dy, window, mode='constant')
    windowed_dxy = convolve(dx * dy, window, mode='constant')

    M = np.zeros((H, W, 2, 2))
    M[:,:,0,0] = windowed_dx2
    M[:,:,0,1] = windowed_dxy
    M[:,:,1,0] = windowed_dxy
    M[:,:,1,1] = windowed_dy2
    
#     M = convolve(M, window)
    
    det = np.linalg.det(M)
    trace = np.trace(M,axis1=-2,axis2=-1)
    
    response = det - k * np.square(trace)

    return response


def simple_descriptor(patch):
    """
    Describe the patch by normalizing the image values into a standard
    normal distribution (having mean of 0 and standard deviation of 1)
    and then flattening into a 1D array.
    The normalization will make the descriptor more robust to change
    in lighting condition.

    Args:
        patch: grayscale image patch of shape (H, W)
    Returns:
        feature: 1D array of shape (H * W)
    """
    feature = []

    patch_mu = patch.mean()
    patch_std = patch.std()
    patch_std = patch_std if patch_std != 0 else 1
    normalized = (patch - patch_mu) / patch_std
    
    feature = normalized.flatten()

    return feature


def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    """
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (y, x) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint
    Returns:
        desc: array of features describing the keypoints
    """

    image.astype(np.float32)
    desc = []

    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[y-(patch_size//2):y+((patch_size+1)//2),
                      x-(patch_size//2):x+((patch_size+1)//2)]
        desc.append(desc_func(patch))
    return np.array(desc)


def match_descriptors(desc1, desc2, threshold=0.5):
    """
    Match the feature descriptors by finding distances between them. A match is formed
    when the distance to the closest vector is much smaller than the distance to the
    second-closest, that is, the ratio of the distances should be strictly smaller
    than the threshold (not equal to). Return the matches as pairs of vector indices.

    Args:
        desc1: an array of shape (M, P) holding descriptors of size P about M keypoints
        desc2: an array of shape (N, P) holding descriptors of size P about N keypoints
    
    Returns:
        matches: an array of shape (Q, 2) where each row holds the indices of one pair
        of matching descriptors
    """
    matches = []

    M = desc1.shape[0]
    dists = cdist(desc1, desc2)

    for i in range(len(dists)):
        l = dists[i]
        if np.sort(l)[0] / np.sort(l)[1] < threshold:
            matches.append([i, np.argmin(l)])
    matches = np.asarray(matches)

    return matches

def fit_affine_matrix(p1, p2):
    """ 
    Fit affine matrix such that p2 * H = p1. First, pad the descriptor vectors
    with a 1 using pad() to convert to homogeneous coordinates, then return
    the least squares fit affine matrix in homogeneous coordinates.
    Hint:
        You can use np.linalg.lstsq function to solve the problem. 
        Explicitly specify np.linalg.lstsq's new default parameter rcond=None 
        to suppress deprecation warnings, and match the autograder.
    Args:
        p1: an array of shape (M, P) holding descriptors of size P about M keypoints
        p2: an array of shape (M, P) holding descriptors of size P about M keypoints
    Return:
        H: a matrix of shape (P+1, P+1) that transforms p2 to p1 in homogeneous
        coordinates
    """

    assert (p1.shape[0] == p2.shape[0]),\
        'Different number of points in p1 and p2'
    p1 = pad(p1)
    p2 = pad(p2)

    H = np.eye(3)
    
    ### YOUR CODE HERE
    H = np.linalg.lstsq(p2, p1, rcond=None)[0]
    ### END YOUR CODE

    # Sometimes numerical issues cause least-squares to produce the last
    # column which is not exactly [0, 0, 1]
    H[:,2] = np.array([0, 0, 1])
    return H


def ransac(keypoints1, keypoints2, matches, n_iters=200, threshold=20):
    """
    Use RANSAC to find a robust affine transformation:
        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers via Euclidean distance
        4. Keep the largest set of inliers
        5. Re-compute least-squares estimate on all of the inliers
    Update max_inliers as a boolean array where True represents the keypoint
    at this index is an inlier, while False represents that it is not an inlier.
    Hint:
        You can use np.linalg.lstsq function to solve the problem. 
        Explicitly specify np.linalg.lstsq's new default parameter rcond=None 
        to suppress deprecation warnings, and match the autograder.
        You can compute elementwise boolean operations between two numpy arrays,
        and use boolean arrays to select array elements by index:
        https://numpy.org/doc/stable/reference/arrays.indexing.html#boolean-array-indexing 
    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        n_iters: the number of iterations RANSAC will run
        threshold: the number of threshold to find inliers
    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints 1
    """
    # Copy matches array, to avoid overwriting it
    orig_matches = matches.copy()
    matches = matches.copy()

    N = matches.shape[0]
    n_samples = int(N * 0.2)

    matched1 = pad(keypoints1[matches[:,0]])
    matched2 = pad(keypoints2[matches[:,1]])

    max_inliers = np.zeros(N, dtype=bool)
    
    H = np.eye(3)

    # RANSAC iteration start
    
    # Note: while there're many ways to do random sampling, please use
    # `np.random.shuffle()` followed by slicing out the first `n_samples`
    # matches here in order to align with the auto-grader.
    # Sample with this code:
    '''
        np.random.shuffle(matches)
        samples = matches[:n_samples]
        sample1 = pad(keypoints1[samples[:,0]])
        sample2 = pad(keypoints2[samples[:,1]])
    '''
    
    ### YOUR CODE HERE
    # intial compute across n_iters
    for _ in range(n_iters):
        np.random.shuffle(matches)
        samples = matches[:n_samples]
        sample1 = pad(keypoints1[samples[:,0]])
        sample2 = pad(keypoints2[samples[:,1]])
        H = np.linalg.lstsq(sample2, sample1, rcond=None)[0]
        H[:,2] = np.array([0, 0, 1])
        inliers = samples[np.linalg.norm(sample2 @ H - sample1, axis=1) < threshold]
        for inlier in inliers:
            max_inliers[np.where(orig_matches == inlier)[0][0]] = True
    
    # recompute for all inliers
    H = np.linalg.lstsq(matched2[max_inliers], matched1[max_inliers], rcond=None)[0]
    H[:,2] = np.array([0, 0, 1])
    inliers = matches[np.linalg.norm(matched2 @ H - matched1, axis=1) < threshold]
    max_inliers = np.zeros(N, dtype=bool)
    for inlier in inliers:
        max_inliers[np.where(orig_matches == inlier)[0][0]] = True
    print('inliers: ',np.where(max_inliers))
    ### END YOUR CODE
    
    return H, orig_matches[max_inliers]



def linear_blend(img1_warped, img2_warped):
    """
    Linearly blend img1_warped and img2_warped by following the steps:
    1. Define left and right margins (already done for you)
    2. Define a weight matrices for img1_warped and img2_warped
        np.linspace and np.tile functions will be useful
    3. Apply the weight matrices to their corresponding images
    4. Combine the images
    Args:
        img1_warped: Refernce image warped into output space
        img2_warped: Transformed image warped into output space
    Returns:
        merged: Merged image in output space
    """
    out_H, out_W = img1_warped.shape # Height and width of output space
    img1_mask = (img1_warped != 0)  # Mask == 1 inside the image
    img2_mask = (img2_warped != 0)  # Mask == 1 inside the image

    # Find column of middle row where warped image 1 ends
    # This is where to end weight mask for warped image 1
    right_margin = out_W - np.argmax(np.fliplr(img1_mask)[out_H//2, :].reshape(1, out_W), 1)[0]

    # Find column of middle row where warped image 2 starts
    # This is where to start weight mask for warped image 2
    left_margin = np.argmax(img2_mask[out_H//2, :].reshape(1, out_W), 1)[0]

    blend = np.linspace(1,0, num=right_margin - left_margin)
    img1_weight = np.concatenate((np.ones(left_margin),blend,np.zeros(out_W - right_margin)))
    img1 = img1_warped * np.tile(img1_weight, (out_H, 1))
    
    blend = np.linspace(0,1,num=right_margin - left_margin)
    img2_weight = np.concatenate((np.zeros(left_margin), blend, np.ones(out_W - right_margin)))
    img2 = img2_warped * np.tile(img2_weight, (out_H, 1))
    
    merged = img1 + img2

    return merged