import cv2
import numpy as np
from AllFunctions.homography import computeHomography
import pdb


def computeHomographyRansac(img1, img2, matches, iterations, threshold):
    """
    RANSAC algorithm.

    Input: img1
           img2
           matches
           iterations
           threshold

    Output: bestH - return best homography matrix

    """

    # Extract all 'feature match points' from both images
    points1 = []
    points2 = []
    for i in range(len(matches)):
        # retrieves the points (x, y) coordinates of the keypoints in the second image (img2) that correspond to the keypoints in the first image (img1) as determined by the matches
        # 'matches' is a list of match objects that represents the correspondences between keypoints in two images
        # '.queryIdx' is an attribute of the match object that stores the index of the corresponding keypoint in first image for this match
        # '.pt' is used to access the (x,y) coordinates of the keypoint in the img2 that corresponds to the keypoint in the img1 specified by 'matches[i].queryIdx'
        points1.append(img1['keypoints'][matches[i].queryIdx].pt) # matches are DMatch objects (see its attributes in internet)
        points2.append(img2['keypoints'][matches[i].trainIdx].pt)

    bestInlierCount = 0
    for i in range(iterations):

        # Step 1: Construct the subsets by randomly choosing 4 matches
        subset1 = []
        subset2 = []
        for _ in range(4):
            idx = np.random.randint(0, len(points1)-1)
            subset1.append(points1[idx])
            subset2.append(points2[idx])

        # step 2: Compute homography of step 1 subsets
        H = computeHomography(subset1, subset2)

        # Step 3: Count the number of inliners for the current H
        inlinerCount = numInliers(np.array(points1).T, np.array(points2).T, H, threshold)

        # Step 4: Keep track of the best Homography
        if inlinerCount > bestInlierCount:
            bestInlierCount = inlinerCount
            bestH = H

    print(f"({img1['id']},{img2['id']}) found {bestInlierCount} RANSAC inliers")
    return bestH


def numInliers(points1, points2, H, threshold):
    """
    Computes the number of inliers for the given homography.

    Inputs:
           points1
           points2
           H
           threshold

    Output: inliearCount

    - Project the image points from image 1 to image 2
    - A point is an inlier if the distance between the projected point and
        the point in image 2 is smaller than threshold
    ## Hint: Construct a Homogeneous point of type 'Vec3' before applying H.
    
    """

    inlierCount = 0
    # Create a homogoneous representations of points ba vertically stacking one row of ones.
    points1_homog = np.vstack((points1, np.ones((1, points1.shape[1]))))
    points2_homog = np.vstack((points2, np.ones((1, points2.shape[1]))))

    # Project points1 from img1 to img2 using Homography H
    points2_estimate_homog = H @ points1_homog

    # Convert estimated points back into non-homogenous coordinates
    points2_estimate = points2_estimate_homog / points2_estimate_homog[-1, :]

    # Calculate Euclidian distances between esitmated and actual points
    distance_vector = np.sqrt(np.sum((points2_estimate - points2_homog) ** 2, axis=0))

    # Count the number of inliers by ckecking threshold
    inlierCount = np.sum(distance_vector < threshold)

    return inlierCount






