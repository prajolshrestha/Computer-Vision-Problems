import numpy as np
import cv2

# Test Homography
def testHomography():
    
    points1 = [(1, 1), (3, 7), (2, -5), (10, 11)]
    points2 = [(25, 156), (51, -83), (-144, 5), (345, 15)]

    H = computeHomography(points1, points2)

    print("Testing Homography ...")
    print(f"Your result: {H}")

    Href = np.array([[-151.2372466105457,   36.67990057507507,   130.7447340624461],
                 [-27.31264543681857,   10.22762978292494,   118.0943169422209],
                 [-0.04233528054472634, -0.3101691983762523, 1]])
    
    print(f"Reference: {Href}")

    # Compute Error
    error = Href - H
    e = np.linalg.norm(error)
    print(f"Error: {e}")

    if (e < 1e-10):
        print("Test: SUCCESS!")
    else:
        print("Test: Fail!")
    print("======================================")


## Compute a homography matrix from 4 point matches
def computeHomography(points1, points2):
    """
    Compute homography matrix from 4 point matches

    Input: 
          points1:- list of 4 points (tuple)
          points2:- list of 4 points (tuple)

    Output:
          H :- Computed Homography
    """

    # Check input has 4 values
    assert(len(points1) == 4)
    assert(len(points2) == 4)

    # Step 1: Initialize 8x9 matrix A & populate it based on the formula from the manual sheet.
    A = np.zeros((8,9))

    for i in range(len(points1)):
        A[i*2 : i*2+2] = np.array([[-points1[i][0], -points1[i][1], -1, 0,0,0,
                                    points1[i][0]*points2[i][0], points1[i][1]*points2[i][0], points2[i][0]],
                                    [0,0,0, -points1[i][0], -points1[i][1], -1,
                                     points1[i][0]*points2[i][1], points1[i][1]*points2[i][1], points2[i][1]]])
        
    # Step 2: Compute SVD of A = UsV'
    U,s,V_T = np.linalg.svd(A, full_matrices=True)
    V = np.transpose(V_T)

    # Step 3,4: Create matrix H using last column of V
    H = V[:,-1].reshape(3,3)

    # Step 5: Normalize H by 1/h8 (where h8 is last value of the matrix H)
    H /= V[:,-1][-1]

    return H

