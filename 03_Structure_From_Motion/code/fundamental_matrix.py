import numpy as np
import cv2

def computeF(points1, points2):
    
    assert (len(points1)==8), "Length of points1 should be 8!"
    assert (len(points2)==8), "Length of points2 should be 8!"

    # Step 1: Create matrix A
    A = np.zeros((8,9)).astype('int')
    for i in range(8):

        px1, py1 = points1[i]
        qx1, qy1 = points2[i]
        A[i] = [px1 * qx1, px1 * qy1, px1, py1 * qx1, py1 * qy1, py1, qx1, qy1, 1]

        #A[i] = np.array([[points1[i][0]*points2[i][0], points1[i][0]*points2[i][1], points1[i][0], points1[i][1]*points2[i][0],
        #                           points1[i][1]*points2[i][1], points1[i][1], points2[i][0], points2[i][1], 1]])


    # Step 2: Solve for Af = 0 ie, SVD
    _,_,V = np.linalg.svd(A, full_matrices=True)
    V = np.transpose(V)
    F = V[:,-1].reshape(3,3)

    # Step 3: Enforce Rank(F) = 2
    U,s,V_T = np.linalg.svd(F)  
    s[2] = 0.0
   
    F = np.dot(U, np.dot(np.diag(s), V_T)) # Compute F with updated s
   

    # Step 4: Normalize F by 1/f8
    F = F * (1.0 / F[2, 2])

    		

    #points1 = np.array(points1)
    #points2 = np.array(points2)
    #G, mask = cv2.findFundamentalMat(points1, points2, cv2.RANSAC, 1)
   

    return F



def epipolarConstraint(p1, p2, F, t):

    p1h = np.array([p1[0], p1[1], 1])
    p2h = np.array([p2[0], p2[1], 1])

    # Step 1: Compute normalized epipolar line
    #e2 = np.dot(F, p1h)
    #e2 /= np.linalg.norm(e2[:2])
    l1 = np.dot(F,p1h)
    l2 = l1 / np.sqrt(l1[0]**2 + l1[1]**2)

    # Step 2: Compute distance to the epipolar line (ie, point point2 and line e2)
    #distance = np.abs(np.dot(e2, p2h))
    distance = np.abs(np.dot(np.transpose(p2h),l2))

    # Step 3: Check if the distance is smaller than t
    if distance < t:
        return True
    else: 
        return False     



def numInliers(points1, points2, F, threshold):

    inliers = []
    for i in range(len(points1)):
        if (epipolarConstraint(points1[i], points2[i], F, threshold)):
            inliers.append(i)

    return inliers


def computeFRANSAC(points1, points2):

    # The best fundamental matrix and the number of inliers for this F:
    bestInlierCount = 0
    threshold = 4
    iterations = 10000

    for k in range(iterations):
        if k % 1000 == 0:
            print(f'{k} iteraions done.')
        subset1 = []
        subset2 = []
        for i in range(8):
            x = np.random.randint(0, len(points1) - 1)
            subset1.append(points1[x])
            subset2.append(points2[x])
        
        F = computeF(subset1, subset2)
        num = numInliers(points1, points2, F, threshold)
        if (len(num) > bestInlierCount):
            bestF = F
            bestInlierCount = len(num)
            bestInliers = num

    return (bestF, bestInliers)

def testFundamentalMat():
    points1 = [(1, 1), (3, 7), (2, -5), (10, 11), (11, 2), (-3, 14), (236, -514), (-5, 1)]
    points2 = [(25, 156), (51, -83), (-144, 5), (345, 15), (215, -156), (151, 83), (1544, 15), (451, -55)]

    F = computeF(points1, points2)

    print ("Testing Fundamental Matrix...")
    print ("Your result:" + str(F))

    Href = np.array([[0.001260822171230067,  0.0001614643951166923, -0.001447955678643285],
                 [-0.002080014358205309, -0.002981504896782918, 0.004626528742122177],
                 [-0.8665185546662642,   -0.1168790312603214,   1]])

    print ("Reference: " + str(Href))

    error = Href - F
    e = np.linalg.norm(error)
    print ("Error: " + str(e))

    if (e < 1e-10):
        print ("Test: SUCCESS!")
    else:
        print ("Test: FAIL!")
    print ("============================")


testFundamentalMat()