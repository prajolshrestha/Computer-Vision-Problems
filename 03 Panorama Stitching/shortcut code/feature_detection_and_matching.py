import cv2
import numpy as np


## Step 1: Load image
img1 = cv2.imread('Skeleton/data/7.jpg')
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img2 = cv2.imread('Skeleton/data/8.jpg')
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

## Step 2: Feature Detection: Find keypoints and descriptors with SIFT
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)
cv2.imshow('original Image Left Keypoints', cv2.drawKeypoints(img1,kp1, None))
cv2.waitKey(0)

## Step 3: Feature Matching: Find corresponding points between two images
bf = cv2.BFMatcher() # Brute-Force Matcher object
matches = bf.knnMatch(des1,des2, k=2) # returns a list of DMatch objects. 
                                        #It has some attributes (eg. Dmatch.distance, Dmatch.trainIdx, Dmatch.queryIdx, Dmatch.imgIdx)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)

# Show Correspondences on original images
img3 = cv2.drawMatches(img1,kp1,img2, kp2, good, None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("Corresponding Points between two images", img3)
cv2.waitKey(0)

 


