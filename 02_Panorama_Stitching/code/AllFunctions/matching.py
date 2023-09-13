import pdb
import cv2
import numpy as np
#from AllFunctions.utils import createMatchImage


def computeMatches(img1, img2):

    knnmatches = matchknn2(img1, img2)
    matches = ratioTest(knnmatches, 0.7)
    print(f"({img1['id']},{img2['id']}) found {len(matches)} matches.")
    return matches

def matchknn2(img1, img2):
    """
    K-Nearest Neighbor Search
    
    Input: descriptors1: ORB feature descriptors of the image 1
           descriptors1.shape = (num_features, 32)

           descriptors2: ORB feature descriptors of the image 2
           descriptors2.shape = (num_features, 32)

    Output: a List of DMatch objects of nearest and second nearest neighbors of 
            descriptor of image 1 in that of image 2.



    Finds the two nearest neighbors for every descriptor in image1.
    i.e, The smallest and second smallest Hamming distance.

    Store the best match (smallest distance) in knnmatches[i][0]
    and second bect match in knnmatches[i][1].
    """

    descriptors1 = img1['descriptors']
    descriptors2 = img2['descriptors']

    
    # Create an instance of BFMatcher
    match = cv2.BFMatcher()
    matches = match.knnMatch(descriptors1, descriptors2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.95 * n.distance:
            good.append([m])

    
    draw_params = dict(matchColor=(0,255,0),
                       singlePointColor=None,
                       flags = 2)
    
    img3 = cv2.drawMatchesKnn(img1['img'], img1['keypoints'], img2['img'], img2['keypoints'], good, None,
                           matchesMask= None, **draw_params)
    #cv2.imshow("Original_Image_drawMatches",img3)
    #cv2.waitKey(0)

    # Find the two nearest neighbours for every descriptor in image 1
    # Initialize empty list of matches (N x 2)
    #print(descriptors1.shape)
    knnmatches = []
    for i in range(descriptors1.shape[0]):
        distance = []
        for j in range(descriptors2.shape[0]):

            distance.append(cv2.norm(descriptors1[i], descriptors2[j], cv2.NORM_HAMMING))
        
        distance = np.asarray(distance)
        # Find first nearest neighbor of "descriptor of image 1" in that of "descriptor of image 2"
        dm1 = cv2.DMatch(i,np.argmin(distance), np.min(distance))
        
        # Find second nearest neighbor
        distance_sorted = np.sort(distance)
        dm2 = cv2.DMatch(i, np.argwhere(distance == distance_sorted[1])[0,0], distance_sorted[1])
        knnmatches.append([dm1, dm2])

    return knnmatches



def ratioTest(knnmatches, ratio_threshold):
    """
    Outlier Removel
    Compute the ratio between the nearest and second nearest neighbor
    Add the nearest neighbor to the output matches if the ratio is smaller than ratio threshold.
    """
    matches = []
    for distances in knnmatches:
        # Ratio
        ratio = distances[0].distance / distances[1].distance
        if ratio < ratio_threshold:
            matches.append(distances[0])
    return matches

