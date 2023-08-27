import numpy as np
import cv2, pdb
from math import floor, ceil

def createStichedImage(image_data_dir):
    
    print(f"Stitching with {len(image_data_dir)} images ...")

    ## Calculate the index of the center image in the list
    center = len(image_data_dir) // 2
    ref = image_data_dir[center] # center image 

    ## Compute homographies and update image_data_dir with computed homographies
    image_data_dir = computeHtoref(image_data_dir, center)

    print(f"Reference Image: {center} - " + ref['file'])

    ## Lets Project images to reference image
    minx = 2353535
    maxx = -2353535
    miny = 2353535
    maxy = -2353535
    for i in range(len(image_data_dir)):

        # Take an image
        img2 = image_data_dir[i]
        
        # Initialize a list and store coordinates of img2
        corners2 = [0,0,0,0]
        corners2[0] = (0,0)
        corners2[1] = (img2['img'].shape[1],0)
        corners2[2] = (img2['img'].shape[1], img2['img'].shape[0])
        corners2[3] = (0,img2['img'].shape[0])
        corners2 = np.array(corners2, dtype='float32')#convert to numpy array
        
        # Apply perspective Transform using Homography 'HtoReference' to the 'corners2' array,
        # effectively projecting img2 onto the refrence image
        corners2_in_1 = cv2.perspectiveTransform(corners2[None, :, :], img2['HtoReference'])

        # Loop over projected corner points
        for p in corners2_in_1[0]:
            minx = min(minx, p[0])
            maxx = max(maxx, p[0])
            miny = min(miny, p[1])
            maxy = max(maxy, p[1])
    
    ## Compute Region of Interest based on calculated min and max coordinates of projected corners
    roi = np.array([floor(minx), floor(miny), ceil(maxx)-floor(minx), ceil(maxy)-floor(miny)])
    print(f"ROI: {roi}")

    ## Translate everything so the top left corner is at (0,0)
    # Note: It can be done by adding negative offset to the Homography

    offsetX = floor(minx)
    offsetY = floor(miny)
    ref['HtoReference'][0, 2] = -offsetX
    ref['HtoReference'][1, 2] = -offsetY
    computeHtoref(image_data_dir, center) # update homographies with the new translations

    cv2.namedWindow('Panorama')
    cv2.moveWindow('Panorama', 0, 500)

    # Initialize empty array to populate stitched image using LOOP
    stitchedImage = np.zeros([roi[3], roi[2],3], dtype='uint8')
    for k in range(len(image_data_dir) + 1):
        # check image number to control the order in which images are processed
        if k % 2 == 0:
            tmp = 1
        else: 
            tmp = -1
        # Calculate the index of the image based on center, tmp,k
        i = center + tmp * ((k+1)//2)

        # Out of index bounds check
        if (i < 0 or i >= len(image_data_dir)):
            continue

        ## Project the image onto the refrence image plane
        img2 = image_data_dir[i]
        tmp = np.zeros([roi[3], roi[2], 3])
        
        tmp = cv2.warpPerspective(img2['img'], img2['HtoReference'], (tmp.shape[1],tmp.shape[0]), cv2.INTER_LINEAR)

        ## Added it to the output image
        for y in range(stitchedImage.shape[0]):
            for x in range(stitchedImage.shape[1]):
                if (x < stitchedImage.shape[1] and y < stitchedImage.shape[0] and np.array_equal(stitchedImage[y,x,:], np.array([0,0,0]))):
                    stitchedImage[y,x] = tmp[y,x,:]

        print (f"Added image {i} - {img2['file']}.")
        print ("Press any key to continue...")
        cv2.imshow("Panorama", stitchedImage)
        cv2.waitKey(0)

    return stitchedImage
        




    




        



def computeHtoref(image_data_dir, center):
    """ 
    Compute homography to the reference image

    Input: image_data_dir
           center

    Output: image_data_dir
    """
    
    # Process images before the center image
    for i in range(center-1, -1, -1):
        # Take two images
        c = image_data_dir[i]
        next_ = image_data_dir[i+1]
        
        # Compute homography H2ref = H2ref @ H2next
        c['HtoReference'] = np.matmul(next_['HtoReference'], c['HtoNext'])

    # Process images after the center image
    for i in range(center+1, len(image_data_dir), 1):
        # Take two images
        c = image_data_dir[i]
        prev = image_data_dir[i-1]

        # Compute homographies
        c['HtoReference'] = np.matmul(prev['HtoReference'], c['HtoPrev'])

    return image_data_dir # Return the updated dictonary


