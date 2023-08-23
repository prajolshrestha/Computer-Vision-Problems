# A Combined Corner and Edge Detector

import numpy as np
import cv2



def main():

    # Initial Processing
    input_image = cv2.imread('blox.jpg') # Read image
    input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY) # Convert to gray
    input_gray = (input_gray - np.min(input_gray)) / (np.max(input_gray) - np.min(input_gray)) # Normalize
    input_gray = input_gray.astype(np.float32) # Convert to float 32
    #print(input_gray.shape)

    # Harris Edge and corner Detector
    response = harrisResponseImage(input_gray)
    #print(response.shape)

    points = harrisKeypoints(response)
    edges = harrisEdges(input_image, response)


    # Display
    imgKeypoints1 = cv2.drawKeypoints(input_image, points, outImage=None, color = (0, 255, 0))
    show("Harris Keypoints", imgKeypoints1, 1, 2)
    show("Harris Edges", edges, 2, 2)



def harrisResponseImage(img):
    
    # Step 1: Compute spatial Derivatives in x and y direction
    dIdx = cv2.Sobel(img,ddepth=cv2.CV_64F, dx=1, dy= 0, ksize=3)
    dIdy = cv2.Sobel(img,ddepth=cv2.CV_64F, dx=0, dy= 1, ksize=3)
    show("dI/dx", abs(dIdx),1,0)
    show("dI/dy", abs(dIdy),2,0)

    # Step 2: Compute Ixx, Iyy, and Ixy with
    Ixx = dIdx * dIdx
    Iyy = dIdy * dIdy
    Ixy = dIdx * dIdy
    #print(Ixx.shape)
    show("Ixx", abs(dIdx), 0, 1)
    show("Iyy", abs(dIdy), 1, 1)
    show("Ixy", abs(dIdx), 2, 1)
    
    # Step 3: Blur the images Ixx, Iyy, and Ixy with a gaussian filter of size(3,3) and std = 1

    kernelSize = (3,3)
    sdev = 1

    A = cv2.GaussianBlur(Ixx, kernelSize, sdev, sdev)
    B = cv2.GaussianBlur(Iyy, kernelSize, sdev, sdev)
    C = cv2.GaussianBlur(Ixy, kernelSize, sdev, sdev)
    #print(A.shape)
    show("A", abs(A) * 5, 0, 1);
    show("B", abs(B) * 5, 1, 1);
    show("C", abs(C) * 5, 2, 1);

    # Step 4: Compute harris corner response

    # Visualize above mixed product as a matrix
    # matrix = [[A, C],
    #           [C, B]]

    k = 0.06
    Det = A * B - C * C
    Trace = A + B
    response = Det - k * Trace * Trace

    # Normalize the response
    dbg = (response - np.min(response)) / (np.max(response) - np.min(response))
    dbg = dbg.astype(np.float32)
    show("Harris Response", dbg, 0, 2)

    return response

def harrisKeypoints(response, threshold = 0.1):

    height, width = response.shape
    #print(response.shape)
    points = []
    
    # Method 1: With neighboorhood Check
    searchspace = 1
    #for y in range(neighborhood_size // 2, height - neighborhood_size //2):
    #    for x in range(neighborhood_size // 2, width - neighborhood_size // 2):
    for y in range(height):
        for x in range(width):        
            if response[y,x] > threshold:
                is_local_max = True

                # Check if the current pixel is a local maximum within its neighborhood
                for dy in range(-searchspace, searchspace +1):
                    for dx in range(-searchspace, searchspace +1):
                        if (dy != 0 or dx != 0) and response[y,x] < response[y + dy, x + dx]:
                            is_local_max = False
                            break
                    if not is_local_max:
                        break
                if is_local_max:
                    points.append(cv2.KeyPoint(x,y,1))

    # Method 2: Without neighborhood check
    # for y in range(height):
    #     for x in range(width):
    #         if response[y,x] > threshold:

    #             points.append(cv2.KeyPoint(x,y,1))

    return points

def harrisEdges(input, response, edge_threshold = -0.01):

    height, width = input.shape[:2]
    result = input.copy()
    # Set edges to red
    for y in range(height):
        for x in range(width):
            if response[y,x] < edge_threshold:
                is_minimum_x = (response[y,x] < response[y, x-1]) and (response[y,x] < response[y, x+1])
                is_minimum_y = (response[y,x] < response[y-1, x]) and (response[y,x] < response[y+1, x])

                if is_minimum_x or is_minimum_y:
                    result[y,x] = [0,0,255] #REd color
                      
    return result



############################# Display ###########################################

def show(name, img, x, y):
    windowStartX = 10
    windowStartY = 50
    windowXoffset = 5
    windowYoffset = 40

    w = img.shape[0] + windowXoffset
    h = img.shape[1] + windowYoffset

    cv2.namedWindow(name)
    cv2.moveWindow(name, windowStartX + w*x, windowStartY + h*y)
    cv2.imshow(name, img)
    cv2.waitKey(0)




if __name__ == "__main__":
    main()