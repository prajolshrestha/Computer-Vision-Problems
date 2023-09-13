from util import computeBilinearWeights
from opticalFlowLK import OpticalFlowLK

from util import *
import numpy as np
import cv2

def test():
    ## Test for bilinear weights
    p = np.array([0.125, 0.82656])
    weights = np.array([0.15176, 0.02168, 0.72324, 0.10332])

    err = computeBilinearWeights(p) - weights
    e = np.linalg.norm(err)

    print("computeBilinearWeights error: " + str(e))
    if e < 1e-6:
        print("Test: SUCCESS!")
    else:
        print("Test: FAIL")

    
    ## Test for Gaussian kernel
    kernel = np.array(
        [[0.1690133273455962, 0.3291930023422986, 0.4111123050281957, 0.3291930023422986, 0.1690133273455962],
         [0.3291930023422986, 0.6411803997536073, 0.8007374099875735, 0.6411803997536073, 0.3291930023422986],
         [0.4111123050281957, 0.8007374099875735, 1, 0.8007374099875735, 0.4111123050281957],
         [0.3291930023422986, 0.6411803997536073, 0.8007374099875735, 0.6411803997536073, 0.3291930023422986],
         [0.1690133273455962, 0.3291930023422986, 0.4111123050281957, 0.3291930023422986, 0.1690133273455962]])
    res = computeGaussianWeights((5, 5), 0.3)
    err = res - kernel
    e = np.linalg.norm(err)
    print("computeGaussianWeights error: " + str(e))
    if e < 1e-6:
        print("Test: SUCCESS!")
    else:
        print("Reference " + str(kernel))
        print("Your result " + str(res))
        print("Test: FAIL")

    ## Tests for matrix inversion and gaussian weights
    A = np.array([[12, 4], [4, 8]])
    Ainv = np.array([[0.1, -0.05], [-0.05, 0.15]])
    err = invertMatrix2x2(A) - Ainv
    e = np.linalg.norm(err)

    print("invertMatrix2x2 error: " + str(e))
    if e < 1e-10:
        print("Test: SUCCESS!")
    else:
        print("Test: FAIL")

    print("=" * 30)


def main():

    #test()
    print('#'*40)
    cap = cv2.VideoCapture('../Skeleton/slow_traffic_small.mp4')

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    last_grey = None
    last_frame = None
    last_keypoints = []
    status = []
    winsize = [25, 25]

    while (cap.isOpened()):

        ## Read Video and do some basic Image Processing
        ret, frame = cap.read()
        scale = 0.7
        frame = cv2.resize(frame, None, fx = scale, fy=scale)
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #print(grey.shape) #(252, 448)
        
        ## Feature Detection
        # Define criteria for corner subpixel refinement
        # Detect corners (keypoint) and refine it
        termcrit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.03)
        subPixWinSize = (10, 10)
        keypoints = cv2.goodFeaturesToTrack(grey, maxCorners=200, qualityLevel=0.01, minDistance=10, blockSize=3, useHarrisDetector=False, k=0.04)
        keypoints = cv2.cornerSubPix(grey, keypoints, subPixWinSize, tuple([-1,-1]), termcrit)
        keypoints = np.array(keypoints)
        keypoints = np.squeeze(keypoints, axis=1)
        #print(keypoints.shape)
       
        
        flowPoints = 0
        if not len(last_keypoints) == 0: # check if there is previously detected keypoints
            
            ## Compute optical Flow between Frames
            # Using your own code
            of = OpticalFlowLK(winsize, 0.03, 20)
            #print(len(last_keypoints)) #139
            points, status = of.compute(last_grey, grey, np.copy(last_keypoints))
            #print(len(points))#139
            
            # Using opencv
            #lk_params = dict( winSize = (25, 25),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            #points, status, err = cv2.calcOpticalFlowPyrLK(last_grey, grey,np.copy(last_keypoints), None, **lk_params)


            # Iterate through the detected keypoints
            for i in range(len(points)):

                if not status[i]:
                    continue
                
                # calculate distance between current and previous keypoint
                diff = points[i] - last_keypoints[i]
                distance = np.linalg.norm(diff)

                # Check if the keypoint have moved significantly
                if distance > 15 or distance < 0.2:
                    continue #skip current iteration and move to next iteration
                
                # Calculate the endpoint of a line indicating motion
                otherP = last_keypoints[i] + diff * 15
                flowPoints += 1 #increment count of keypoints with motion

                # Draw a circle at the current keypoint position and a line indicating motion
                color = tuple([0,255,0])
                cv2.circle(last_frame, (int(last_keypoints[i][0]), int(last_keypoints[i][1])), 1, color)
                cv2.line(last_frame, (int(last_keypoints[i][0]), int(last_keypoints[i][1])), (int(otherP[0]), int(otherP[1])), color)

            cv2.imshow('output', last_frame)
            cv2.waitKey(1)

            print(f'Keypoints moving/total: {flowPoints} / {len(points)}')

        last_keypoints = np.copy(keypoints)
        last_grey = np.copy(grey)
        last_frame = np.copy(frame)



if __name__ == '__main__':
    main()