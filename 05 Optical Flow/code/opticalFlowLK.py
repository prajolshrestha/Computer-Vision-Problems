import numpy as np
import cv2
from util import computeBilinearWeights
from util import *


class OpticalFlowLK:

    def __init__(self, winsize, epsilon, iterations) -> None:
        
        self.winsize = winsize
        self.epsilon = epsilon
        self.iterations = iterations

    ## Lucas Kannade Algorithm for optical flow estimation
    def compute(self, prevImg, nextImg, prevPts):

        assert prevImg.size != 0 and nextImg.size != 0, "check prevImg and nextImg"
        assert prevImg.shape[0] == nextImg.shape[0], "size mismatch, rows."
        assert prevImg.shape[1] == nextImg.shape[1], "size mismatch, cols."

        N = prevPts.shape[0]
        status = np.ones(N, dtype=int)
        nextPts = np.copy(prevPts) # to store updated keypoints
        
        # Use normalized Scharr filter to get derivative wrt x and y 
        prevDerivx = cv2.Scharr(prevImg, cv2.CV_64F,1,0, scale=1/32.0)
        prevDerivy = cv2.Scharr(prevImg, cv2.CV_64F,0,1, scale=1/32.0)


        # Find center of search window 
        halfWin = np.array([(self.winsize[0] - 1)* 0.5, (self.winsize[1] - 1)* 0.5])
        #print(halfWin)
        
        weights = computeGaussianWeights(self.winsize, 0.3)

        # Iterate over keypoints to predict same keypoints in next frame 
        for ptidx in range(N):
            u0 = prevPts[ptidx] # current ith keypoint (subpixel coordinates)
            #print(u0) #[164.28429  105.038086]
            u0 -= halfWin # positioning the point at the center of search window
            #print(u0) # [152.28429   93.038086]
            u = u0 # Initialize target point u with the keypoint in previous image.
            
            # Convert subpixel coordinate to integer coordinates
            iu0 = [int(np.floor(u0[0])), int(np.floor(u0[1]))] # Top left corner of search window
            #print(iu0) # [152, 93]

            # Check if the search window goes out of bounds in 'prevImg'
            if iu0[0] < 0 or \
                    iu0[0] + self.winsize[0] >= prevImg.shape[1] - 1 or \
                    iu0[1] < 0 or \
                    (iu0[1] + self.winsize[1] >= (prevImg.shape[0] - 1)):
                status[ptidx] = 0 #set current point to 0 and continue to next keypoint
                continue
            
            #Bilinear interpolation weights for current keypoint u0
            bw = computeBilinearWeights(u0)
            

            # Initialize
            bprev = np.zeros((self.winsize[0] * self.winsize[1], 1)) # to store values computed based on pixel intensities
            A = np.zeros((self.winsize[0]*self.winsize[1], 2)) # to store spatial derivative
            AtWA = np.zeros((2,2)) 
            invAtWA = np.zeros((2,2))

            # Process inside a search window
            for y in range(self.winsize[1]):
                for x in range(self.winsize[0]):
                    
                    # Calculate global coordinates = searchwindow coordinate(top left corner of search window) + local coordinate within the window
                    gx = int(iu0[0] + x)
                    gy = int(iu0[1] + y)

                    # TODO 3.1
                    #populate bprev based on provide equations
                    bprev[int(y*self.winsize[0]+x), 0] = bw[0] * prevImg[gy, gx] + \
                                                         bw[1] * prevImg[gy, gx + 1] + \
                                                         bw[2] * prevImg[gy + 1, gx] + \
                                                         bw[3] * prevImg[gy + 1, gx + 1]
                    
                      
                    # Populate A based on the equations in the problem
                    A[y*self.winsize[0] + x, 0] = bw[0] * prevDerivx[gy, gx] + \
                                                  bw[1] * prevDerivx[gy, gx + 1] + \
                                                  bw[2] * prevDerivx[gy + 1, gx] + \
                                                  bw[3] * prevDerivx[gy + 1, gx + 1]
                    A[y*self.winsize[0] + x, 1] = bw[0] * prevDerivy[gy, gx] + \
                                                  bw[1] * prevDerivy[gy, gx + 1] + \
                                                  bw[2] * prevDerivy[gy + 1, gx] + \
                                                  bw[3] * prevDerivy[gy + 1, gx + 1]

                    

            # Compute invAtwA
            weighted_A = A* weights.reshape(-1, 1)
            AtWA = np.dot(np.transpose(A), weighted_A)
            invAtWA = invertMatrix2x2(AtWA)


            ## Estimate target point with the previous point
            u = u0

            ## Iterative solver
            for j in range(self.iterations):
                iu = [int(np.floor(u[0])), int(np.floor(u[1]))]

                if iu[0] < 0 or iu[0] + self.winsize[0] >= prevImg.shape[1] - 1 \
                        or iu[1] < 0 or iu[1] + self.winsize[1] >= prevImg.shape[0] - 1:
                    status[ptidx] = 0
                    break

                bw = computeBilinearWeights(u)

                bnext = np.zeros((self.winsize[0] * self.winsize[1], 1))
                #AtWbnbp = [0, 0]
                for y in range(self.winsize[1]):
                    for x in range(self.winsize[0]):
                        gx = iu[0] + x
                        gy = iu[1] + y

                        # Populate bnext based on the equations in the problem
                        bnext[y*self.winsize[0]+x, 0] = bw[0] * nextImg[gy, gx] + \
                                                        bw[1] * nextImg[gy, gx + 1] + \
                                                        bw[2] * nextImg[gy + 1, gx] + \
                                                        bw[3] * nextImg[gy + 1, gx + 1]

                weights_flat = weights.reshape(-1,1).T       
                AtWbnbp = np.dot((np.transpose(A)*weights_flat) , (bnext -bprev)) # (2x1) 
                
                

                deltaU = -np.dot(invAtWA, AtWbnbp)
                # Update target point
                u += deltaU.reshape((2,))
                #print(deltaU)

                if np.linalg.norm(deltaU) < self.epsilon: #We got sufficient motion, so lets break the iteration loop! Algorithm converged to best estimate
                 
                    break

            nextPts[ptidx] = u + halfWin #Predicted keypoint by centering back to original position

        return nextPts, status

            





        
        