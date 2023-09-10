import numpy as np
import cv2
import matplotlib.pyplot as plt


class Params():

    def __init__(self):
        # Instance Variables
        self.segmentationThreshold = 0.7
        self.minRegionSize = 50
        self.minDisp = 0
        self.maxDisp = 14


class Disparity:

    def __init__(self):

        '''
        Instance Variables:
                    PatchRadius: The total patch size is (2*patchRadius) * (2*patchRadius+1)
                    initIterations: Number of random guesses in initialization
                    iterations: Number of spatial propagations
                    optimizeIterations: Number of random guesses in optimization step of spatial propagation                   
        '''
        self.patchRadius = 2
        self.initIterations = 1
        self.iterations = 1
        self.optimizeIterations = 1

        self.params = Params() #Instance Created
        self.showDebug = True

  
    def cost(self, img1, img2, x, y, d):

        """
        Input Parameters: img1, img2, x, y, d

        Interpolation technique has been used to get non-integer pixel intensity for right image

        Output: Normalized cost
        """
        
        # Conditional statement that checks if the patch goes out of bounds or not.
        if (y+self.patchRadius >= img1.shape[0]
                or y-self.patchRadius < 0
                or x-self.patchRadius < 0
                or x+self.patchRadius >= img1.shape[1]
                or int(x-d-self.patchRadius) < 0
                or int(x-d+self.patchRadius+1) >= img1.shape[1]):
            return 10000000 # Out of bound so return large value indicating invalid cost 

        # Initialize the required variables
        cost = 0.0
        h, w = img1.shape #5,5
        patch_radius = self.patchRadius #1
        patch_size = 2 * patch_radius + 1 #3

        # Iterate over vertical and horizontal offsets within a patch.
        for dy in range(-patch_radius, patch_radius + 1):
            for dx in range(-patch_radius, patch_radius + 1):
                
                x1, y1 = x + dx, y + dy # coordinates in left images
                x2 = x1 - d # x coordinate in right image (using disparity)

                # Check if the calculated coordinates are within the bounds of both images (img1 & img2) and if they are within the patch.
                #if (0 <= x1 < w and 0 <= x2 < w and 0 <= y1 < h):

                intensity1 = img1[y1, x1] # Retrive intensity value of img1 at given coordinate
                    
                # Linear interpolation for sampling the right image
                x2_low = int(x2)
                x2_high = x2_low + 1
                alpha = x2 - x2_low

                # Ensure that high values do not go out of bounds
                if x2_high >= w:
                    x2_high = w - 1

                intensity2_low = img2[y1, x2_low]
                intensity2_high = img2[y1, x2_high]
                interpolated_intensity2 = (1 - alpha) * intensity2_low + alpha * intensity2_high # Interpolate

                cost += np.abs(intensity1 - interpolated_intensity2)
       
        return cost / (patch_size ** 2)  # Return the average SAD cost for the patch.

    
    def computeDisparity(self, img1, img2, minDisp, maxDisp):
        """
        Input Parameters: img1, img2, minDisp, maxDisp

        Output: 
        """

        self.params.minDisp = minDisp
        self.params.maxDisp = maxDisp

        # Step 1: Disparity image is initialized with random values 
        scoreF = np.zeros((img1.shape[0], img1.shape[1]))
        dispF, scoreF = self.initRandom(img1, img2, scoreF)

        # Step 2 & 3: Propagate and Random search 
        for i in range(self.iterations):

            # Forward mode
            #self.propagate(img1, img2, dispF, scoreF, True)
            self.randomSearch(img1, img2, dispF, scoreF)
            
            # Backward mode
            #self.propagate(img1, img2, dispF, scoreF, False)
            #self.randomSearch(img1, img2, dispF, scoreF)

        self.showDebugDisp("1_after_loop", 0, 1, dispF)
        dispF = self.thresholding(dispF, scoreF, 15)
        #dispF = self.segmentation(dispF)

   

        return dispF


    # Random Initialization
    def initRandom(self, img1, img2, scoreF):

        print(f"Initializing Disparity with random values in the range ({self.params.minDisp},{self.params.maxDisp})")

        disp = np.zeros((img1.shape[0], img1.shape[1]), dtype=np.int32)
        ndisp = 50
        
        # Random initialization of the disparity map.
        disparity_maps = np.zeros([img1.shape[0], img1.shape[1], ndisp], dtype=np.float32)
        kernel = np.array([[1, 1, 1],
                           [1, -1, 1],
                           [1, 1, 1]])
        #kernel = np.ones((5,5), dtype=np.float32)

        # For each pixel compute "initIterations" random disparity values in the range[minDisp, maxDisp]
        for d in range(0, ndisp):

            ## Step 1: Sample a random disparity value in the specified range
            random_disparity = np.random.uniform(self.params.minDisp, self.params.maxDisp)
            
            ## Step 2: Shift img2 randomly in horizontal direction
            translation_matrix = np.float32([[1,0,random_disparity],
                                             [0,1,0]])
            shifted_image = cv2.warpAffine(img2, translation_matrix, (img2.shape[1],img2.shape[0]))

            ## Step 3: Calculate squared differences between img1 and shifted img2 and find SAD
            SAD = np.abs(np.float32(img1) - np.float32(shifted_image))
            filtered_SAD = cv2.filter2D(SAD, -1, kernel) #sum
            disparity_maps[:, :, d] = filtered_SAD # Store the cost in the disparity map
            

            # For Visualization and debugging
            disparity = np.argmin(disparity_maps, axis=2)
            disparity = np.uint8(disparity * 255 / ndisp)
            disparity = cv2.equalizeHist(disparity)
            cv2.imshow('image', disparity)
            cv2.waitKey(1)


        # Choose the best disparity according to the cost function
        disp = np.argmin(disparity_maps, axis=2)
        #print(disp.shape) # (144,192)

        # Store the best cost in scoreF
        best_cost = np.min(disparity_maps, axis=2)
        scoreF = best_cost
      
        self.showDebugDisp('0_Init',0, 1, disp)

        return disp, scoreF
    
    # Spatial Propagation
    def propagate(self, img1, img2, disp, scoreF, forward=True):
        """
        Iterate over the disparity image and propagate good values to nearby pixels.
        """

        h, w = disp.shape
        dn = w * h
        new_disp = np.copy(disp)  # To store updated disparity values

        if forward:
            print("Propagate Forward")
            for y in range(h):
                for x in range(w):
                    # Normalize previous disparities
                    if y > 0:
                        new_disp[x, y - 1] = new_disp[x, y - 1] / dn
                    if x > 0:
                        new_disp[x - 1, y] = new_disp[x - 1, y] / dn

                    # Forward propagation: find the lowest cost disparity
                    min_cost = self.cost(img1, img2, x, y, disp[x, y])  # Use the current disparity here
                    best_disp = disp[x, y]  # Use the current disparity as the initial best

                    if x > 0:
                        left_cost = self.cost(img1, img2, x, y, disp[x - 1, y])
                        if left_cost < min_cost:
                            best_disp = disp[x - 1, y]
                            min_cost = left_cost

                    if y > 0:
                        up_cost = self.cost(img1, img2, x, y, disp[x, y - 1])
                        if up_cost < min_cost:
                            best_disp = disp[x, y - 1]
                            min_cost = up_cost

                    new_disp[x, y] = best_disp
                    scoreF[x, y] = min_cost

        else:
            print("Propagate Backward")
            for y in range(h - 1, -1, -1):
                for x in range(w - 1, -1, -1):
                    # Normalize previous disparities
                    if y < h - 1:
                        new_disp[x, y + 1] = new_disp[x, y + 1] / dn
                    if x < w - 1:
                        new_disp[x + 1, y] = new_disp[x + 1, y] / dn

                    # Backward propagation: find the lowest cost disparity
                    min_cost = self.cost(img1, img2, x, y, disp[x, y])  # Use the current disparity here
                    best_disp = disp[x, y]  # Use the current disparity as the initial best

                    if x < w - 1:
                        right_cost = self.cost(img1, img2, x, y, disp[x + 1, y])
                        if right_cost < min_cost:
                            best_disp = disp[x + 1, y]
                            min_cost = right_cost

                    if y < h - 1:
                        down_cost = self.cost(img1, img2, x, y, disp[x, y + 1])
                        if down_cost < min_cost:
                            best_disp = disp[x, y + 1]
                            min_cost = down_cost

                    new_disp[x, y] = best_disp
                    scoreF[x, y] = min_cost

        self.showDebugDisp('1_Init', 0, 1, disp)

        return new_disp, scoreF


    


    def randomSearch(self, img1, img2, disp, scoreF):

        """
            Algo:
            1.It initializes R0 based on the specified range of disparities.
            2.It iterates over each pixel in the disparity map.
            3.For each pixel, it performs random search with decreasing radii for the specified number of iterations (self.optimizeIterations).
            4.It calculates a new disparity by adding a random offset within the current radius.
            5.It checks if the new disparity is within the specified disparity range (self.params.minDisp and self.params.maxDisp).
            6.If the new disparity is valid and has a lower cost, it updates the disparity map and cost.
            7.Finally, it shows the disparity map for debugging purposes.
                    
        """

        print("Random Search")
        # Initialize R0 based on the specified range of disparities
        R0 = (self.params.maxDisp - self.params.minDisp) * 0.25

        alpha = 0.2
        # Iterate over each pixel in disparity map
        for y in range(disp.shape[0]):
            for x in range(disp.shape[1]):
                current_disparity = disp[y,x]

                # Perform random search with decreasing radii for the specified number of iterations
                for i in range(self.optimizeIterations):
                    radius = (alpha * R0) / (2 ** i)
                    random_offset = radius * np.random.uniform(-1,1) # random offset within current radius
                    new_disparity = current_disparity + random_offset # new disparity

                    # checks if new disparity is  within the specified disparity range
                    if self.params.minDisp <= new_disparity <= self.params.maxDisp:
                        
                        # Calculate the cost for new disparity
                        new_cost = self.cost(img1, img2, x, y, new_disparity)

                        # Update disparity map and cost
                        if new_cost < scoreF[y,x]:
                            disp[y,x] = new_disparity
                            scoreF[y,x] = new_cost
        
        self.showDebugDisp('Random',0, 1, disp)


        return disp, scoreF
    

    # Outlier Removal

    # Thresholding
    def thresholding(self, disp, scoreF, t):
        
        print("Thresdholding with t = " + str(t))

        
        ## Remove all disparities with score > t
        disp[scoreF > t] = 0

        self.showDebugDisp("2_Thresholding", 1, 1, disp)
        return disp

    def segmentation(self, disp):

        print("Filter by segmentation")

        h,w = disp.shape
        visited = np.zeros((h,w), dtype = bool)

        def grow_region(x,y, label):
            stack = [(x,y)]
            region_size = 0

            while stack:

                cx, cy = stack.pop()
                if visited[cx,cy]:
                    continue

                visited[cx, cy]= True
                disp[cx, cy] = label
                region_size += 1

                # Check neighbors for similarity
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = cx + dx, cy + dy
                        if (0 <= nx < h and 
                            0 <= ny < w and not 
                            visited[nx, ny]
                            and abs(int(disp[nx, ny]) - int(disp[cx, cy])) <= self.params.segmentationThreshold):
                            
                            stack.append((nx, ny))

            return region_size
        
        labels = np.zeros((h, w), dtype=int)
        label_counter = 1
        for y in range(h):
            for x in range(w):
                if not visited[y, x] and disp[y,x] != 0:
                    region_size = grow_region(y,x, label_counter)
                    if region_size < self.params.minRegionSize:
                    
                        for i in range(h):
                            for j in range(w):
                                if labels[i, j] == label_counter:
                                    disp[i, j] = 0
                        
                    else:
                        label_counter += 1
       

        self.showDebugDisp("3_Segmentation", 1, 2, disp)

        return disp
    
  
        

    def consistencyCheck(self, dispL, dispR):
        print("Applying consistency check")

        h, w = dispL.shape
        disp_threshold = 1.0  # You can adjust this threshold as needed

        for i in range(h):
            for j in range(w):
                # Calculate the absolute differences between disparities in the left and right images
                diff = np.sum(np.abs(dispL - dispR))

                # Find the minimum difference and its location
                min_diff = np.min(diff)
                min_index = np.argmin(diff)

                if min_index > 1 and min_index < w - 1:
                    # Calculate the updated disparity based on the algorithm
                    dispL[i, j] = (min_index - 1) - (
                        0.5 * (diff[min_index + 1] - diff[min_index - 1]) /
                        (diff[min_index - 1] - (2 * diff[min_index]) + diff[min_index + 1])
                    )
                else:
                    dispL[i, j] = min_index - 1

        self.showDebugDisp("4_ConsistencyCheck", 1, 3, dispL)

        return dispL







################################ Display #######################################

    def showDebugDisp(self, name, x, y, disp):

        if self.showDebug:
            vis = np.copy(disp)
            # Normalize disparity values to [0,1]
            vis = vis - self.params.minDisp
            vis = vis / (self.params.maxDisp - self.params.minDisp)

            if self.params.minDisp < 0:
                vis = 1 - vis # invert values
            
            out = np.uint8(vis * 255.0) # scale to [0,255]
            cv2.imwrite(name + ".png", out)
