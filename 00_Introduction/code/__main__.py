import numpy as np
import cv2

# O1: Load image, disply and write image
file = "..\data\img.png"
image = cv2.imread(file)
cv2.imshow('Image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('img.jpg',image)

# 02: Resize the image by factor of 0.5 in both direction
height, width, _ = image.shape
resized_image = cv2.resize(image, (height, width), fx =0.5,fy=0.5)
cv2.imshow('resized_image', resized_image)
cv2.imwrite('small.jpg',resized_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 03: Create three images with each channel (R,G,B)
      
img_R = np.copy(image)
img_R[:,:,0:2] = 0

cv2.imshow('image_red',img_R)
cv2.waitKey(0)

img_G = np.copy(image)
img_G[:,:,::2] = 0

cv2.imshow('image_green',img_G)
cv2.waitKey(0)

img_B = np.copy(image)
img_B[:,:,1] = 0
img_B[:,:,2] = 0
        
cv2.imshow('image_blue',img_B)

cv2.waitKey(0)
cv2.destroyAllWindows()
