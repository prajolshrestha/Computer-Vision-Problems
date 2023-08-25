import cv2

# Function to find keypoints and descriptors using ORB detector
def computeFeatures(image_data):

    # Initiate ORB detector (Binary string based descriptor)
    orb = cv2.ORB_create(nfeatures = 500, scoreType= cv2.ORB_FAST_SCORE)

    # Find Keypoints and descriptors with ORB
    keypoints, descriptors = orb.detectAndCompute(image_data['img'], None)

    print(f"Found {len(keypoints)} ORB features on image {image_data['id']}")

    # Store computed features in our Dictonary
    image_data['keypoints'] = keypoints
    image_data['descriptors'] = descriptors # vector with 32 elements is stored row-wise in numpy matrix

    return image_data

# Draw matches
def createMatchImage(img1, img2, matches):
    img_matches = cv2.drawMatches(img1['img'], img1['keypoints'],
                                  img2['img'], img2['keypoints'],
                                  matches, outImg= None, matchColor= (0, 255, 0),
                                  singlePointColor= (0, 255, 0), flags= 2)
    return img_matches