import numpy as np

def computeBilinearWeights(q):

    '''
        Compute Bilinear weights for point q

        :return: List of bilinear weights [w00, w01, w10, w11] for the surrounding pixels (x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)
    '''
    # Extract integer and fractional parts of the coordinates
    x, y = int(q[0]), int(q[1])
    a, b = q[0] - x, q[1] - y

    # Compute bilinear interpolation weights
    weight_x0y0 = (1-a) * (1-b)
    weight_x1y0 = a * (1 - b)
    weight_x0y1 = (1-a)*b
    weight_x1y1 = a*b

    weights = (weight_x0y0, weight_x1y0, weight_x0y1,weight_x1y1)

    return weights 

def computeGaussianWeights(winsize, sigma):

    '''
        Compute gaussian weight.
                    (-used later: the further the pixel away from the center the lesser they should contribute to the result)

        :return- Gaussian weights
    '''
    height, width = winsize
    # Compute center coordinates
    cx, cy = (width - 1) / 2, (height - 1) / 2

    # Create an empty array to store the weights
    weights = np.zeros((height,width))
    for y in range(height):
        for x in range(width):

            #Calculate the centered and normalized coordinates
            x_hat = (x - cx) / width
            y_hat = (y - cy) / height

            # Compute Gaussian weight using the formula
            weights[y,x] = np.exp(-(x_hat**2 + y_hat**2) / (2 * sigma**2))
           
    return weights

def invertMatrix2x2(A):

    '''
        Compute Inverse of the 2x2 matrix A

        :return- Inverse of A

    '''
    a, b, c, d = A[0, 0], A[0, 1], A[1, 0], A[1, 1]

    det = a*d - b*c
    inv_det = 1.0 / det
    invA = np.array([[d*inv_det, -b*inv_det],
                     [-c*inv_det, a*inv_det]])

    return invA 

