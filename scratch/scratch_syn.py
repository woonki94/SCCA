import cv2
import numpy as np

import helper
from ALS_CCA import ALS_CCA

# Example Usage
if __name__ == "__main__":
    #init algorithms
    als_cca = ALS_CCA()

    image_monkey = cv2.imread("../test_img/monkey.jpeg", cv2.IMREAD_GRAYSCALE)
    if image_monkey is None:
        raise ValueError("Image not found! Make sure the path is correct.")
    image_apple = cv2.imread("../test_img/woman.jpeg", cv2.IMREAD_GRAYSCALE)
    if image_apple is None:
        raise ValueError("Image not found! Make sure the path is correct.")

    height, width = image_monkey.shape



    split_point = width // 2  # Integer division

    if width % 2 != 0:
        left_half = image_monkey[:, :split_point]
        right_half = image_monkey[:, split_point+1:]
    else:
        left_half = image_monkey[:, :split_point]
        right_half = image_monkey[:, split_point:]

    print(left_half.shape)
    print(right_half.shape)

    X = left_half.astype(np.float32)
    Y = right_half.astype(np.float32)


    '''
    split_point = height // 2  # Integer division
    # Handle odd heights
    if height % 2 != 0:
        upper_half = image_monkey[:split_point , :]  # Extra row to upper half
        lower_half = image_monkey[split_point + 1:, :]
    else:
        upper_half = image_monkey[:split_point, :]
        lower_half = image_monkey[split_point:, :]

    print(upper_half.shape)
    print(lower_half.shape)

    # Flatten the halves to create feature matrices
    X = upper_half.astype(np.float32)
    Y = lower_half.astype(np.float32)
    '''

    ''' When testing with 2 different images
    X = image_monkey.astype(np.float32)
    Y = image_apple.astype(np.float32)
    '''

    ''' When testing with 2 same images
    X = image_monkey.astype(np.float32)
    Y = image_monkey.astype(np.float32)
    '''
    X = X.reshape(1,-1)
    Y = Y.reshape(1, -1)
    print(X.shape)
    print(Y.shape)


    # Normalize data (zero mean, unit variance)
    X = (X - np.mean(X, axis=1, keepdims=True)) / (np.std(X, axis=1, keepdims=True) + 1e-8)
    Y = (Y - np.mean(Y, axis=1, keepdims=True)) / (np.std(Y, axis=1, keepdims=True) + 1e-8)

    u_als, v_als, elapsed_time = als_cca.fit(X, Y, SGD=False)
    X_proj = u_als.T @ X  # Shape: (num_components, samples)
    Y_proj = v_als.T @ Y

    helper.plot_correlation_points(X, Y, u_als, v_als)
    corr = helper.canonical_correlation(X,Y,u_als,v_als)

    #helper.plot_correlation_points(X, Y, u_als_sgd, v_als_sgd)
    #sgd_corr = helper.canonical_correlation(X, Y, u_als_sgd, v_als_sgd)

    print(X_proj.shape)
    print(Y_proj.shape)
    print("Correlation: " ,corr)
    #print("SGD ALS Correlation: ", sgd_corr)

