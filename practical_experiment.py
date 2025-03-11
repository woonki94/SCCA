import cv2
import numpy as np

import helper
from ALS_CCA import ALS_CCA

# Example Usage
if __name__ == "__main__":
    #init algorithms
    als_cca = ALS_CCA()

    image = cv2.imread("test_img/monkey.jpeg", cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found! Make sure the path is correct.")
    image_apple = cv2.imread("test_img/woman.jpeg", cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found! Make sure the path is correct.")

    # Get image dimensions
    height, width = image.shape

    # Split image into left and right halves
    #left_half = image[:, :width // 2]
    #right_half = image[:, width // 2:]
    '''
    upper_half = image[:height // 2, :]  # Top half
    lower_half = image[height // 2:, :] # Bottom half

    # Flatten the halves to create feature matrices
    X = upper_half.astype(np.float32)
    Y = lower_half.astype(np.float32)
    '''
    X = image.astype(np.float32)
    Y = image_apple.astype(np.float32)
    X = X.reshape(1,-1)
    Y = Y.reshape(1, -1)

    # Normalize data (zero mean, unit variance)
    X = (X - np.mean(X, axis=1, keepdims=True)) / (np.std(X, axis=1, keepdims=True) + 1e-8)
    Y = (Y - np.mean(Y, axis=1, keepdims=True)) / (np.std(Y, axis=1, keepdims=True) + 1e-8)

    u_als, v_als, elapsed_time = als_cca.fit(X, Y, SGD=False)
    X_proj = u_als.T @ X  # Shape: (num_components, samples)
    Y_proj = v_als.T @ Y

    Z = np.stack((X_proj, Y_proj))
    print(image.shape)
    print(X.shape)
    print(Y.shape)
    print(Z.shape)
    print(u_als.shape)
    print(v_als.shape)

    helper.plot_correlation_points(X, Y, u_als, v_als)
    corr = helper.canonical_correlation(X,Y,u_als,v_als)

    #helper.plot_correlation_points(X, Y, u_als_sgd, v_als_sgd)
    #sgd_corr = helper.canonical_correlation(X, Y, u_als_sgd, v_als_sgd)

    print("Correlation: " ,corr)
    #print("SGD ALS Correlation: ", sgd_corr)
