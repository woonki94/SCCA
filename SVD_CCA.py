import time

import numpy as np

'''
None-iterative solution
Just for comparison
'''
class SVD_CCA:

    def __init__(self):
        self.u = None  # Canonical direction for X
        self.v = None  # Canonical direction for Y

    def fit(self, X, Y):
        start_time = time.time()
        N, dx = X.shape
        _, dy = Y.shape

        # Compute covariance matrices
        Sigma_xx = (X.T @ X) / N
        Sigma_yy = (Y.T @ Y) / N
        Sigma_xy = (X.T @ Y) / N

        # Perform SVD for whitening
        Ux, Sigma_x, _ = np.linalg.svd(Sigma_xx)  # SVD of Sxx
        Uy, Sigma_y, _ = np.linalg.svd(Sigma_yy)  # SVD of Syy

        # Compute inverse square root of eigenvalues (whitening transformation)
        Sigma_x_inv_sqrt = np.diag(1.0 / np.sqrt(Sigma_x + 1e-10))  # Adding small epsilon for numerical stability
        Sigma_y_inv_sqrt = np.diag(1.0 / np.sqrt(Sigma_y + 1e-10))

        # Whitening matrices
        Sigma_xx_inv_sqrt = Ux @ Sigma_x_inv_sqrt @ Ux.T
        Sigma_yy_inv_sqrt = Uy @ Sigma_y_inv_sqrt @ Uy.T

        # Compute transformed cross-covariance matrix
        M = Sigma_xx_inv_sqrt @ Sigma_xy @ Sigma_yy_inv_sqrt

        # SVD of M to get first canonical component
        U, _, Vt = np.linalg.svd(M)

        # Store the first canonical directions
        self.u = Sigma_xx_inv_sqrt @ U[:, 0]  # Now (dx,)
        self.v = Sigma_yy_inv_sqrt @ Vt.T[:, 0]  # Now (dy,)

        elapsed_time = time.time() - start_time
        return self.u, self.v, elapsed_time

    def transform(self, X, Y):
        if self.u is None or self.v is None:
            raise ValueError("Model has not been fitted. Call `fit(X, Y)` first.")

        X_proj = X @ self.u.reshape(-1, 1)
        Y_proj = Y @ self.v.reshape(-1, 1)

        return X_proj, Y_proj
