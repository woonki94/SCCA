import numpy as np
import time
import helper


class ALS_CCA:
    def __init__(self, gamma_x=1e-5, gamma_y=1e-5, max_iter=100, tol=1e-6):
        self.gamma_x = gamma_x
        self.gamma_y = gamma_y
        self.max_iter = max_iter
        self.tol = tol
        self.u = None
        self.v = None

    def fit(self, X,Y,SGD= False):
        """
        Alternating Least Squares (ALS) for Canonical Correlation Analysis (CCA).

        Args:
            X (numpy.ndarray): Data matrix for view 1 (dx x N)
            Y (numpy.ndarray): Data matrix for view 2 (dy x N)
            gamma_x (float): Regularization parameter for Σxx
            gamma_y (float): Regularization parameter for Σyy
            max_iter (int): Maximum number of iterations
            tol (float): Convergence tolerance

        Returns:
            u (numpy.ndarray): Left canonical vector
            v (numpy.ndarray): Right canonical vector
        """
        dx, N = X.shape
        dy, _ = Y.shape

        # Compute covariance matrices
        Σxx = (X @ X.T) / N + self.gamma_x * np.eye(dx)
        Σyy = (Y @ Y.T) / N + self.gamma_y * np.eye(dy)
        Σxy = (X @ Y.T) / N  # Cross-covariance

        # Initialize u and v randomly
        u = np.random.randn(dx, 1)
        v = np.random.randn(dy, 1)

        # Normalize initial vectors
        u /= np.sqrt(u.T @ Σxx @ u)
        v /= np.sqrt(v.T @ Σyy @ v)

        start_time = time.time()
        for _ in range(self.max_iter):
            # Update u
            if SGD:
                u_new = self.svrg(X, Y, self.gamma_x, u, v, num_epochs=self.max_iter, eta=1e-3)
            else:
                u_new = np.linalg.solve(Σxx, Σxy @ v)

            u_new /= np.sqrt(u_new.T @ Σxx @ u_new)  # Normalize

            # Update v
            if SGD:
                v_new = self.svrg(X, Y, self.gamma_y, v, u, num_epochs=self.max_iter, eta=1e-3)
            else:
                v_new = np.linalg.solve(Σyy, Σxy.T @ u_new)

            v_new /= np.sqrt(v_new.T @ Σyy @ v_new)  # Normalize

            # Check for convergence
            if np.linalg.norm(u_new - u) < self.tol and np.linalg.norm(v_new - v) < self.tol:
                break

            u, v = u_new, v_new  # Update variables
        end_time = time.time()
        elapsed_time = end_time - start_time
        return u, v, elapsed_time

    def svrg(self, X, Y, gamma, u_init, v_fixed, num_epochs=10, batch_size=128, eta=1e-1):
        """
        Implements the SVRG algorithm to solve the least squares problem:
            min_u (1/2N) * ||X.T @ u - Y.T @ v_fixed||^2 + (gamma/2) * ||u||^2

        Parameters:
            X (np.array): Data matrix (dx, N)
            Y (np.array): Data matrix (dy, N)
            gamma (float): Regularization parameter
            u_init (np.array): Initial solution (dx,)
            v_fixed (np.array): Fixed vector from previous iteration (dy,)
            num_epochs (int): Number of outer iterations
            batch_size (int): Mini-batch size
            eta (float): Learning rate

        Returns:
            u (np.array): Updated solution
        """
        N = X.shape[1]
        u = u_init.copy()
        Σxx = (X @ X.T) / N + gamma * np.eye(X.shape[0])  # Precompute Σxx for normalization

        for epoch in range(num_epochs):
            # Compute full gradient once per epoch
            full_grad = (1 / N) * (X @ (X.T @ u - Y.T @ v_fixed)) + gamma * u

            # Store snapshot of u
            u_snapshot = u.copy()

            for i in range(max(1, N // batch_size)):  # Avoid division by zero if batch_size > N
                idx = np.random.choice(N, batch_size, replace=False)
                x_i = X[:, idx]
                y_i = Y[:, idx]

                # Compute stochastic gradients
                grad_i = (1 / batch_size) * (x_i @ (x_i.T @ u - y_i.T @ v_fixed)) + gamma * u
                grad_i_snap = (1 / batch_size) * (x_i @ (x_i.T @ u_snapshot - y_i.T @ v_fixed)) + gamma * u_snapshot

                # Update step
                u_prev = u.copy()
                u -= eta * (grad_i - grad_i_snap + full_grad)

                # Normalize after each update
                u /= np.sqrt(u.T @ Σxx @ u)

                # Early stopping
                #if np.linalg.norm(u - u_prev) < self.tol:
                #    return u

        return u
