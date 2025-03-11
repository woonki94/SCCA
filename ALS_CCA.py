import numpy as np
import time
import helper

import numpy as np
import time

import numpy as np
import time


class ALS_CCA:
    def __init__(self, gamma_x=1e-5, gamma_y=1e-5, max_iter=600, tol=1e-6):
        self.gamma_x = gamma_x
        self.gamma_y = gamma_y
        self.max_iter = max_iter
        self.tol = tol
        self.u = None
        self.v = None

    def fit(self, X, Y, method = "ALS"):
        N, dx = X.shape
        _, dy = Y.shape

        Sigma_xx = (X.T @ X) / N + self.gamma_x * np.eye(dx)
        Sigma_yy = (Y.T @ Y) / N + self.gamma_y * np.eye(dy)
        Sigma_xy = (X.T @ Y) / N

        u = np.ones((dx, 1))
        v = np.ones((dy, 1))
        u /= np.linalg.norm(u)
        v /= np.linalg.norm(v)

        # Compute exact solution using SVD for comparison
        #U, S, Vt = np.linalg.svd(np.linalg.solve(Sigma_xx, Sigma_xy) @ np.linalg.inv(Sigma_yy), full_matrices=False)
        #u_opt = U[:, [0]]
        #v_opt = Vt.T[:, [0]]

        suboptimality = []
        num_passes = 0

        lr = 1e-3

        start_time = time.time()
        for _ in range(self.max_iter):
            if method == "SVRG":
                u_new = self.svrg(X, Y, self.gamma_x, u, v, num_epochs=100, lr=lr)
                #u_new = self.svrg(X, Y, self.gamma_x, u, v, num_epochs=30, lr=lr)
            elif method == "GD":
                u_new = self.gradient_descent(X, Y, self.gamma_x, u, v, num_epochs=100, lr=lr)
            elif method == "ASVRG":
                u_new = self.asvrg(X, Y, self.gamma_x, u, v, num_epochs=100, lr=lr)
            else:  # Default ALS update
                u_new = np.linalg.solve(Sigma_xx, Sigma_xy @ v)

            if method == "SVRG":
                v_new = self.svrg(Y, X, self.gamma_y, v, u_new, num_epochs=100, lr=lr)
                #v_new = self.svrg(Y, X, self.gamma_y, v, u_new, num_epochs=30, lr=lr)
            elif method == "GD":
                v_new = self.gradient_descent(Y, X, self.gamma_y, v, u_new, num_epochs=100, lr=lr)
            elif method == "ASVRG":
                v_new = self.asvrg(Y, X, self.gamma_x, v, u_new, num_epochs=100, lr=lr)
            else:
                v_new = np.linalg.solve(Sigma_yy, Sigma_xy.T @ u_new)

            u_new /= np.linalg.norm(u_new,axis=0) + 1e-8
            v_new /= np.linalg.norm(v_new,axis=0) + 1e-8

            num_passes += 1
            #suboptimality.append(helper.compute_suboptimality(u_new, u_opt, v_new, v_opt))


            #if np.linalg.norm(u_new - u) < self.tol and np.linalg.norm(v_new - v) < self.tol:
            #    break

            u, v = u_new, v_new

        end_time = time.time()
        self.u = u
        self.v = v

        return self.u, self.v, end_time - start_time, suboptimality

    def transform(self, X, Y):
        """
        Projects X and Y onto their respective first canonical directions.

        Parameters:
        X: np.ndarray (N, dx) - Data matrix X
        Y: np.ndarray (N, dy) - Data matrix Y

        Returns:
        X_proj: np.ndarray (N, 1) - Projected X data onto first canonical direction
        Y_proj: np.ndarray (N, 1) - Projected Y data onto first canonical direction
        """
        if self.u is None or self.v is None:
            raise ValueError("Model has not been fitted. Call `fit(X, Y)` first.")

        X_proj = X @ self.u  # Now correctly (N, dx) @ (dx, 1) → (N, 1)
        Y_proj = Y @ self.v  # Now correctly (N, dy) @ (dy, 1) → (N, 1)
        return X_proj, Y_proj

    #TODO: Make it Nesterov Accelerated Gradient
    def gradient_descent(self, X, Y, gamma, u_init, v_fixed, num_epochs=10, lr=1e-3):
        N, dx = X.shape
        u = u_init.copy()

        for epoch in range(num_epochs):
            grad = (X.T @ (X @ u - Y @ v_fixed)) / N + gamma * u
            u_new = u - lr * grad
            u_new /= np.linalg.norm(u_new)
            u = u_new

        return u

    def svrg(self, X, Y, gamma, u_init, v_fixed, num_epochs=10, batch_size=64, lr=1e-3):
        N, dx = X.shape
        u = u_init.copy()

        for epoch in range(num_epochs):
            shuffled_indices = np.random.permutation(N)  # Shuffle data indices
            full_grad = (X.T @ (X @ u - Y @ v_fixed)) / N + gamma * u  # Full dataset gradient

            u_snapshot = u.copy()  # Store a snapshot

            for i in range(0, N, batch_size):
                batch_indices = shuffled_indices[i:i + batch_size]
                x_batch = X[batch_indices, :]
                y_batch = Y[batch_indices, :]

                actual_batch_size = len(batch_indices)  # Handle last batch size
                grad_i = (x_batch.T @ (x_batch @ u - y_batch @ v_fixed)) / actual_batch_size + gamma * u
                grad_i_snap = (x_batch.T @ (
                            x_batch @ u_snapshot - y_batch @ v_fixed)) / actual_batch_size + gamma * u_snapshot

                u -= lr * (grad_i - grad_i_snap + full_grad)  # SVRG update

            u /= np.linalg.norm(u) + 1e-8  # Normalize once per epoch to prevent instability

        return u

    import numpy as np

    def asvrg(self, X, Y, gamma, u_init, v_fixed, num_epochs=100, batch_size=128, lr=1e-3, beta=0.9, l1_reg=0):
        """
        Implements Accelerated Proximal SVRG (ASVRG) for ALS.

        Parameters:
            X: np.ndarray (N, dx) - Data matrix X
            Y: np.ndarray (N, dy) - Data matrix Y
            gamma (float): Regularization parameter
            u_init: np.array (dx,) - Initial vector for u
            v_fixed: np.array (dy,) - Fixed vector for v
            num_epochs (int): Number of outer iterations
            batch_size (int): Mini-batch size
            lr (float): Learning rate
            beta (float): Momentum factor (typically 0.7 - 0.99)
            l1_reg (float): L1 regularization coefficient

        Returns:
            u (np.array): Updated solution
        """
        N, dx = X.shape
        u = u_init.copy()
        u_prev = u.copy()
        Sigma_xx = (X.T @ X) / N + gamma * np.eye(dx)

        for epoch in range(num_epochs):
            shuffled_indices = np.random.permutation(N)
            full_grad = (X.T @ (X @ u - Y @ v_fixed)) / N + gamma * u
            u_snapshot = u.copy()

            for i in range(0, N, batch_size):
                batch_indices = shuffled_indices[i:i + batch_size]
                x_batch = X[batch_indices, :]
                y_batch = Y[batch_indices, :]

                actual_batch_size = len(batch_indices)

                # Nesterov Acceleration
                u_temp = u + beta * (u - u_prev)

                # Compute stochastic variance reduced gradient
                grad_i = (x_batch.T @ (x_batch @ u_temp - y_batch @ v_fixed)) / actual_batch_size + gamma * u_temp
                grad_i_snap = (x_batch.T @ (
                            x_batch @ u_snapshot - y_batch @ v_fixed)) / actual_batch_size + gamma * u_snapshot

                # ASVRG update
                u_new = u_temp - lr * (grad_i - grad_i_snap + full_grad)

                # Apply proximal operator for L1 regularization (soft thresholding)
                if l1_reg > 0:
                    u_new = np.sign(u_new) * np.maximum(0, np.abs(u_new) - lr * l1_reg)

                # Update u_prev before modifying u
                u_prev = u.copy()
                u = u_new

                # Normalize per epoch (not per iteration)
            u /= (np.linalg.norm(u) + 1e-8)

            # Convergence check (relative stopping criterion)
            #if np.linalg.norm(u - u_snapshot) / (np.linalg.norm(u_snapshot) + 1e-8) < 1e-4:
            #    break

        return u

    def gsc(self, C, W):
        """
        Generalized Gram-Schmidt orthogonalization (GSC) with respect to a positive definite metric C.
        Ensures that W is orthonormal under the metric C.
        """
        return self.gs(W,C)

    def inprod(u, M, v=None):
        v = u if v is None else v
        return u.dot(M.dot(v.T))

    def gs(self,A, M):
        """
        Gram-Schmidt with inner product for M
        """
        A = A.copy()
        A[:, 0] = A[:, 0] / np.sqrt(self.inprod(A[:, 0], M))

        for i in range(1, A.shape[1]):
            Ai = A[:, i]
            for j in range(0, i):
                Aj = A[:, j]
                t = self.inprod(Ai, M, Aj)
                Ai = Ai - t * Aj
            norm = np.sqrt(self.inprod(Ai, M))
            if norm == 0:
                A[:, i] = Ai
            else:
                A[:, i] = Ai / norm
        return A
