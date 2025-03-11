import cv2
import numpy as np
from scipy.linalg import solve
import time

import helper


import numpy as np
import time
from scipy.linalg import solve


class SI_CCA:
    def __init__(self, gamma_x=1e-2, gamma_y=1e-2, delta_tilde=1e-3, m1=300, m2=200, epsilon_tilde=1e-3):
        self.gamma_x = gamma_x
        self.gamma_y = gamma_y
        self.delta_tilde = delta_tilde
        self.m1 = m1
        self.m2 = m2
        self.epsilon_tilde = epsilon_tilde
        self.u = None
        self.v = None

    def fit(self, X, Y, method):

        #u_opt, v_opt = helper.SVD_solver(X,Y)
        start_time = time.time()

        N, dx = X.shape
        _, dy = Y.shape

        # Compute covariance matrices
        Sigma_xx = (X.T @ X) / N + self.gamma_x * np.eye(dx)
        Sigma_yy = (Y.T @ Y) / N + self.gamma_y * np.eye(dy)
        Sigma_xy = (X.T @ Y) / N

        # Compute exact solution using SVD for comparison
        U, S, Vt = np.linalg.svd(np.linalg.solve(Sigma_xx, Sigma_xy) @ np.linalg.inv(Sigma_yy), full_matrices=False)
        u_opt = U[:, [0]]
        v_opt = Vt.T[:, [0]]
        suboptimality = []
        num_passes = 0

        u = np.ones((dx, 1))
        v = np.ones((dy, 1))
        u /= np.linalg.norm(u)
        v /= np.linalg.norm(v)

        # Initialize u0, v0 with proper scaling
        u0 = np.random.randn(dx, 1) / np.sqrt(dx)
        v0 = np.random.randn(dy, 1) / np.sqrt(dy)

        # Normalize initial vectors
        u0 /= np.sqrt(u0.T @ Sigma_xx @ u0 + 1e-8)
        v0 /= np.sqrt(v0.T @ Sigma_yy @ v0 + 1e-8)

        # Phase I: Shift-and-invert preconditioning
        s = 0
        lambda_s = 1 + self.delta_tilde
        u_prev, v_prev = u, v

        while True:
            s += 1
            for _ in range(self.m1):
                A = np.block([
                    [lambda_s * Sigma_xx, -Sigma_xy],
                    [-Sigma_xy.T, lambda_s * Sigma_yy]
                ])
                b = np.block([
                    [Sigma_xx @ u_prev],
                    [Sigma_yy @ v_prev]
                ])
                if method == "SVRG":
                    solution = self.svrg(A, b, np.vstack([u_prev, v_prev]), num_epochs=200, lr=1e-3)
                elif method == "ASVRG":
                    solution = self.asvrg(A, b, np.vstack([u_prev, v_prev]), num_epochs=200, lr=1e-3, momentum=0.9)

                u_t, v_t = solution[:dx], solution[dx:]

                # Normalization step
                norm_factor = np.sqrt(2 / np.abs(u_t.T @ Sigma_xx @ u_t) + np.abs(v_t.T @ Sigma_yy @ v_t))
                u_t *= norm_factor
                v_t *= norm_factor

                u_prev, v_prev = u_t, v_t
                suboptimality.append(helper.compute_suboptimality(u_t, u_opt, v_t, v_opt))

            w_A = np.block([
                [lambda_s * Sigma_xx, -Sigma_xy],
                [-Sigma_xy.T, lambda_s * Sigma_yy]
            ])
            w_b = np.block([
                [Sigma_xx @ u_prev],
                [Sigma_yy @ v_prev]
            ])

            if method == "SVRG":
                w_s = self.svrg(w_A, w_b, np.vstack([u_prev, v_prev]), num_epochs=200, lr=1e-3)
            elif method == "ASVRG":
                w_s = self.asvrg(w_A, w_b, np.vstack([u_prev, v_prev]), num_epochs=200, lr=1e-3, momentum=0.9)


            # Compute δ_s
            dot_product_term = 0.5 * np.block([u_prev.T, v_prev.T]) @ np.block([
                [Sigma_xx, np.zeros_like(Sigma_xy)],
                [np.zeros_like(Sigma_xy.T), Sigma_yy]
            ]) @ w_s

            delta_s = 0.5 / (dot_product_term - (2 * np.sqrt(self.epsilon_tilde) / self.delta_tilde))

            # Check stopping criterion
            if delta_s < self.delta_tilde:
                break

            # Update λ_s safely
            lambda_s -= delta_s/2

        lambda_final = lambda_s

        # Phase II: Refinement
        for _ in range(self.m2):
            A = np.block([
                [lambda_final * Sigma_xx, -Sigma_xy],
                [-Sigma_xy.T, lambda_final * Sigma_yy]
            ])
            b = np.block([
                [Sigma_xx @ u_prev],
                [Sigma_yy @ v_prev]
            ])
            if method == "SVRG":
                solution = self.svrg(A, b, np.vstack([u_prev, v_prev]), num_epochs=200, lr=1e-3)
            elif method == "ASVRG":
                solution = self.asvrg(A, b, np.vstack([u_prev, v_prev]), num_epochs=200, lr=1e-3, momentum=0.9)
            u_t, v_t = solution[:dx], solution[dx:]

            # Normalization step
            norm_factor = np.sqrt(2 / max(np.abs(u_t.T @ Sigma_xx @ u_t) + np.abs(v_t.T @ Sigma_yy @ v_t), 1e-10))
            u_t *= norm_factor
            v_t *= norm_factor

            u_prev, v_prev = u_t, v_t

            suboptimality.append(helper.compute_suboptimality(u_t, u_opt, v_t, v_opt))

        # Final normalization
        u_hat = u_prev / np.sqrt(max(np.abs(u_prev.T @ Sigma_xx @ u_prev), 1e-10))
        v_hat = v_prev / np.sqrt(max(np.abs(v_prev.T @ Sigma_yy @ v_prev), 1e-10))

        self.u = u_hat
        self.v = v_hat

        elapsed_time = time.time() - start_time

        return self.u, self.v, elapsed_time, suboptimality

    def transform(self, X, Y):
        """
        Projects X and Y onto their respective first canonical directions.

        Args:
            X: np.ndarray (N, dx) - Data matrix X.
            Y: np.ndarray (N, dy) - Data matrix Y.

        Returns:
            X_proj: np.ndarray (N, 1) - Projected X data onto first canonical direction.
            Y_proj: np.ndarray (N, 1) - Projected Y data onto first canonical direction.
        """
        if self.u is None or self.v is None:
            raise ValueError("Model has not been fitted. Call `fit(X, Y)` first.")

        X_proj = X @ self.u  # (N, dx) @ (dx, 1) → (N, 1)
        Y_proj = Y @ self.v  # (N, dy) @ (dy, 1) → (N, 1)

        return X_proj, Y_proj

    def svrg(self,A, b, u_init, num_epochs=100, batch_size=32, lr=1e-3):
        """
        SVRG optimizer for solving Ax = b iteratively.
        """
        N, d = A.shape
        u = u_init.copy()

        for epoch in range(num_epochs):
            # Compute full gradient
            full_grad = A.T @ (A @ u - b) / N

            # Snapshot of weights
            u_snapshot = u.copy()

            # Mini-batch updates
            shuffled_indices = np.random.permutation(N)
            for i in range(0, N, batch_size):
                batch_indices = shuffled_indices[i:i + batch_size]
                A_batch = A[batch_indices, :]
                b_batch = b[batch_indices, :]

                grad_i = A_batch.T @ (A_batch @ u - b_batch) / len(batch_indices)
                grad_i_snap = A_batch.T @ (A_batch @ u_snapshot - b_batch) / len(batch_indices)

                u -= lr * (grad_i - grad_i_snap + full_grad)

            # Normalize once per epoch to prevent instability
            u /= np.linalg.norm(u) + 1e-8

        return u

    def asvrg(self, A, b, u_init, num_epochs=100, batch_size=128, lr=1e-3, momentum=0.9):
        """
        Accelerated SVRG (ASVRG) optimizer for solving Ax = b iteratively.

        Parameters:
            A (numpy.ndarray): System matrix (N, d)
            b (numpy.ndarray): Target vector (N, 1)
            u_init (numpy.ndarray): Initial solution (d, 1)
            num_epochs (int): Number of outer iterations
            batch_size (int): Mini-batch size
            lr (float): Learning rate
            momentum (float): Momentum factor (typically between 0.8 and 0.95)

        Returns:
            numpy.ndarray: Updated solution u
        """
        N, d = A.shape
        u = u_init.copy()
        v = u.copy()  # Momentum term (like Nesterov acceleration)

        for epoch in range(num_epochs):
            # Compute full gradient
            full_grad = A.T @ (A @ u - b) / N

            # Snapshot of weights
            u_snapshot = u.copy()
            v_snapshot = v.copy()

            # Mini-batch updates
            shuffled_indices = np.random.permutation(N)
            for i in range(0, N, batch_size):
                batch_indices = shuffled_indices[i:i + batch_size]
                A_batch = A[batch_indices, :]
                b_batch = b[batch_indices, :]

                # Compute stochastic gradients
                grad_i = A_batch.T @ (A_batch @ v - b_batch) / len(batch_indices)
                grad_i_snap = A_batch.T @ (A_batch @ u_snapshot - b_batch) / len(batch_indices)

                # Momentum update
                v_new = u + momentum * (u - u_snapshot)

                # ASVRG update (variance reduction applied to momentum-adjusted point)
                u -= lr * (grad_i - grad_i_snap + full_grad)

                # Update momentum
                v = v_new

            # Normalize once per epoch to prevent instability
            u /= np.linalg.norm(u) + 1e-8

        return u