import numpy as np
import time
import helper

import numpy as np
import time

import numpy as np
import time


class ALS_CCA:
    def __init__(self, gamma_x=1e-6, gamma_y=1e-6, max_iter=600, tol=1e-6,n_components=1):
        self.gamma_x = gamma_x
        self.gamma_y = gamma_y
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.u = None
        self.v = None

    def fit(self, X, Y, method = "ALS"):
        N, dx = X.shape
        _, dy = Y.shape

        Sigma_xx = (X.T @ X) / N + self.gamma_x * np.eye(dx)
        Sigma_yy = (Y.T @ Y) / N + self.gamma_y * np.eye(dy)
        Sigma_xy = (X.T @ Y) / N

        u = np.random.randn(dx, self.n_components)
        v =  np.random.randn(dy, self.n_components)

        u = self.gs(u, Sigma_xx)
        v = self.gs(v, Sigma_yy)

        # Compute exact solution using SVD for calculating subopt
        U_svd, S_svd, V_svd = np.linalg.svd(np.linalg.solve(Sigma_xx, Sigma_xy) @ np.linalg.inv(Sigma_yy), full_matrices=False)
        u_opt = U_svd[:, [0]]
        v_opt = V_svd.T[:, [0]]

        suboptimality = []
        #num_passes = 0

        lr = 1e-3
        lr_svrg = 1e-3
        #lr_svrg = 1e-7

        start_time = time.time()
        for _ in range(self.max_iter):
            if method == "SVRG":
                u_new = self.svrg(X, Y, self.gamma_x, u, v,Sigma_xx, num_epochs=100, lr=lr_svrg)
            elif method == "GD":
                u_new = self.gradient_descent(X, Y, self.gamma_x, u, v,Sigma_xx, num_epochs=100, lr=lr)
            elif method == "ASVRG":
                u_new = self.asvrg(X, Y, self.gamma_x, u, v,Sigma_xx, num_epochs=100, lr=lr)
            else:
                u_new = np.linalg.solve(Sigma_xx, Sigma_xy @ v)

            u_new = self.gs(u_new, Sigma_xx)

            if method == "SVRG":
                v_new = self.svrg(Y, X, self.gamma_y, v, u_new,Sigma_yy, num_epochs=100, lr=lr_svrg)
            elif method == "GD":
                v_new = self.gradient_descent(Y, X, self.gamma_y, v, u_new,Sigma_yy, num_epochs=100, lr=lr)
            elif method == "ASVRG":
                v_new = self.asvrg(Y, X, self.gamma_x, v, u_new,Sigma_yy, num_epochs=100, lr=lr)
            else:
                v_new = np.linalg.solve(Sigma_yy, Sigma_xy.T @ u_new)

            v_new = self.gs(v_new, Sigma_yy)

            #u_new /= np.linalg.norm(u_new,axis=0) + 1e-8
            #v_new /= np.linalg.norm(v_new,axis=0) + 1e-8

            #num_passes += 1
            suboptimality.append(helper.compute_suboptimality(u_new[:, [0]], u_opt, v_new[:, [0]], v_opt))


            #if np.linalg.norm(u_new - u) < self.tol and np.linalg.norm(v_new - v) < self.tol:
            #    break

            u, v = u_new, v_new

        end_time = time.time()
        self.u = u
        self.v = v

        return self.u, self.v, end_time - start_time, suboptimality

    def transform(self, X, Y):
        if self.u is None or self.v is None:
            raise ValueError("Model has not been fitted. Call `fit(X, Y)` first.")

        X_proj = X @ self.u  # (N, dx) @ (dx, 1) → (N, 1)
        Y_proj = Y @ self.v  # (N, dy) @ (dy, 1) → (N, 1)
        return X_proj, Y_proj

    #TODO: Make it Nesterov Accelerated Gradient
    def gradient_descent(self, X, Y, gamma, u_init, v_fixed,Sigma_xx, num_epochs=10, lr=1e-3):
        N, dx = X.shape
        u = u_init.copy()

        for epoch in range(num_epochs):
            grad = (X.T @ (X @ u - Y @ v_fixed)) / N + gamma * u
            u_new = u - lr * grad
            u_new /= np.linalg.norm(u_new)
            #u_new = self.gs(u_new, Sigma_xx)
            u = u_new

        return u
    '''512'''
    def svrg(self, X, Y, gamma, u_init, v_fixed, Sigma_xx,  num_epochs=100, batch_size=32, lr=1e-3):
        N, dx = X.shape
        u = u_init.copy()

        for epoch in range(num_epochs):
            shuffled_indices = np.random.permutation(N)

            # Compute full gradient at snapshot point
            full_grad = (X.T @ (X @ u - Y @ v_fixed)) / N + gamma * u
            u_snapshot = u.copy()  # Store a snapshot

            for i in range(0, N, batch_size):
                batch_indices = shuffled_indices[i:i + batch_size]
                x_batch = X[batch_indices, :]
                y_batch = Y[batch_indices, :]
                actual_batch_size = len(batch_indices)

                # Compute gradients for current u and snapshot u_snapshot
                grad_i = (x_batch.T @ (x_batch @ u - y_batch @ v_fixed)) / actual_batch_size + gamma * u
                grad_i_snap = (x_batch.T @ (
                            x_batch @ u_snapshot - y_batch @ v_fixed)) / actual_batch_size + gamma * u_snapshot

                # SVRG update rule
                u -= lr * (grad_i - grad_i_snap + full_grad)

            u /= (np.linalg.norm(u) + 1e-10)

        return u

    '''128'''
    def asvrg(self, X, Y, gamma, u_init, v_fixed,Sigma_xx, num_epochs=10, batch_size=512, lr=1e-3, beta=0.8, l1_reg=0):

        N, dx = X.shape
        u = u_init.copy()
        u_prev = u.copy()

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
            u /= (np.linalg.norm(u) + 1e-10)
            #u = self.gs(u, Sigma_xx)


        return u

    def inprod(self, u, M, v=None):
        # Implementation based on:
        # "Momentum-Based Variance Reduction in Non-Convex SGLD" - Cutkosky & Orabona, NeurIPS 2019
        # Paper: https://papers.neurips.cc/paper_files/paper/2019/file/af3b6a54e9e9338abc54258e3406e485-Paper.pdf
        # Code Repository: https://github.com/BouchardLab/ML_4_prec_prognosis
        """
        Inner product with respect to a metric M.
        """
        v = u if v is None else v
        return u.T @ M @ v

    def gs(self, A, M):
        # Implementation based on:
        # "Momentum-Based Variance Reduction in Non-Convex SGLD" - Cutkosky & Orabona, NeurIPS 2019
        # Paper: https://papers.neurips.cc/paper_files/paper/2019/file/af3b6a54e9e9338abc54258e3406e485-Paper.pdf
        # Code Repository: https://github.com/BouchardLab/ML_4_prec_prognosis
        """
        Modified Gram-Schmidt orthogonalization with respect to metric M.
        Ensures numerical stability and prevents rank loss.
        """
        A = A.copy()

        for i in range(A.shape[1]):
            Ai = A[:, i]

            for j in range(i):
                Aj = A[:, j]
                t = self.inprod(Ai, M, Aj) / self.inprod(Aj, M, Aj)  # Projection coefficient
                Ai -= t * Aj  # Remove projection

            norm = np.sqrt(self.inprod(Ai, M))
            if norm > 1e-8:  # Prevent divide by zero
                A[:, i] = Ai / norm
            else:
                print(f"Warning: Column {i} became numerically unstable and was left unchanged.")
                A[:, i] = Ai  # Keep as is if too small

        return A