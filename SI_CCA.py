import cv2
import numpy as np
from scipy.linalg import solve
import time

import helper

import numpy as np
import time
from scipy.linalg import solve


class SI_CCA:
    def __init__(self, gamma_x=50, gamma_y=50, delta_tilde=1e-3, m1=600, m2=300, epsilon_tilde=1e-3, n_components=1):
        self.gamma_x = gamma_x
        self.gamma_y = gamma_y
        self.delta_tilde = delta_tilde
        self.m1 = m1
        self.m2 = m2
        self.epsilon_tilde = epsilon_tilde
        self.n_components = n_components
        self.u = None
        self.v = None
        self.dx = None
        self.Sigma_xx = None
        self.Sigma_yy = None
        self.lr = 1e-3

    '''
    Find the vectors that maximizes correlation
    '''
    def fit(self, X, Y, method):
        start_time = time.time()

        N, dx = X.shape
        _, dy = Y.shape

        self.dx = dx
        # Compute covariance matrices
        Sigma_xx = (X.T @ X) / N + self.gamma_x * np.eye(dx)
        Sigma_yy = (Y.T @ Y) / N + self.gamma_y * np.eye(dy)
        Sigma_xy = (X.T @ Y) / N

        self.Sigma_xx = Sigma_xx
        self.Sigma_yy = Sigma_yy

        u = np.random.randn(dx, self.n_components)
        v = np.random.randn(dy, self.n_components)

        u = self.gs(u, Sigma_xx)
        v = self.gs(v, Sigma_yy)

        # Compute exact solution using SVD for calculating subopt
        U, S, Vt = np.linalg.svd(np.linalg.solve(Sigma_xx, Sigma_xy) @ np.linalg.inv(Sigma_yy), full_matrices=False)
        u_opt = U[:, [0]]
        v_opt = Vt.T[:, [0]]

        suboptimality = []

        # Phase I: Shift-and-invert preconditioning
        s = 0
        lambda_s = 1 + self.delta_tilde
        u_prev, v_prev = u, v

        while True:
            s += 1
            for _ in range(self.m1):
                #first solve dominant eigenvector usin
                A = np.block([
                    [lambda_s * Sigma_xx, -Sigma_xy],
                    [-Sigma_xy.T, lambda_s * Sigma_yy]
                ])
                b = np.block([
                    [Sigma_xx @ u_prev],
                    [Sigma_yy @ v_prev]
                ])
                #Step(1): Iterative eigenvector estimation via alternating least squares problem to align with the shifted system.
                if method == "SVRG":
                    solution = self.svrg(A, b, np.vstack([u_prev, v_prev]),num_epochs=100, lr=self.lr)
                elif method == "ASVRG":
                    solution = self.asvrg(A, b, np.vstack([u_prev, v_prev]), num_epochs=100, lr=self.lr, momentum=0.8)

                u_t, v_t = solution[:dx], solution[dx:]

                # Normalization step
                u_t = self.gs(u_t, Sigma_xx)
                v_t = self.gs(v_t, Sigma_yy)
                '''
                norm_factor = np.sqrt(2 / max(np.abs(u_t.T @ Sigma_xx @ u_t) + np.abs(v_t.T @ Sigma_yy @ v_t), 1e-10))
                u_t *= norm_factor
                v_t *= norm_factor
                
                u_t /= np.sqrt(u_t.T @ Sigma_xx @ u_t + 1e-8)
                v_t /= np.sqrt(v_t.T @ Sigma_yy @ v_t + 1e-8)
                scale_factor = np.sqrt(2 / (u_t.T @ Sigma_xx @ u_t + v_t.T @ Sigma_yy @ v_t + 1e-10))
                u_t *= scale_factor
                v_t *= scale_factor
                '''

                u_prev, v_prev = u_t, v_t
                suboptimality.append(helper.compute_suboptimality(u_t[:, [0]], u_opt, v_t[:, [0]], v_opt))

            w_A = np.block([
                [lambda_s * Sigma_xx, -Sigma_xy],
                [-Sigma_xy.T, lambda_s * Sigma_yy]
            ])
            w_b = np.block([
                [Sigma_xx @ u_prev],
                [Sigma_yy @ v_prev]
            ])

            if method == "SVRG":
                w_s = self.svrg(w_A, w_b, np.vstack([u_prev, v_prev]), num_epochs=100, lr=self.lr)
            elif method == "ASVRG":
                w_s = self.asvrg(w_A, w_b, np.vstack([u_prev, v_prev]), num_epochs=100, lr=self.lr, momentum=0.9)

            #Step(3): Update λ using the estimated dominant eigen-value to improve spectral localization.
            dot_product_term = 0.5 * np.block([u_prev.T, v_prev.T]) @ np.block([
                [Sigma_xx, np.zeros_like(Sigma_xy)],
                [np.zeros_like(Sigma_xy.T), Sigma_yy]
            ]) @ w_s

            delta_s = 0.5 / (dot_product_term - (2 * np.sqrt(self.epsilon_tilde) / self.delta_tilde))

            # Check stopping criterion
            if np.min(delta_s) < self.delta_tilde:
                break

            # Update λ_s safely
            lambda_s -= delta_s / 2

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
                solution = self.svrg(A, b, np.vstack([u_prev, v_prev]), num_epochs=100, lr=self.lr)
            elif method == "ASVRG":
                solution = self.asvrg(A, b, np.vstack([u_prev, v_prev]), num_epochs=100, lr=self.lr,
                                      momentum=0.9)


            u_t, v_t = solution[:dx], solution[dx:]
            # Normalization step
            u_t = self.gs(u_t, Sigma_xx)
            v_t = self.gs(v_t, Sigma_yy)

            '''
            norm_factor = np.sqrt(2 / max(np.abs(u_t.T @ Sigma_xx @ u_t) + np.abs(v_t.T @ Sigma_yy @ v_t), 1e-10))
            u_t *= norm_factor
            v_t *= norm_factor
           
            u_t /= np.sqrt(u_t.T @ Sigma_xx @ u_t + 1e-8)
            v_t /= np.sqrt(v_t.T @ Sigma_yy @ v_t + 1e-8)
            scale_factor = np.sqrt(2 / (u_t.T @ Sigma_xx @ u_t + v_t.T @ Sigma_yy @ v_t + 1e-10))
            u_t *= scale_factor
            v_t *= scale_factor
            '''

            u_prev, v_prev = u_t, v_t

            #suboptimality.append(helper.compute_suboptimality(u_t[:, [0]], u_opt, v_t[:, [0]], v_opt))

        # Final normalization
        '''
        u_hat = u_prev / np.sqrt(u_prev.T @ Sigma_xx @ u_prev + 1e-10)
        v_hat = v_prev / np.sqrt(v_prev.T @ Sigma_yy @ v_prev + 1e-10)
        '''
        u_hat = u_prev / np.sqrt(np.diag(u_prev.T @ Sigma_xx @ u_prev))
        v_hat = v_prev/ np.sqrt(np.diag(v_prev.T @ Sigma_yy @ v_prev))

        self.u = u_hat
        self.v = v_hat

        elapsed_time = time.time() - start_time

        return self.u, self.v, elapsed_time, suboptimality

    '''
    Return projections
    '''
    def transform(self, X, Y):
        if self.u is None or self.v is None:
            raise ValueError("Model has not been fitted. Call `fit(X, Y)` first.")

        X_proj = X @ self.u
        Y_proj = Y @ self.v

        return X_proj, Y_proj

    def svrg(self, A, b, u_init, num_epochs=100, batch_size=32, lr=1e-3):
        N, d = A.shape
        u = u_init.copy()

        for epoch in range(num_epochs):
            # Compute full gradient
            full_grad = A.T @ (A @ u - b) / N

            u_snapshot = u.copy()

            # Mini-batch updates
            shuffled_indices = np.random.permutation(N)
            for i in range(0, N, batch_size):
                #random shuffle to reduce variance
                batch_indices = shuffled_indices[i:i + batch_size]
                A_batch = A[batch_indices, :]
                b_batch = b[batch_indices, :]

                grad_i = A_batch.T @ (A_batch @ u - b_batch) / len(batch_indices)
                grad_i_snap = A_batch.T @ (A_batch @ u_snapshot - b_batch) / len(batch_indices)

                u -= lr * (grad_i - grad_i_snap + full_grad)

            # Normalize once per epoch to prevent instability
            # u /= np.linalg.norm(u) + 1e-8
            u[:self.dx] = self.gs(u[:self.dx], self.Sigma_xx)
            u[self.dx:] = self.gs(u[self.dx:], self.Sigma_yy)

        return u

    '''
    Nesterov acceleration applied svrg
    '''
    def asvrg(self, A, b, u_init, num_epochs=100, batch_size=512, lr=1e-3, momentum=0.9):
        N, d = A.shape
        u = u_init.copy()
        v = u.copy()  # Momentum term (like Nesterov acceleration)

        for epoch in range(num_epochs):
            shuffled_indices = np.random.permutation(N)
            full_grad = A.T @ (A @ u - b) / N

            # Snapshot of weights
            u_snapshot = u.copy()
            v_snapshot = v.copy()

            # Mini-batch updates
            for i in range(0, N, batch_size):
                #random shuffle to reduce variance
                batch_indices = shuffled_indices[i:i + batch_size]
                A_batch = A[batch_indices, :]
                b_batch = b[batch_indices, :]

                actual_batch_size = len(batch_indices)

                # Compute stochastic gradients
                grad_i = A_batch.T @ (A_batch @ v - b_batch) / actual_batch_size
                grad_i_snap = A_batch.T @ (A_batch @ u_snapshot - b_batch) / actual_batch_size

                # Momentum update
                v_new = u + momentum * (u - u_snapshot)

                u -= lr * (grad_i - grad_i_snap + full_grad)

                # Update momentum
                v = v_new

            # Normalize once per epoch to prevent instability
            u[:self.dx] = self.gs(u[:self.dx], self.Sigma_xx)
            u[self.dx:] = self.gs(u[self.dx:], self.Sigma_yy)

        return u


    def inprod(self, u, M, v=None):
        # Implementation based on:
        # "Momentum-Based Variance Reduction in Non-Convex SGLD" - Cutkosky & Orabona, NeurIPS 2019
        # Paper: https://papers.neurips.cc/paper_files/paper/2019/file/af3b6a54e9e9338abc54258e3406e485-Paper.pdf
        # Code Repository: https://github.com/BouchardLab/ML_4_prec_prognosis
        v = u if v is None else v
        return u.T @ M @ v

    def gs(self, A, M):
        # Implementation based on:
        # "Momentum-Based Variance Reduction in Non-Convex SGLD" - Cutkosky & Orabona, NeurIPS 2019
        # Paper: https://papers.neurips.cc/paper_files/paper/2019/file/af3b6a54e9e9338abc54258e3406e485-Paper.pdf
        # Code Repository: https://github.com/BouchardLab/ML_4_prec_prognosis
        A = A.copy()

        for i in range(A.shape[1]):
            Ai = A[:, i]

            for j in range(i):
                Aj = A[:, j]
                t = self.inprod(Ai, M, Aj) / self.inprod(Aj, M, Aj)
                Ai -= t * Aj

            norm = np.sqrt(self.inprod(Ai, M))
            if norm > 1e-8:
                A[:, i] = Ai / norm
            else:
                print(f"Warning: Column {i} became numerically unstable and was left unchanged.")
                A[:, i] = Ai

        return A