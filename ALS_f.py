import numpy as np
from sklearn.cross_decomposition import CCA

import helper


class TALS_CCA:
    def __init__(self, X, Y, k, T):
        """
        Initialize the TALS-CCA algorithm.

        Parameters:
        - X: Data matrix for first view (n_samples × d_x)
        - Y: Data matrix for second view (n_samples × d_y)
        - k: Number of canonical components
        - T: Number of iterations
        """
        self.X = X
        self.Y = Y
        self.k = k
        self.T = T
        self.n, self.dx = X.shape
        _, self.dy = Y.shape

        # Compute covariance matrices
        self.Cxx = (X.T @ X) / self.n + np.eye(self.dx) * 1e-10  # Regularization for stability
        self.Cyy = (Y.T @ Y) / self.n + np.eye(self.dy) * 1e-10
        self.Cxy = (X.T @ Y) / self.n

        # Initialize random Φ and Ψ
        self.Φ = self.gs(np.random.randn(self.dx, k), self.Cxx)
        self.Ψ = self.gs(np.random.randn(self.dy, k), self.Cyy)

    def inprod(self, u, M, v=None):
        """
        Inner product with respect to a metric M.
        """
        v = u if v is None else v
        return u.T @ M @ v

    def gs(self, A, M):
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

    def solve_least_squares(self, A, B):
        """
        Solve the least-squares problem: AX = B using a few iterations of conjugate gradient descent.
        """

        #return np.linalg.solve(A, B)
        return np.linalg.solve(A + 1e-6 * np.eye(A.shape[0]), B)


    def run(self):
        """
        Run the TALS-CCA algorithm.
        """
        for t in range(self.T):
            # Solve for Φ
            Φ_tilde = self.svrg(self.X, self.Y, 1e-3, self.Φ, self.Ψ, num_epochs=50, lr=1e-3)
            self.Φ = self.gs(Φ_tilde, self.Cxx)

            # Solve for Ψ
            Ψ_tilde = self.svrg(self.Y, self.X, 1e-3, self.Ψ,  self.Φ, num_epochs=50, lr=1e-3)
            self.Ψ = self.gs(Ψ_tilde, self.Cyy)

            # Track convergence
            corrs = np.diag(self.Φ.T @ self.Cxy @ self.Ψ)
            print(f"Canonical correlations: {corrs}")
            print(f"Iteration {t}: Norm(Φ) = {np.linalg.norm(self.Φ)}, Norm(Ψ) = {np.linalg.norm(self.Ψ)}")

        return self.Φ, self.Ψ

    def svrg(self, X, Y, gamma, U_init, V_fixed, num_epochs=100, batch_size=8, lr=1e-2):
        N, dx = X.shape
        U = U_init.copy()

        for epoch in range(num_epochs):
            shuffled_indices = np.random.permutation(N)
            full_grad = (X.T @ (X @ U - Y @ V_fixed)) / N + gamma * U  # Compute full gradient

            U_snapshot = U.copy()

            for i in range(0, N, batch_size):
                batch_indices = shuffled_indices[i:i + batch_size]
                x_batch = X[batch_indices, :]
                y_batch = Y[batch_indices, :]

                actual_batch_size = len(batch_indices)

                grad_i = (x_batch.T @ (x_batch @ U - y_batch @ V_fixed)) / actual_batch_size + gamma * U
                grad_i_snap = (x_batch.T @ (
                            x_batch @ U_snapshot - y_batch @ V_fixed)) / actual_batch_size + gamma * U_snapshot

                U -= lr * (grad_i - grad_i_snap + full_grad)  # Update all components at once

            # Normalize each component using Gram-Schmidt
            U = self.gs(U, self.Cxx)

        return U


# Example usage
'''
X,Y = helper.generate_controlled_cca_data(dx=10, dy=10, N=100,noise_level=2) #Low rank, Sparsity ensured Correlated data

k = 3  # Number of canonical components
T = 50  # Number of iterations
cca = CCA(n_components=3, max_iter=10000)
cca.fit(X, Y)
X_proj, Y_proj = cca.transform(X, Y)

tals_cca = TALS_CCA(X, Y, k, T)
a, b = tals_cca.run()
helper.plot_correlation_points(X_proj, Y_proj, title="Built-in")
corr = helper.canonical_correlation(X_proj, Y_proj)

helper.plot_correlation_points(X@a, Y@b, title="t")
corr_svd = helper.canonical_correlation(X@a, Y@b)

print("Φ (Canonical Subspace for X):\n", a)
print("Ψ (Canonical Subspace for Y):\n", b)

print(corr)
print(corr_svd)
'''
