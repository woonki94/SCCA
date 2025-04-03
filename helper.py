from datetime import time

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy.sparse import random as sparse_random
from scipy.linalg import svdvals, svd

'''
Plots a scatter plot of the projections of X and Y onto their CCA directions.
'''
def plot_correlation_points(X_proj, Y_proj,title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_proj, Y_proj, color='#228B22', alpha=0.7, s=20, edgecolors='k', label="X Projections")
    plt.scatter(Y_proj, X_proj, color='#CC5500', alpha=0.7, s=20, edgecolors='k', label="Y Projections")

    # Labels and title
    plt.xlabel("Projection of X onto u")
    plt.ylabel("Projection of Y onto v")
    plt.title(f"Scatter Plot of Canonical Correlation Projections using {title}")  # Fixed title formatting
    plt.legend()

    # Show plot
    plt.show()

'''
Computes the canonical correlation between two projected datasets.
'''
def canonical_correlation(X_proj, Y_proj):

    X_proj, Y_proj = X_proj.ravel(), Y_proj.ravel()

    return np.corrcoef(X_proj, Y_proj)[0, 1]

def generate_controlled_cca_data(N=1000, dx=100, dy=100, rank=4, kappa=34080, delta=2, rho1=0.98, rho2=0.9, noise_level=0.1):
    np.random.seed(42)

    # Generate latent factors
    latent_factors = np.random.randn(N, rank)

    # Construct Sigma_xx and Sigma_yy with controlled condition numbers
    Ux, _ = np.linalg.qr(np.random.randn(dx, dx))  # Random orthogonal basis
    Uy, _ = np.linalg.qr(np.random.randn(dy, dy))  # Random orthogonal basis

    # Create singular values that satisfy the condition number
    singular_vals_x = np.linspace(1, kappa, dx)  # Spread from 1 to kappa
    singular_vals_y = np.linspace(1, kappa, dy)

    Sigma_xx = Ux @ np.diag(singular_vals_x) @ Ux.T
    Sigma_yy = Uy @ np.diag(singular_vals_y) @ Uy.T

    # Generate X and Y with low-rank structure
    Wx = np.random.randn(dx, rank)
    Wy = np.random.randn(dy, rank)

    # Ensure canonical correlations match rho1, rho2
    S = np.diag([rho1, rho2] + [0] * (rank - 2))  # First two components have desired correlation

    # Generate correlated factors
    shared_factors = latent_factors @ S.T

    # Create X and Y
    X = shared_factors @ Wx.T + noise_level * np.random.randn(N, dx)
    Y = shared_factors @ Wy.T + noise_level * np.random.randn(N, dy)

    return X, Y


def generate_strictly_low_rank_data(N=1000, dx=100, dy=100, rank=10, sparsity=0.8, noise_level=1):

    # Shared latent factors (rank-limited)
    Z = np.random.randn(N, rank) * 2  # Now (N, rank), increasing variance

    # Sparse transformation matrices with controlled rank
    A = np.random.randn(rank, dx)
    B = np.random.randn(rank, dy)

    # Introduce sparsity
    A[np.random.rand(*A.shape) > sparsity] = 0
    B[np.random.rand(*B.shape) > sparsity] = 0

    # Generate X and Y with the correct shape (N, dx) and (N, dy)
    X = Z @ A + noise_level * np.random.randn(N, dx)
    Y = Z @ B + noise_level * np.random.randn(N, dy)

    # Apply Singular Value Thresholding (SVT) to enforce strict low-rank structure
    def enforce_low_rank(matrix, rank):
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        S[rank:] = 0  # Zero out singular values beyond the target rank
        return U @ np.diag(S) @ Vt

    X = enforce_low_rank(X, rank)
    Y = enforce_low_rank(Y, rank)

    return X, Y

'''
Compute suboptimality based on the principal angle between subspaces.
'''
def compute_suboptimality(U_new, U_opt, V_new, V_opt):

    # Normalize the columns
    def normalize_columns(matrix):
        return matrix / np.linalg.norm(matrix, axis=0, keepdims=True)

    U_new_c = normalize_columns(U_new)
    U_opt_c = normalize_columns(U_opt)
    V_new_c = normalize_columns(V_new)
    V_opt_c = normalize_columns(V_opt)

    # Compute subspace similarity using SVD
    _, s_u, _ = svd(U_new_c.T @ U_opt_c)  # Singular values represent cosine of principal angles
    _, s_v, _ = svd(V_new_c.T @ V_opt_c)

    # Measure how close they are: sum of singular values should be close to dimensionality
    u_similarity = np.sum(s_u) / len(s_u)
    v_similarity = np.sum(s_v) / len(s_v)

    return 1 - (u_similarity + v_similarity) / 2  # Convert to a "distance-like" measure
