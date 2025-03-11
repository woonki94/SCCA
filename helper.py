from datetime import time

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy.sparse import random as sparse_random
from scipy.linalg import svdvals, svd


def plot_points(X,Y):

    # Plot X and Y in different colors
    plt.figure(figsize=(8, 6))

    # Scatter plot for X
    plt.scatter(X[0, :], X[1, :], color='blue', alpha=0.5, label='X')

    # Scatter plot for Y
    plt.scatter(Y[0, :], Y[1, :], color='red', alpha=0.5, label='Y')

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Scatter Plot of X (Blue) and Y (Red)")

def plot_correlation_points(X_proj, Y_proj,title):
    """
    Plots a scatter plot of the projections of X and Y onto their CCA directions.

    Args:
    X (numpy.ndarray): Data matrix for view 1 (N x dx)
    Y (numpy.ndarray): Data matrix for view 2 (N x dy)
    u (numpy.ndarray): Canonical vector for X (dx,)
    v (numpy.ndarray): Canonical vector for Y (dy,)
    """
    # Ensure proper shape alignment for matrix multiplication

    # Create scatter plot with different colors
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


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.abs(a.T @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def plot_correlation(X, Y, u, v,bins=10):
    """
    Plots a heatmap of the density of projections of X and Y onto their CCA directions.

    Args:
    X (numpy.ndarray): Data matrix for view 1 (dx x N)
    Y (numpy.ndarray): Data matrix for view 2 (dy x N)
    u (numpy.ndarray): Canonical vector for X
    v (numpy.ndarray): Canonical vector for Y
    bins (int): Number of bins for the 2D histogram
    """

    X_proj = u.T @ X
    Y_proj = v.T @ Y

    # Compute 2D histogram for density visualization
    hist, x_edges, y_edges = np.histogram2d(X_proj.flatten(), Y_proj.flatten(), bins=bins)

    # Create heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(hist.T, xticklabels=np.round(x_edges, 2), yticklabels=np.round(y_edges, 2),
            cmap="coolwarm", cbar=True, annot=False)

    plt.xlabel("Projection of X onto u")
    plt.ylabel("Projection of Y onto v")
    plt.title("Heatmap of Canonical Correlation Projections")
    plt.show()



def canonical_correlation(X_proj, Y_proj):
    """
    Computes the canonical correlation between two projected datasets.

    Parameters:
        X_proj (numpy.ndarray): Transformed X data (N,)
        Y_proj (numpy.ndarray): Transformed Y data (N,)

    Returns:
        float: Correlation coefficient
    """
    # Ensure inputs are 1D
    X_proj, Y_proj = X_proj.ravel(), Y_proj.ravel()

    # Check for empty arrays
    if X_proj.size == 0 or Y_proj.size == 0:
        raise ValueError("One of the input arrays is empty.")

    # Check for zero variance (constant arrays)
    if np.std(X_proj) == 0 or np.std(Y_proj) == 0:
        raise ValueError("One of the input arrays has zero variance, correlation is undefined.")

    return np.corrcoef(X_proj, Y_proj)[0, 1]


import numpy as np

def generate_correlated_data(dx=10, dy=10, N=100, noise_level=0.1):
    """
    Generates a synthetic dataset where X and Y are strongly correlated
    using a shared latent variable. The output is formatted for scikit-learn CCA.

    Args:
        dx (int): Number of features in X
        dy (int): Number of features in Y
        N (int): Number of samples
        noise_level (float): Amount of noise to add

    Returns:
        X (numpy.ndarray): Correlated data matrix for view 1 (N, dx)
        Y (numpy.ndarray): Correlated data matrix for view 2 (N, dy)
    """
    # Shared latent factors (rank=dx=dy for simplicity)
    Z = np.random.randn(N, dx)  # Now (N, dx) instead of (dx, N)

    # Transformation matrices (random but fixed)
    A = np.random.randn(dx, dx)  # Transform latent Z to X
    B = np.random.randn(dy, dx)  # Transform latent Z to Y

    # Generate correlated X and Y
    X = Z @ A.T + noise_level * np.random.randn(N, dx)  # Now (N, dx)
    Y = Z @ B.T + noise_level * np.random.randn(N, dy)  # Now (N, dy)

    return X, Y


def generate_hard_svd_easy_svrg(dx=100, dy=100, N=50, noise_level=0.1, sparsity=0.95):
    """
    Generates a synthetic dataset where SVRG performs well and SVD struggles.

    Key Modifications:
        - High-dimensional, sparse inputs (dx, dy >> N)
        - Poorly conditioned covariance matrices (high condition number)
        - Small singular value gap (makes SVD slower)

    Args:
        dx (int): Number of features in X (high-dimensional)
        dy (int): Number of features in Y (high-dimensional)
        N (int): Number of samples (small compared to dx, dy)
        noise_level (float): Amount of noise to add
        sparsity (float): Fraction of zero elements (for making matrices sparse)

    Returns:
        X (numpy.ndarray): Correlated, high-dimensional sparse data (N, dx)
        Y (numpy.ndarray): Correlated, high-dimensional sparse data (N, dy)
    """
    # Shared latent factors (low-rank)
    rank = min(dx, dy, N) // 5  # Low-rank structure
    Z = np.random.randn(N, rank)

    # Generate sparse, poorly conditioned transformation matrices
    A = sparse_random(dx, rank, density=1 - sparsity).A * np.random.randn(dx, rank)
    B = sparse_random(dy, rank, density=1 - sparsity).A * np.random.randn(dy, rank)

    # Generate X and Y (correlated, high-dimensional, sparse)
    X = Z @ A.T + noise_level * np.random.randn(N, dx)
    Y = Z @ B.T + noise_level * np.random.randn(N, dy)

    # Make the covariance matrices ill-conditioned
    X = X @ np.diag(1.0 / np.linspace(1, 1e5, dx))  # Large condition number
    Y = Y @ np.diag(1.0 / np.linspace(1, 1e5, dy))  # Large condition number

    # Check singular value gaps to confirm difficulty for SVD
    singular_values = svdvals(X.T @ X)
    print(f"Condition number of X: {singular_values[0] / singular_values[-1]}")

    singular_values = svdvals(Y.T @ Y)
    print(f"Condition number of Y: {singular_values[0] / singular_values[-1]}")

    return X, Y


def generate_controlled_cca_data(N=1000, dx=100, dy=100, rank=2, kappa=300, delta=2, rho1=0.98, rho2=0.9, noise_level=0.1):
    """
    Generate a dataset (X, Y) where we can control:
    - kappa': Condition number of Σxx and Σyy
    - δ: Ratio based on canonical correlations rho1, rho2
    - Noise level

    Parameters:
    - N: Number of samples
    - dx: Dimensionality of X
    - dy: Dimensionality of Y
    - rank: Rank of the shared latent space
    - kappa: Condition number of Σxx, Σyy
    - delta: Defined as (rho1² / (1 - rho2²))
    - rho1, rho2: Canonical correlations
    - noise_level: Standard deviation of added Gaussian noise

    Returns:
    - X, Y: Controlled correlated datasets
    """
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
    """
    Generates a strictly low-rank synthetic dataset by projecting onto the top singular vectors.

    Args:
        N (int): Number of samples
        dx (int): Dimensionality of X
        dy (int): Dimensionality of Y
        rank (int): The true rank of the latent structure
        sparsity (float): Fraction of nonzero elements in transformation matrices
        noise_level (float): Amount of noise to add

    Returns:
        X (numpy.ndarray): Strictly low-rank, sparse, high-variance correlated data matrix for view 1 (N, dx)
        Y (numpy.ndarray): Strictly low-rank, sparse, high-variance correlated data matrix for view 2 (N, dy)
    """
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


def generate_deterministic_cca_data():
    N = 100  # Number of samples
    dx = 10  # Dimensionality of X
    dy = 10  # Dimensionality of Y

    # Define two latent variables with a clear pattern
    t = np.linspace(0, 2 * np.pi, N)
    latent_1 = np.sin(t)  # First shared latent factor
    latent_2 = np.cos(t)  # Second shared latent factor

    # Construct X deterministically
    X = np.zeros((N, dx))
    X[:, 0] = latent_1  # Strong correlation with latent_1
    X[:, 1] = latent_2  # Strong correlation with latent_2
    X[:, 2] = 2 * latent_1 + latent_2  # Linear combination
    X[:, 3] = np.sqrt(np.abs(latent_1))  # Nonlinear transformation
    X[:, 4] = 0.5 * latent_2  # Scaled version
    # Other columns remain zero for sparsity

    # Construct Y deterministically with a different combination
    Y = np.zeros((N, dy))
    Y[:, 0] = latent_2  # Swap latent order
    Y[:, 1] = latent_1
    Y[:, 2] = latent_1 - latent_2
    Y[:, 3] = np.log1p(np.abs(latent_1))  # Log transformation
    Y[:, 4] = np.exp(-np.abs(latent_2))  # Exponential decay
    # Other columns remain zero for sparsity

    return X, Y

def canonical_correlation(X_proj, Y_proj):
    return np.corrcoef(X_proj.ravel(), Y_proj.ravel())[0, 1]


def reconstruction_error(X, Y, u, v):
    """ Compute reconstruction error for CCA """
    u = u.reshape(-1, 1)  # Ensure column vectors
    v = v.reshape(-1, 1)

    X_proj = X.T @ u  # Project X onto u
    Y_pred = X_proj @ v.T  # Reconstruct Y

    error = np.linalg.norm(Y.T - Y_pred, 'fro')
    return error


# Extract patches (row-wise split)
def extract_patches(image, num_patches=10, split_ratio=0.5):
    """
    Dynamically extracts patches from an image, splitting each row-wise patch into X and Y parts.

    Parameters:
        - image: numpy array representing the image.
        - num_patches: number of patches to extract along the height.
        - split_ratio: fraction of width for X_patch (e.g., 0.5 for equal split).

    Returns:
        - Stacked X_patches and Y_patches.
    """
    import numpy as np

    # Get image dimensions
    height, width = image.shape[:2]

    # Calculate the patch height
    patch_height = height // num_patches

    # Determine split columns dynamically
    split_col = int(width * split_ratio)  # Example: split at 50% of the width

    patches_X = []
    patches_Y = []

    for i in range(num_patches):
        start_row = i * patch_height
        end_row = (i + 1) * patch_height

        # Extract patches dynamically
        X_patch = image[start_row:end_row, :split_col].astype(np.float32)

        if width % 2 == 0:
            Y_patch = image[start_row:end_row, split_col:].astype(np.float32)
        else:
            Y_patch = image[start_row:end_row, split_col + 1:].astype(np.float32)

        # Flatten the patches
        X_patch = X_patch.reshape(1, -1)
        Y_patch = Y_patch.reshape(1, -1)

        # Normalize each patch separately
        X_patch = (X_patch - np.mean(X_patch, axis=1, keepdims=True)) / (
                    np.std(X_patch, axis=1, keepdims=True) + 1e-8)
        Y_patch = (Y_patch - np.mean(Y_patch, axis=1, keepdims=True)) / (
                    np.std(Y_patch, axis=1, keepdims=True) + 1e-8)

        patches_X.append(X_patch)
        patches_Y.append(Y_patch)

    return np.vstack(patches_X), np.vstack(patches_Y)


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


def plot_tsne(features, labels, title="t-SNE Visualization of CCA Features", num_classes=10):
    """
    Apply t-SNE and plot the 2D representation of CCA features.

    Parameters:
        - features: The high-dimensional CCA feature representations.
        - labels: The corresponding class labels.
        - title: Title for the plot.
        - num_classes: Number of unique classes.
    """
    # Reduce dimensionality with t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_2d = tsne.fit_transform(features)

    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='jet', alpha=0.7)
    plt.colorbar(scatter, ticks=range(num_classes))
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()





def SVD_solver(X, Y):
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
    u = Sigma_xx_inv_sqrt @ U[:, 0]  # Now (dx,)
    v = Sigma_yy_inv_sqrt @ Vt.T[:, 0]  # Now (dy,)

    return u, v



def compute_suboptimality(U_new, U_opt, V_new, V_opt):
    """
    Compute suboptimality based on the principal angle between subspaces.
    """
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
