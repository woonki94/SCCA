import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

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

def plot_correlation_points(X, Y, u, v):
    """
    Plots a scatter plot of the projections of X and Y onto their CCA directions.

    Args:
    X (numpy.ndarray): Data matrix for view 1 (N x dx)
    Y (numpy.ndarray): Data matrix for view 2 (N x dy)
    u (numpy.ndarray): Canonical vector for X (dx,)
    v (numpy.ndarray): Canonical vector for Y (dy,)
    """
    # Ensure proper shape alignment for matrix multiplication
    X_proj = u.T @ X
    Y_proj = v.T @ Y

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X_proj, Y_proj, alpha=0.7, edgecolors='k')

    # Labels and title
    plt.xlabel("Projection of X onto u")
    plt.ylabel("Projection of Y onto v")
    plt.title("Scatter Plot of Canonical Correlation Projections")

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


def generate_correlated_data(dx=10, dy=10, N=100, noise_level=0.1):
    """
    Generates a synthetic dataset where X and Y are strongly correlated
    using a shared latent variable.

    Args:
        dx (int): Dimensionality of X
        dy (int): Dimensionality of Y
        N (int): Number of samples
        noise_level (float): Amount of noise to add

    Returns:
        X (numpy.ndarray): Correlated data matrix for view 1
        Y (numpy.ndarray): Correlated data matrix for view 2
    """
    # Shared latent factors (rank=dx=dy for simplicity)
    Z = np.random.randn(dx, N)

    # Transformation matrices (random but fixed)
    A = np.random.randn(dx, dx)  # Transform latent Z to X
    B = np.random.randn(dy, dx)  # Transform latent Z to Y

    # Generate correlated X and Y
    X = A @ Z + noise_level * np.random.randn(dx, N)  # Add noise
    Y = B @ Z + noise_level * np.random.randn(dy, N)  # Add noise

    return X, Y


def generate_strictly_low_rank_data(dx=100, dy=100, N=1000, rank=10, sparsity=0.8, noise_level=1):
    """
    Generates a strictly low-rank synthetic dataset by projecting onto the top singular vectors.

    Args:
        dx (int): Dimensionality of X
        dy (int): Dimensionality of Y
        N (int): Number of samples
        rank (int): The true rank of the latent structure
        sparsity (float): Fraction of nonzero elements in transformation matrices
        noise_level (float): Amount of noise to add

    Returns:
        X (numpy.ndarray): Strictly low-rank, sparse, high-variance correlated data matrix for view 1
        Y (numpy.ndarray): Strictly low-rank, sparse, high-variance correlated data matrix for view 2
    """
    # Shared latent factors (strongly limited to rank dimensions)
    Z = np.random.randn(rank, N) * 2  # Increase variance by scaling

    # Sparse transformation matrices with controlled rank
    A = np.random.randn(dx, rank)
    B = np.random.randn(dy, rank)

    # Introduce sparsity
    A[np.random.rand(*A.shape) > sparsity] = 0
    B[np.random.rand(*B.shape) > sparsity] = 0

    # Generate X and Y before enforcing strict low-rank structure
    X = A @ Z + noise_level * np.random.randn(dx, N)
    Y = B @ Z + noise_level * np.random.randn(dy, N)

    # Apply Singular Value Thresholding (SVT) to enforce strict low-rank structure
    def enforce_low_rank(matrix, rank):
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        S[rank:] = 0  # Zero out singular values beyond the target rank
        return U @ np.diag(S) @ Vt

    X = enforce_low_rank(X, rank)
    Y = enforce_low_rank(Y, rank)

    return X, Y


def canonical_correlation(X, Y, u, v):
    proj_X = u.T @ X
    proj_Y = v.T @ Y
    return np.corrcoef(proj_X, proj_Y)[0, 1]

def compute_correlation(u, v, X, Y):
    """ Compute the correlation of projected X and Y onto u and v """
    proj_X = u.T @ X
    proj_Y = v.T @ Y
    return np.corrcoef(proj_X, proj_Y)[0, 1]  # Extract correlation value

def reconstruction_error(X, Y, u, v):
    """ Compute reconstruction error for CCA """
    u = u.reshape(-1, 1)  # Ensure column vectors
    v = v.reshape(-1, 1)

    X_proj = X.T @ u  # Project X onto u
    Y_pred = X_proj @ v.T  # Reconstruct Y

    error = np.linalg.norm(Y.T - Y_pred, 'fro')
    return error
