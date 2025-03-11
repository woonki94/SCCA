import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import helper
from ALS_CCA import ALS_CCA
from ALS_f import TALS_CCA
from SI_CCA import SI_CCA


def iterative_cca(cca_model, X1, X2, n_components):
    """
    Iteratively extracts multiple CCA components by fitting ALS_CCA multiple times.

    Parameters:
        cca_model: An instance of ALS_CCA() that supports `.fit()` and `.transform()`.
        X1: ndarray (samples, features) - First dataset.
        X2: ndarray (samples, features) - Second dataset.
        n_components: int - Number of components to extract.

    Returns:
        X1_cca: ndarray (samples, n_components) - Transformed data for X1.
        X2_cca: ndarray (samples, n_components) - Transformed data for X2.
    """

    X1_cca_list, X2_cca_list = [], []
    X1_residual, X2_residual = X1.copy(), X2.copy()

    for i in range(n_components):
        # Fit ALS_CCA on residual data
        cca_model.fit(X1_residual, X2_residual,method="ASVRG")
        X1_cca, X2_cca = cca_model.transform(X1_residual, X2_residual)

        # Store the extracted CCA component
        X1_cca_list.append(X1_cca)
        X2_cca_list.append(X2_cca)

        # Compute residuals by removing projections onto the extracted component
        X1_residual -= np.dot(X1_cca, X1_cca.T).dot(X1_residual)
        X2_residual -= np.dot(X2_cca, X2_cca.T).dot(X2_residual)

    # Stack all components to form the final transformed datasets
    X1_cca_final = np.hstack(X1_cca_list)
    X2_cca_final = np.hstack(X2_cca_list)

    return X1_cca_final, X2_cca_final



# Set random seed for reproducibility
np.random.seed(42)

# Number of samples and features
n_samples = 10000
n_features = 5 # Latent features
noise_level = 0.5

# Generate latent variables
X_latent = np.random.randn(n_samples, n_features)

# Create two different "views" using linear transformations
W1 = np.random.randn(n_features, 5)  # First view transformation
W2 = np.random.randn(n_features, 5)  # Second view transformation

X1 = X_latent @ W1 + noise_level * np.random.randn(n_samples, 5)  # First view with noise
X2 = X_latent @ W2 + noise_level * np.random.randn(n_samples, 5)  # Second view with noise
#print(X2.shape)
# Generate binary labels based on latent features
y = (X_latent[:, 0] + X_latent[:, 1] > 0).astype(int)  # Simple linear decision boundary

# Split into training and testing sets
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, test_size=0.3, random_state=42)

# Train SVM on first view only
svm = SVC(kernel='linear', random_state=42)
svm.fit(X1_train, y_train)
y_pred = svm.predict(X1_test)
acc_svm_single = accuracy_score(y_test, y_pred)
print(f"SVM Accuracy on Single View: {acc_svm_single:.4f}")

# Apply Canonical Correlation Analysis (CCA) to combine views
#cca = CCA(n_components=10)  #
cca = CCA(n_components=5)
tals_cca = TALS_CCA(X1_train, X2_train, 5, 10)
tals_cca_test = TALS_CCA(X1_test, X2_test, 5, 10)
a, b = tals_cca.run()
a1, b1 = tals_cca_test.run()

X1_train_cca, X2_train_cca = cca.fit_transform(X1_train, X2_train)
X1_test_cca, X2_test_cca = cca.transform(X1_test, X2_test)

'''
cca.fit(X1_train, X2_train,method = "SVRG")
X1_train_cca, X2_train_cca = cca.transform( X1_train, X2_train)
cca.fit(X1_test, X2_test,method = "SVRG")
X1_test_cca, X2_test_cca = cca.transform( X1_test, X2_test)
'''


X1_train_cca = X1_train@a
X2_train_cca = X2_train@b
X1_test_cca = X1_test@a
X2_test_cca = X2_test@b

# Concatenate the CCA-transformed views
X_train_cca = np.concatenate([X1_train_cca, X2_train_cca], axis=1)
X_test_cca = np.concatenate([X1_test_cca, X2_test_cca], axis=1)
print("done")

scaler = StandardScaler()
#X_train_cca = scaler.fit_transform(X_train_cca)
#X_test_cca = scaler.transform(X_test_cca)
# Train SVM on the concatenated CCA features
svm_cca = SVC(kernel='linear', random_state=42)
svm_cca.fit(X_train_cca, y_train)
y_pred_cca = svm_cca.predict(X_test_cca)
acc_svm_cca = accuracy_score(y_test, y_pred_cca)
print(f"SVM Accuracy after CCA: {acc_svm_cca:.4f}")

# Compare results
if acc_svm_cca > acc_svm_single:
    print("CCA improved SVM classification accuracy!")
else:
    print("CCA did not improve SVM classification accuracy.")

helper.plot_correlation_points(X1_train_cca,X2_train_cca,title = "dasdasd")
corr = helper.canonical_correlation(X1_train_cca, X2_train_cca)
print(corr)