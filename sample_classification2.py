import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize pixel values
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

# Split images into left and right halves
X1_train, X2_train = X_train[:, :, :14].reshape(len(X_train), -1), X_train[:, :, 14:].reshape(len(X_train), -1)
X1_test, X2_test = X_test[:, :, :14].reshape(len(X_test), -1), X_test[:, :, 14:].reshape(len(X_test), -1)

# Apply Linear CCA directly on left and right views
cca = CCA(n_components=10, max_iter=10000)
X1_train_cca, X2_train_cca = cca.fit_transform(X1_train, X2_train)

# Compute the correlation matrix between transformed views
corr_cca_direct = np.corrcoef(X1_train_cca.T, X2_train_cca.T)[:10, 10:]

print(corr_cca_direct)

# Convert to DataFrame for visualization
corr_cca_df = pd.DataFrame(corr_cca_direct,
                           index=[f'Left_CCA_{i}' for i in range(10)],
                           columns=[f'Right_CCA_{i}' for i in range(10)])

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_cca_df, cmap="coolwarm", annot=False, fmt=".2f", center=0)
plt.title("Correlation Between Left and Right Image Halves (CCA-Transformed)")
plt.show()
