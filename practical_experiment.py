import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, ToPILImage
from torchvision.datasets import FashionMNIST
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

from ALS_CCA import ALS_CCA
from SI_CCA import SI_CCA

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Define sample size
num_train_samples = 1000  # Limit training set
num_test_samples = 10000 # Limit test set

transform = Compose([
    ToTensor(),
    Normalize((0.5,), (0.5,))  # Normalizes to [-1, 1]
])

# Load FashionMNIST dataset
train_dataset = FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Convert dataset to PyTorch tensors and move to MPS
X_train_tensor = train_dataset.data.to(device, dtype=torch.float32) / 255.0
y_train_tensor = train_dataset.targets.to(device)
X_test_tensor = test_dataset.data.to(device, dtype=torch.float32) / 255.0
y_test_tensor = test_dataset.targets.to(device)

# Randomly sample training and test data
train_indices = torch.randperm(X_train_tensor.shape[0])[:num_train_samples]
test_indices = torch.randperm(X_test_tensor.shape[0])[:num_test_samples]

X_train_sampled = X_train_tensor[train_indices]
y_train_sampled = y_train_tensor[train_indices]
X_test_sampled = X_test_tensor[test_indices]
y_test_sampled = y_test_tensor[test_indices]


# Define transformation with PIL conversion
#TODO: Try left and right.
to_pil = ToPILImage()
flip_transform = Compose([RandomHorizontalFlip(p=1), ToTensor()])

# Apply horizontal flip correctly (convert tensor -> PIL -> flip -> tensor)
X_train_flipped = torch.stack([flip_transform(to_pil(img.cpu())) for img in X_train_sampled])
X_test_flipped = torch.stack([flip_transform(to_pil(img.cpu())) for img in X_test_sampled])

# Flatten images for PCA & CCA
X_train_flattened = X_train_sampled.view(X_train_sampled.shape[0], -1).cpu().numpy()
X_test_flattened = X_test_sampled.view(X_test_sampled.shape[0], -1).cpu().numpy()
X_train_flipped_flattened = X_train_flipped.view(X_train_flipped.shape[0], -1).cpu().numpy()
X_test_flipped_flattened = X_test_flipped.view(X_test_flipped.shape[0], -1).cpu().numpy()

y_train = y_train_sampled.cpu().numpy()
y_test = y_test_sampled.cpu().numpy()

### ---- PCA  ---- ###
print("\nApplying PCA...")
n_components = 5
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_flattened)
X_test_pca = pca.transform(X_test_flattened)


# Train SVM classifier

svm_pca = SVC(kernel='rbf', gamma='auto', C=20,  random_state=42)
#svm_pca = SVC(kernel='linear', C=20, random_state=42)
svm_pca.fit(X_train_pca, y_train)

# Predict and calculate accuracy
y_pred_pca = svm_pca.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test, y_pred_pca)
end = time.time()


print(f"PCA Accuracy: {accuracy_pca:.2f}")
print("\nPCA Classification Report:")
print(classification_report(y_test, y_pred_pca))

# Plot confusion matrix
cm_pca = confusion_matrix(y_test, y_pred_pca)
ConfusionMatrixDisplay(cm_pca).plot()
plt.title("PCA Confusion Matrix")
plt.show()


#=========CCA===================
#als_cca = ALS_CCA()

# Step 1: Apply PCA
X_train_flipped_pca = pca.transform(X_train_flipped_flattened)
X_test_flipped_pca = pca.transform(X_test_flipped_flattened)
'''
X_train_pca = X_train_flattened
X_test_pca = X_test_flattened

X_train_flipped_pca = X_train_flipped_flattened
X_test_flipped_pca = X_test_flipped_flattened
'''
# Step 2: Apply CCA
print(X_train_pca.shape)
'''
cca = CCA(n_components=5,max_iter=10000)
X_train_cca, X_train_flipped_cca = cca.fit_transform(X_train_pca, X_train_flipped_pca)
X_test_cca, X_test_flipped_cca = cca.transform(X_test_pca, X_test_flipped_pca)
'''

cca = SI_CCA(n_components=5)
cca.fit(X_train_pca, X_train_flipped_pca,method= "ASVRG")
X_train_cca, X_train_flipped_cca = cca.transform(X_train_pca, X_train_flipped_pca)
X_test_cca, X_test_flipped_cca = cca.transform(X_test_pca, X_test_flipped_pca)


#TODO: needs some param setting to get a okay graphs
als_cca_sgd = ALS_CCA(n_components=5)
als_cca_asvrg = ALS_CCA(n_components=5)
si_cca = SI_CCA(n_components=5)
si_cca_asvrg = SI_CCA(n_components=5)
u_als_sgd, v_als_sgd, sgd_elapsed_time,suboptimality_svrg = als_cca_sgd.fit(X_train_pca, X_train_flipped_cca,method = "SVRG")
u_als_asvrg, v_als_asvrg, asvrg_elapsed_time, suboptimality_asvrg = als_cca_asvrg.fit(X_train_pca, X_train_flipped_cca, method="ASVRG")
u_si, v_si, si_elapsed_time,suboptimality_si  = si_cca.fit(X_train_pca,X_train_flipped_cca,method = "SVRG")
u_si_asvrg, v_si_asvrg, si_elapsed_time_asvrg,suboptimality_si_asvrg  = si_cca_asvrg.fit(X_train_pca,X_train_flipped_cca,method = "ASVRG")


def plot_convergence(suboptimality_gd, suboptimality_svrg, suboptimality_asvrg, suboptimality_si, suboptimality_si_asvrg):
    plt.figure(figsize=(6, 4))

    # Plot Gradient Descent (GD)
    plt.plot(suboptimality_gd, label="GD", color="blue", linestyle="-", linewidth=2)

    # Plot Stochastic Variance Reduced Gradient (SVRG)
    plt.plot(suboptimality_svrg, label="SVRG", color="red", linestyle="--", linewidth=2)

    # Plot Accelerated SVRG (ASVRG)
    plt.plot(suboptimality_asvrg, label="ASVRG", color="green", linestyle="-.", linewidth=2)

    plt.plot(suboptimality_si, label="SI-SVRG", color="orange", linestyle="-.", linewidth=2)

    plt.plot(suboptimality_si_asvrg, label="SI-ASVRG", color="grey", linestyle="-.", linewidth=2)

    plt.yscale("log")  # Log scale for better visualization
    plt.xlabel("Passes")
    plt.ylabel("Suboptimality")
    plt.legend()
    plt.title("Convergence Plot")
    plt.grid()
    plt.show()

plot_convergence(suboptimality_svrg, suboptimality_svrg,suboptimality_asvrg,suboptimality_si,suboptimality_si_asvrg)

'''
X_train_cca = X_train_cca[:, 0]
X_train_flipped_cca = X_train_flipped_cca[:,0]
X_test_cca = X_test_cca[:,0]
X_test_flipped_cca = X_test_flipped_cca[:,0]


u, v, elapsed_time = als_cca.fit(X_train_pca, X_train_flipped_pca, SVRG=False)
u_flip, v_flip, elapsed_time = als_cca.fit(X_test_pca, X_test_flipped_pca, SVRG=False)

X_train_cca = u.T@X_train_pca
X_train_flipped_cca = v.T@X_train_flipped_pca
X_test_cca = u_flip.T@X_test_pca
X_test_flipped_cca = u_flip.T@X_test_flipped_pca
'''

canonical_correlations = [np.corrcoef(X_train_cca[:, i], X_train_flipped_cca[:, i])[0, 1] for i in range(X_train_cca.shape[1])]

overall_correlation = np.mean(canonical_correlations)
print("Overall Canonical Correlation:", overall_correlation)

# Step 3: Concatenate CCA components for classification
X_train_cca_combined = np.hstack([X_train_cca, X_train_flipped_cca])
X_test_cca_combined = np.hstack([X_test_cca, X_test_flipped_cca])


# Step 4: Train SVM classifier
#svm_cca = SVC(kernel='linear', C=20, random_state=42)
svm_cca = SVC(kernel='rbf', gamma='auto', C=20,  random_state=42)
svm_cca.fit(X_train_cca_combined, y_train)
print(X_train_cca_combined.shape)
print(y_train.shape)

# Step 5: Evaluate
y_pred_cca = svm_cca.predict(X_test_cca_combined)
accuracy_cca = accuracy_score(y_test, y_pred_cca)

print(f"CCA Accuracy after PCA preprocessing: {accuracy_cca:.2f}")
print("\nCCA Classification Report:")
print(classification_report(y_test, y_pred_cca))

# Step 6: Plot Confusion Matrix
cm_cca = confusion_matrix(y_test, y_pred_cca)
ConfusionMatrixDisplay(cm_cca).plot()
plt.title("CCA Confusion Matrix (with PCA Preprocessing)")
plt.show()

