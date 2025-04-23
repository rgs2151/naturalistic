import numpy as np
from skimage.metrics import structural_similarity
from scipy.stats import wasserstein_distance
# from numba import njit, prange

def to_gray(f):
    if f.ndim == 3:
        return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    return f

def l1(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Mean Absolute Error (L1) between two frames.
    Returns the average of absolute pixel differences.
    """
    diff = np.abs(frame1.astype(np.float32) - frame2.astype(np.float32))
    return float(np.mean(diff))

def sum_abs(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Sum of Absolute Errors between two frames.
    Returns the total absolute pixel difference.
    """
    diff = np.abs(frame1.astype(np.float32) - frame2.astype(np.float32))
    return float(np.sum(diff))

def l2(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    L2 Error (Euclidean norm) between two frames.
    Returns sqrt of sum of squared pixel differences.
    """
    diff = frame1.astype(np.float32) - frame2.astype(np.float32)
    return float(np.linalg.norm(diff))

def ssim(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Compute the Structural Similarity Index (SSIM) between two image frames.
    """
    # For multichannel (color) images, set multichannel=True
    return float(structural_similarity(frame1,
                      frame2,
                      data_range=frame2.max() - frame2.min(),
                      multichannel=(frame1.ndim == 3)))

def histogram_diff(frame1: np.ndarray,frame2: np.ndarray, bins: int = 256) -> float:
    """
    Compute the L1 distance between normalized grayscale histograms of two frames.
    """
    gray1 = to_gray(frame1).flatten()
    gray2 = to_gray(frame2).flatten()

    # compute normalized histograms over [0, 255]
    h1, _ = np.histogram(gray1, bins=bins, range=(0, 255), density=True)
    h2, _ = np.histogram(gray2, bins=bins, range=(0, 255), density=True)

    return float(np.sum(np.abs(h1 - h2)))

def wasserstein(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Compute the 1D Earth Mover’s Distance (Wasserstein distance)
    between the grayscale pixel distributions of two frames.
    """
    arr1 = to_gray(frame1).ravel()
    arr2 = to_gray(frame2).ravel()

    return float(wasserstein_distance(arr1, arr2))

def correlation(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Compute the Pearson correlation coefficient between two frames.
    """

    # Flatten to 1D float arrays
    v1 = frame1.astype(np.float32).ravel()
    v2 = frame2.astype(np.float32).ravel()

    # Subtract means
    v1_mean = v1 - v1.mean()
    v2_mean = v2 - v2.mean()

    # Compute Pearson r
    numerator = np.sum(v1_mean * v2_mean)
    denom = np.sqrt(np.sum(v1_mean**2) * np.sum(v2_mean**2))
    if denom == 0:
        return 0.0
    return float(numerator / denom)

def cosine(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Compute the cosine similarity between two frames.
    """
    # Flatten to 1D float arrays
    v1 = frame1.astype(np.float32).ravel()
    v2 = frame2.astype(np.float32).ravel()

    # Compute dot product and norms
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))