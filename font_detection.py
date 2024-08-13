# font_detection.py
import warnings

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning



def extract_text_regions(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to get binary image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Find contours (text regions)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours, image


def extract_font_features(contour, image):
    # Extract the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Crop the text region
    roi = image[y:y + h, x:x + w]

    # Convert to grayscale
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Resize to standard size for feature extraction
    roi_resized = cv2.resize(roi_gray, (50, 50), interpolation=cv2.INTER_AREA)

    # Calculate features (e.g., mean, standard deviation of pixel intensities)
    mean_intensity = np.mean(roi_resized)
    std_intensity = np.std(roi_resized)

    # Return the feature vector
    return [mean_intensity, std_intensity]


def cluster_fonts(feature_vectors, n_clusters=2):
    # Use KMeans clustering to group similar fonts together
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore", category=ConvergenceWarning)
    feature_vectors = np.unique(feature_vectors, axis=0)

    # Adjust n_clusters to be less than or equal to the number of distinct points
    n_clusters = min(len(feature_vectors), 1267)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(feature_vectors)

    return kmeans.labels_


def detect_font_features(image_path):
    contours, image = extract_text_regions(image_path)

    # Extract font features for each text region
    feature_vectors = []
    for contour in contours:
        features = extract_font_features(contour, image)
        feature_vectors.append(features)

    # Cluster the font features to detect unique fonts
    labels = cluster_fonts(feature_vectors, n_clusters=len(contours))

    # Return the font labels for each text region
    return labels
