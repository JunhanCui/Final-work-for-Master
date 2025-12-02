"""
Master's Thesis Implementation: 
Topological Data Analysis (TDA) for Medical Image Classification (Cubical Homology)

Refactored and Normalized Version
"""


# Standard Library
import os
import time
import math
from math import log2

# Third-Party Libraries
import numpy as np
import h5py
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# TDA Libraries (Cubical Ripser)
# Assuming these are available in your environment
import cripser
import tcripser


# ==============================================================================
# 2. HELPER FUNCTIONS: IO & VISUALIZATION
# ==============================================================================

def visualize_h5(file_name, dataset_name):
    """
    Visualizes the cross-sectional slices of a 3D medical volume stored in HDF5 format.
    
    This function extracts the first, middle, and last 2D slices from the 3D volume 
    to provide a quick overview of the anatomical structure and scan quality.
    
    Parameters
    ----------
    file_name : str
        Path to the .h5 file.
    dataset_name : str
        Key to access the image data within the HDF5 file (e.g., 'image').
    """
    try:
        with h5py.File(file_name, 'r') as f:
            if dataset_name not in f:
                print(f"Dataset {dataset_name} not found in the file.")
                return
            data = f[dataset_name][...]
            
        print(f"Data shape: {data.shape}")

        # Case 1: 3D Data (num_slices, height, width)
        if data.ndim == 3:
            num_slices = data.shape[0]
            print(f"Number of slices: {num_slices}")
            
            # Slice indices
            indices = [0, num_slices // 2, num_slices - 1]
            titles = ['Slice 0 (First)', f'Slice {indices[1]} (Middle)', f'Slice {indices[2]} (Last)']
            
            for idx, title in zip(indices, titles):
                plt.figure()
                plt.imshow(data[idx], cmap='gray')
                plt.title(title)
                plt.show()

        # Case 2: 2D Data
        elif data.ndim == 2:
            plt.figure()
            plt.imshow(data, cmap='gray')
            plt.title('2D Image')
            plt.show()
            
    except Exception as e:
        print(f"Error reading h5 file: {e}")


def image_data_h5(file_name, dataset_name):
    """
    Loads raw image data from an HDF5 file into a NumPy array.
    
    Parameters
    ----------
    file_name : str
        Path to the .h5 file.
    dataset_name : str
        Key for the dataset (e.g., 'image').
        
    Returns
    -------
    np.ndarray or None
        The extracted image data array, or None if the dataset is not found.
    """
    with h5py.File(file_name, 'r') as f:
        if dataset_name in f:
            return f[dataset_name][...]
        else:
            print(f"Dataset {dataset_name} not found.")
            return None


def load_images_from_folder(folder_path):
    """
    Load all images from a folder, convert to Grayscale ('L'), 
    and return as a list of numpy arrays.
    """
    image_data_list = []
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found: {folder_path}")
        return image_data_list

    file_list = [f for f in os.listdir(folder_path) 
                 if os.path.isfile(os.path.join(folder_path, f))]
    
    print(f"Loading {len(file_list)} images from {folder_path}...")
    
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        try:
            image = Image.open(file_path)
            gray_image = image.convert('L')
            image_data_list.append(np.array(gray_image))
        except Exception as e:
            print(f"Error loading image {file_name}: {e}")
            
    return image_data_list


# ==============================================================================
# 3. CORE ALGORITHMS: TOPOLOGY & FEATURE ENGINEERING
# ==============================================================================

def clean_persistence_diagram(diagram, dim=0):
    """
    Filters and normalizes the raw persistence diagram.
    
    This process removes topological noise (features with invalid birth/death times)
    and corrects the coordinate representation for H1 features (loops) to ensure 
    consistency with standard TDA conventions.
    
    Parameters
    ----------
    diagram : np.ndarray
        The raw persistence diagram output from the filtration.
    dim : int, optional
        Homology dimension (0 for components, 1 for loops). Defaults to 0.
        
    Returns
    -------
    np.ndarray
        A cleaned persistence diagram containing only valid features within the [0, 255] range.
    """
    # Create a copy to avoid modifying original data
    clean_dg = diagram.copy()
    
    if dim == 1:
        # Specific preprocessing for H1 (from original code logic)
        # Convert signs and swap birth/death columns
        clean_dg[:, 1] *= -1
        clean_dg[:, 2] *= -1
        clean_dg[:, [1, 2]] = clean_dg[:, [2, 1]]

    # Filter invalid rows (values must be within [0, 255])
    # Column 1 is Birth, Column 2 is Death in the cleaned version
    valid_mask = (
        (clean_dg[:, 1] <= 255) & (clean_dg[:, 1] >= 0) & 
        (clean_dg[:, 2] <= 255) & (clean_dg[:, 2] >= 0)
    )
    return clean_dg[valid_mask]


def calculate_entropy_and_persistence(diagram):
    """
    Computes scalar topological invariants from a persistence diagram.
    
    This function extracts two key summary statistics:
    1. Total Persistence: The sum of lifetimes of all topological features.
    2. Persistence Entropy: A Shannon-entropy-like measure quantifying the 
       diversity of feature lifetimes.
       
    Parameters
    ----------
    diagram : np.ndarray
        A cleaned persistence diagram (N x 2 array of [birth, death]).
        
    Returns
    -------
    tuple
        (total_persistence, persistence_entropy)
    """
    if len(diagram) == 0:
        return 0.0, 0.0

    # lifetimes = death - birth
    lifetimes = diagram[:, 2] - diagram[:, 1]
    
    # 1. Total Persistence
    total_persistence = np.sum(lifetimes)
    
    # Avoid division by zero
    if total_persistence == 0:
        return 0.0, 0.0

    # 2. Persistence Entropy
    # Formula: - sum ( (L_i / L_tot) * log2(L_i / L_tot) )
    probs = lifetimes / total_persistence
    # Filter out zero probabilities to avoid log2(0) error
    probs = probs[probs > 0] 
    entropy = -np.sum(probs * np.log2(probs))
    
    return total_persistence, entropy


def extract_topological_features(image_list, label_value):
    """
    Batch processes a list of images to generate a topological feature matrix.
    
    For each image, this pipeline:
    1. Computes Cubical Homology for H0 (components) and H1 (loops).
    2. Cleans the resulting persistence diagrams.
    3. Calculates Total Persistence and Persistence Entropy for both dimensions.
    4. Appends the class label.
    
    Parameters
    ----------
    image_list : list of np.ndarray
        List of 2D image arrays.
    label_value : int
        Class label (e.g., 0 for Normal, 1 for Tumor).
        
    Returns
    -------
    np.ndarray
        A feature matrix of shape (N_images, 5), columns:
        [H0_Persistence, H1_Persistence, H0_Entropy, H1_Entropy, Label]
    """
    features = []
    
    for idx, img_array in enumerate(image_list):
        # 1. Compute Homology
        # H0 using standard cripser
        dgm_0 = cripser.computePH(img_array, maxdim=0)
        # H1 using padded tcripser (embedded=True)
        dgm_1 = tcripser.computePH(img_array, maxdim=0, embedded=True)
        
        # 2. Clean Diagrams
        clean_0 = clean_persistence_diagram(dgm_0, dim=0)
        clean_1 = clean_persistence_diagram(dgm_1, dim=1)
        
        # 3. Calculate Scalar Features
        pers_0, ent_0 = calculate_entropy_and_persistence(clean_0)
        pers_1, ent_1 = calculate_entropy_and_persistence(clean_1)
        
        features.append([pers_0, pers_1, ent_0, ent_1, label_value])
        
    return np.array(features)


# ==============================================================================
# 4. BENCHMARKING
# ==============================================================================

def run_benchmark_test(folder_path):
    """
    Compares the execution time of two different PH computation strategies.
    """
    print("\n--- Starting Benchmark Test ---")
    
    # Load data
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
    all_image_data = []
    for f in file_list:
        path = os.path.join(folder_path, f)
        # Assuming 'image' is the dataset name
        data = image_data_h5(path, 'image') 
        if data is not None:
            all_image_data.append(data)
            
    if not all_image_data:
        print("No data found for benchmark.")
        return

    # Method 1: Standard ComputePH (maxdim=2)
    start_time = time.time()
    for img in all_image_data:
        _ = cripser.computePH(img, maxdim=2)
    time_method_1 = time.time() - start_time
    print(f"Method 1 (maxdim=2) Time: {time_method_1:.4f} seconds")

    # Method 2: Hybrid Efficient Approach
    start_time = time.time()
    for img in all_image_data:
        _ = cripser.computePH(img, maxdim=1) # H0, H1
        _ = tcripser.computePH(img, maxdim=0, embedded=True) # Specific H0/H1 check
    time_method_2 = time.time() - start_time
    print(f"Method 2 (Hybrid) Time: {time_method_2:.4f} seconds")


# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    
    # --- A. Setup Paths (请根据实际情况修改路径) ---
    base_path = r"D:\TFM\brain_tumor_classification"
    normal_path = os.path.join(base_path, "normal")
    tumor_path = os.path.join(base_path, "meningioma_tumor")
    
    # --- B. Load Images ---
    print("\n[1/4] Loading Images...")
    imgs_normal = load_images_from_folder(normal_path)
    imgs_tumor = load_images_from_folder(tumor_path)
    
    if not imgs_normal or not imgs_tumor:
        print("Error: Images not loaded. Please check the paths.")
    else:
        # --- C. Feature Extraction ---
        print("\n[2/4] Extracting Topological Features...")
        
        # Process Normal Images (Label = 0)
        print(f"Processing {len(imgs_normal)} normal images...")
        feats_normal = extract_topological_features(imgs_normal, label_value=0)
        
        # Process Tumor Images (Label = 1)
        print(f"Processing {len(imgs_tumor)} tumor images...")
        feats_tumor = extract_topological_features(imgs_tumor, label_value=1)
        
        # Combine Datasets
        # Matrix Structure: [Pers_H0, Pers_H1, Ent_H0, Ent_H1, Class_Label]
        dataset_matrix = np.vstack((feats_normal, feats_tumor))
        print(f"Final Dataset Shape: {dataset_matrix.shape}")
        
        # --- D. Machine Learning Classification ---
        print("\n[3/4] Training Classifier...")
        
        # Split Features (X) and Labels (y)
        X = dataset_matrix[:, :4]
        y = dataset_matrix[:, 4]
        
        # Train/Test Split (33% test size, as in original code)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42
        )
        
        # Logistic Regression
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        # Prediction
        predictions = model.predict(X_test)
        
        # --- E. Evaluation ---
        print("\n[4/4] Evaluation Results:")
        accuracy = accuracy_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        report = classification_report(y_test, predictions)
        
        print(f"Accuracy: {accuracy:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(report)

        # --- Optional: Benchmark Test (Commented out) ---
        # benchmark_path = r"D:\TFM\ACDC_preprocessed\ACDC_training_volumes"
        # run_benchmark_test(benchmark_path)