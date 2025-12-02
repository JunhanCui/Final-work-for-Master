import cripser
import tcripser
import numpy as np
import h5py
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import math
from math import log2

# =========================================================
# Part 1: Basic Test with Matrix
# =========================================================
# To compute the first example with a matrix 3 * 3.
a = np.matrix([[1, 4, 5], [3, 2, 9], [2, 4, 7]])
pd = cripser.computePH(a, maxdim=2)
pd_t = tcripser.computePH(a, maxdim=2)
pd_t_padded = tcripser.computePH(a, maxdim=2, embedded=True)


# =========================================================
# Part 2: Visualization Functions
# =========================================================
# To visualize the slices of a 3D graph in h5 format.
def visualize_h5(file_name, dataset_name):
    with h5py.File(file_name, 'r') as f:
        if dataset_name in f:
            data = f[dataset_name][...]
        else:
            print(f"Dataset {dataset_name} not found in the file.")
            return

    print(f"Data shape: {data.shape}")

    # If dim == 3 with three components (num_slices, height, width)
    if data.ndim == 3:
        num_slices = data.shape[0]
        print(f"Number of slices: {num_slices}")

        plt.imshow(data[0], cmap='gray')
        plt.title('Slice 0')
        plt.show()

        mid_slice = num_slices // 2
        plt.imshow(data[mid_slice], cmap='gray')
        plt.title(f'Slice {mid_slice}')
        plt.show()

        plt.imshow(data[-1], cmap='gray')
        plt.title(f'Slice {num_slices - 1}')
        plt.show()

    # If the image is two-dimensional
    elif data.ndim == 2:
        plt.imshow(data, cmap='gray')
        plt.title('2D Image')
        plt.show()

# To extract the image data set from h5 file f.
def image_data_h5(file_name, dataset_name):
    with h5py.File(file_name, 'r') as f:
        if dataset_name in f:
            data = f[dataset_name][...]
            return data
        else:
            print(f"Dataset {dataset_name} not found in the file.")
            return


# =========================================================
# Part 3: Computation Time Analysis (Benchmark)
# =========================================================
# To compute the velocity of computation we need to calculate the image data for all patients.
# (Assuming path is valid for benchmark)
benchmark_folder_path = r'D:\TFM\ACDC_preprocessed\ACDC_training_volumes'
all_image_data = []

if os.path.exists(benchmark_folder_path):
    file_list = [os.path.join(file) for file in os.listdir(benchmark_folder_path) 
                 if os.path.isfile(os.path.join(benchmark_folder_path, file))]
    
    for file_name in file_list:
        # Assuming we can read 'image' dataset from these files
        try:
            image_data = image_data_h5(os.path.join(benchmark_folder_path, file_name), 'image')
            if image_data is not None:
                all_image_data.append(image_data)
        except:
            pass

if len(all_image_data) > 0:
    # -----------------------------------------------------
    # Algorithm 1: Standard Calculation
    # -----------------------------------------------------
    # To compute the time for the first algorithm for hundred patients.
    print("Starting Method 1 (Standard)...")
    start_time = time.time()

    for image_data in all_image_data:
        # To compute all homology groups using V-construction without padding.
        results = cripser.computePH(image_data, maxdim=2)

    end_time = time.time()

    # To compute the running time.
    elapsed_time_no_efficient = end_time - start_time
    print(f"Method 1 Time: {elapsed_time_no_efficient}")
    
    # Times for five tries (Original comments from thesis)
    # 291.207 + 292.754 + 286.756 + 293.957 + 288.773


    # -----------------------------------------------------
    # Algorithm 2: Optimized/Hybrid Calculation
    # -----------------------------------------------------
    # To compute the time for the second algorithm for hundred patients.
    print("Starting Method 2 (Optimized)...")
    start_time = time.time()

    for image_data in all_image_data:
        # To compute homology groups without H2 using V-construction without padding.
        results_v_no2 = cripser.computePH(image_data, maxdim=1)
        
        # To compute H0 using padded T-construction.
        results_tp_0 = tcripser.computePH(image_data, maxdim=0, embedded=True)

    end_time = time.time()

    # To compute the running time.
    elapsed_time_efficient = end_time - start_time
    print(f"Method 2 Time: {elapsed_time_efficient}")

    # Times for five tries (Original comments from thesis)
    # 279.198 + 278.014 + 279.474 + 279.536 + 275.868
else:
    print("Benchmark data not found, skipping timing analysis.")


# =========================================================
# Part 4: Brain Tumor Classification (Main Logic)
# =========================================================

folder_path_normal = r'D:\TFM\brain_tumor_classification\normal'
folder_path_tumor = r'D:\TFM\brain_tumor_classification\meningioma_tumor'

image_data_normal = []
image_data_tumor = []

# Load Normal Images
if os.path.exists(folder_path_normal):
    file_list = [os.path.join(file) for file in os.listdir(folder_path_normal) 
                 if os.path.isfile(os.path.join(folder_path_normal, file))]
    for file_name in file_list:
        file_path = os.path.join(folder_path_normal, file_name)
        image = Image.open(file_path)
        gray_image = image.convert('L')
        image_data_normal.append(np.array(gray_image))

# Load Tumor Images
if os.path.exists(folder_path_tumor):
    file_list = [os.path.join(file) for file in os.listdir(folder_path_tumor) 
                 if os.path.isfile(os.path.join(folder_path_tumor, file))]
    for file_name in file_list:
        file_path = os.path.join(folder_path_tumor, file_name)
        image = Image.open(file_path)
        gray_image = image.convert('L')
        image_data_tumor.append(np.array(gray_image))


# ---------------------------------------------------------
# Processing NORMAL Images
# ---------------------------------------------------------
# #### The following is for normal #######################

result_normal_0 = []
result_normal_1 = []

for img_array in image_data_normal:
    normal_0 = cripser.computePH(img_array, maxdim=0)
    normal_1 = tcripser.computePH(img_array, maxdim=0, embedded=True)
    result_normal_0.append(normal_0)
    result_normal_1.append(normal_1)

# Eliminate rows with infinite or invalid values for H0
result_normal_cleaned_0 = []
for array in result_normal_0:
    valid_rows = (array[:, 1] <= 255) & (array[:, 1] >= 0) & (array[:, 2] <= 255) & (array[:, 2] >= 0)
    cleaned_array = array[valid_rows]
    result_normal_cleaned_0.append(cleaned_array)

# First convert all negative signs to positive and then swap the positions of birth and death for H1
for array in result_normal_1:
    array[:, 1] *= -1
    array[:, 2] *= -1
    array[:, [1, 2]] = array[:, [2, 1]]

result_normal_cleaned_1 = []
for array in result_normal_1:
    valid_rows = (array[:, 1] <= 255) & (array[:, 1] >= 0) & (array[:, 2] <= 255) & (array[:, 2] >= 0)
    cleaned_array = array[valid_rows]
    result_normal_cleaned_1.append(cleaned_array)

# Total persistence and persistence entropy of H0 for normal images
total_persistence_matrix_0 = np.zeros((len(result_normal_cleaned_0), 1))
for i in range(len(result_normal_cleaned_0)):
    differences = result_normal_cleaned_0[i][:, 2] - result_normal_cleaned_0[i][:, 1]
    result = np.sum(differences)
    total_persistence_matrix_0[i] = result

persistence_entropy_matrix_0 = np.zeros((len(result_normal_cleaned_0), 1))
for i in range(len(result_normal_cleaned_0)):
    differences = 0
    if total_persistence_matrix_0[i, 0] != 0:
        for j in range(result_normal_cleaned_0[i].shape[0]):
            lifespan = result_normal_cleaned_0[i][j, 2] - result_normal_cleaned_0[i][j, 1]
            prob = lifespan / total_persistence_matrix_0[i, 0]
            if prob > 0:
                differences += (-1) * prob * log2(prob)
    persistence_entropy_matrix_0[i] = differences

# Total persistence and persistence entropy of H1 for normal images
total_persistence_matrix_1 = np.zeros((len(result_normal_cleaned_1), 1))
for i in range(len(result_normal_cleaned_1)):
    differences = result_normal_cleaned_1[i][:, 2] - result_normal_cleaned_1[i][:, 1]
    result = np.sum(differences)
    total_persistence_matrix_1[i] = result

persistence_entropy_matrix_1 = np.zeros((len(result_normal_cleaned_1), 1))
for i in range(len(result_normal_cleaned_1)):
    differences = 0
    if total_persistence_matrix_1[i, 0] != 0:
        for j in range(result_normal_cleaned_1[i].shape[0]):
            lifespan = result_normal_cleaned_1[i][j, 2] - result_normal_cleaned_1[i][j, 1]
            prob = lifespan / total_persistence_matrix_1[i, 0]
            if prob > 0:
                differences += (-1) * prob * log2(prob)
    persistence_entropy_matrix_1[i] = differences

# Create an identification column matrix, the column for normal graphs is all 0.
class_normal_matrix = np.zeros((len(result_normal_cleaned_0), 1))

# Merge the features
topo_features_normal_matrix = np.hstack((
    total_persistence_matrix_0, total_persistence_matrix_1,
    persistence_entropy_matrix_0, persistence_entropy_matrix_1,
    class_normal_matrix
))


# ---------------------------------------------------------
# Processing TUMOR Images
# ---------------------------------------------------------
# #### The following is for tumor #######################
# We can actually do the same thing just replacing 'normal' by 'tumor' in the names.

result_tumor_0 = []
result_tumor_1 = []

for img_array in image_data_tumor:
    tumor_0 = cripser.computePH(img_array, maxdim=0)
    tumor_1 = tcripser.computePH(img_array, maxdim=0, embedded=True)
    result_tumor_0.append(tumor_0)
    result_tumor_1.append(tumor_1)

result_tumor_cleaned_0 = []
for array in result_tumor_0:
    valid_rows = (array[:, 1] <= 255) & (array[:, 1] >= 0) & (array[:, 2] <= 255) & (array[:, 2] >= 0)
    cleaned_array = array[valid_rows]
    result_tumor_cleaned_0.append(cleaned_array)

for array in result_tumor_1:
    array[:, 1] *= -1
    array[:, 2] *= -1
    array[:, [1, 2]] = array[:, [2, 1]]

result_tumor_cleaned_1 = []
for array in result_tumor_1:
    valid_rows = (array[:, 1] <= 255) & (array[:, 1] >= 0) & (array[:, 2] <= 255) & (array[:, 2] >= 0)
    cleaned_array = array[valid_rows]
    result_tumor_cleaned_1.append(cleaned_array)

# ############## Total persistence and persistence entropy of H0 for tumor images
total_persistence_tumor_matrix_0 = np.zeros((len(result_tumor_cleaned_0), 1))

for i in range(len(result_tumor_cleaned_0)):
    differences = result_tumor_cleaned_0[i][:, 2] - result_tumor_cleaned_0[i][:, 1]
    result = np.sum(differences)
    total_persistence_tumor_matrix_0[i] = result

persistence_entropy_tumor_matrix_0 = np.zeros((len(result_tumor_cleaned_0), 1))

for i in range(len(result_tumor_cleaned_0)):
    differences = 0
    if total_persistence_tumor_matrix_0[i, 0] != 0:
        for j in range(result_tumor_cleaned_0[i].shape[0]):
            lifespan = result_tumor_cleaned_0[i][j, 2] - result_tumor_cleaned_0[i][j, 1]
            prob = lifespan / total_persistence_tumor_matrix_0[i, 0]
            if prob > 0:
                differences += (-1) * prob * log2(prob)
    persistence_entropy_tumor_matrix_0[i] = differences

# ############## Total persistence and persistence entropy for H1 of the tumor image.
total_persistence_tumor_matrix_1 = np.zeros((len(result_tumor_cleaned_1), 1))

for i in range(len(result_tumor_cleaned_1)):
    differences = result_tumor_cleaned_1[i][:, 2] - result_tumor_cleaned_1[i][:, 1]
    result = np.sum(differences)
    total_persistence_tumor_matrix_1[i] = result

persistence_entropy_tumor_matrix_1 = np.zeros((len(result_tumor_cleaned_1), 1))

for i in range(len(result_tumor_cleaned_1)):
    differences = 0
    if total_persistence_tumor_matrix_1[i, 0] != 0:
        for j in range(result_tumor_cleaned_1[i].shape[0]):
            lifespan = result_tumor_cleaned_1[i][j, 2] - result_tumor_cleaned_1[i][j, 1]
            prob = lifespan / total_persistence_tumor_matrix_1[i, 0]
            if prob > 0:
                differences += (-1) * prob * log2(prob)
    persistence_entropy_tumor_matrix_1[i] = differences

# Create an identification column matrix, the column for tumor images is all 1.
class_tumor_matrix = np.ones((len(result_tumor_cleaned_0), 1))

# Merge the first four columns and the last column together.
topo_features_tumor_matrix = np.hstack((
    total_persistence_tumor_matrix_0, total_persistence_tumor_matrix_1,
    persistence_entropy_tumor_matrix_0, persistence_entropy_tumor_matrix_1,
    class_tumor_matrix
))


# =========================================================
# Part 5: Classification Model
# =========================================================
# Now I merge the topological feature matrices of these two categories together.

if len(topo_features_normal_matrix) > 0 and len(topo_features_tumor_matrix) > 0:
    topo_features_all_matrix = np.vstack((topo_features_normal_matrix, topo_features_tumor_matrix))
    labels = topo_features_all_matrix[:, 4]

    # Use the train_test_split function
    train_set, test_set, train_labels, test_labels = train_test_split(
        topo_features_all_matrix, labels, test_size=1/3, random_state=42
    )

    # Create a logistic regression model
    model = LogisticRegression()

    # train it
    model.fit(train_set, train_labels)

    # predict the test_set
    predictions = model.predict(test_set)

    # Evaluating the Model
    accuracy = accuracy_score(test_labels, predictions)
    conf_matrix = confusion_matrix(test_labels, predictions)
    class_report = classification_report(test_labels, predictions)

    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)
else:
    print("Data not loaded completely. Skipping classification.")