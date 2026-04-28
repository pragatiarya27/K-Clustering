# K-Clustering
Brain MRI image segmentation using K-Means clustering
Medical Image Brain Scanning using Unsupervised Learning (K-Means Clustering)
This project performs brain MRI image segmentation using K-Means clustering, an unsupervised machine learning algorithm. The goal is to divide the image into meaningful regions (clusters) based on pixel intensity.



1. Install and Import Libraries
pip install pydicom
import pydicom
pydicom: Used to read medical imaging files in .dcm (DICOM) format.

2. Load the Brain Scan Image
img = pydicom.dcmread("/content/swi_tra_p2_448_1800000004191561.dcm")
img_array = img.pixel_array.astype(float)
plt.imshow(img_array, cmap="grey")
Reads the DICOM file.
Extracts pixel data and converts it to float.
Displays the grayscale brain MRI image.

3. Reshape Image for Clustering
h, w = img_array.shape
pixels = img_array.reshape(h*w, 1)
Converts the 2D image into a 1D array of pixels.
Required because K-Means expects input in the form of feature vectors.

4. Find Optimal Number of Clusters (Elbow Method)
sum_of_distance = []
for i in range(1, 11):
    model = KMeans(n_clusters=i)
    model.fit_predict(pixels)
    sum_of_distance.append(model.inertia_)
Runs K-Means for cluster values from 1 to 10.
inertia_: Measures how tightly the clusters are formed.
Lower values indicate better clustering.
plt.plot(range(1,11), sum_of_distance, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Sum of Distance")
Plots the Elbow Curve to determine optimal clusters.
The “elbow point” suggests the best number of clusters (here ≈ 3).

5. Apply K-Means Clustering
model = KMeans(n_clusters=3)
group_number = model.fit_predict(pixels)
Applies K-Means with 3 clusters.
Each pixel is assigned a cluster label.

6. Generate Segmented Image
segmented_images = model.cluster_centers_[group_number].reshape(h, w)
plt.imshow(segmented_images, cmap="grey")
Replaces each pixel with its cluster center value.
Produces a segmented version of the brain scan.

7. Separate Brain Regions
for i in range(3):
    cluster_mask = group_number.reshape(h, w) == i
    segment = img_array * cluster_mask
    plt.figure()
    plt.imshow(segment, cmap="grey")
Creates a mask for each cluster.
Extracts and displays individual regions (e.g., tissues, possible tumor areas).
