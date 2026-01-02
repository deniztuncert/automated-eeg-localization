"""
Automated EEG Electrode Localization Pipeline
Author: Deniz Tuncer Tepe
Description: This script processes 3D head scans to identify electrode positions,
aligns them to RAS space, and maps them to the standard 10-20 system.
"""

#-------------------------------------------------------------

# Imports
import numpy as np
import pyvista as pv
from sklearn.cluster import DBSCAN
from scipy.spatial import distance


# Standart 10-20 electrode coordinates. Currently using only 32 electrode layout.
masterdict = {
    "LPA": [-86.0761, -19.9897, -47.9860],    "RPA": [85.7939, -20.0093, -48.0310],    "Nz": [0.0083, 86.8110, -39.9830],
    "Fp1": [-29.4367, 83.9171, -6.9900],    "Fp2": [29.8723, 84.8959, -7.0800],    "F7": [-70.2629, 42.4743, -11.4200],
    "F3": [-50.2438, 53.1112, 42.1920],    "Fz": [0.3122, 58.5120, 66.4620],    "F4": [51.8362, 54.3048, 40.8140],
    "F8": [73.0431, 44.4217, -12.0000],    "FT7": [-80.7750, 14.1203, -11.1350],    "FC3": [-60.1819, 22.7162, 55.5440],
    "FCz": [0.3761, 27.3900, 88.6680],    "FC4": [62.2931, 23.7228, 55.6300],    "FT8": [81.8151, 15.4167, -11.3300],
    "C3": [-65.3581, -11.6317, 64.3580],    "Cz": [0.4009, -9.1670, 100.2440],    "C4": [67.1179, -10.9003, 63.5800],
    "TP7": [-84.8302, -46.0217, -7.0560],    "CP3": [-63.5562, -47.0088, 65.6240],    "CPz": [0.3858, -47.3180, 99.4320],
    "CP4": [66.6118, -46.6372, 65.5800],    "TP8": [85.5488, -45.5453, -7.1300],    "P3": [-53.0073, -78.7878, 55.9400],
    "Pz": [0.3247, -81.1150, 82.6150],    "P4": [55.6667, -78.5602, 56.5610],    "O1": [-29.4134, -112.4490, 8.8390],
    "Oz": [0.1076, -114.8920, 14.6570],    "O2": [29.8426, -112.1560, 8.8000],    "T3": [-84.1611, -16.0187, -9.3460],
    "T5": [-72.4343, -73.4527, -2.4870],    "T4": [85.0799, -15.0203, -9.4900],    "T6": [73.0557, -73.0683, -2.5400],
}

#-------------------------------------------------------------

# First we select the fiducial and reference points to align our mesh and electrode positions.
mesh = pv.read('data/EEG_Clipped_Final.ply')
pv.set_jupyter_backend('trame')

# This is where our reference points will be stored
mesh_elec = {}
# Order of selection
labels_to_pick = ['Nz', 'LPA', 'RPA', 'Cz', 'Oz']
current_pick_index = 0

p = pv.Plotter()
p.add_mesh(mesh, color='antiquewhite')

def callback(point):
    global current_pick_index

    if current_pick_index < len(labels_to_pick):
        label = labels_to_pick[current_pick_index]
        mesh_elec[label] = np.array(point)

        # Visual feedback: Place a sphere and a label showing what is selected and where.
        p.add_mesh(pv.Sphere(radius=3, center=point), color='red')
        p.add_point_labels([point], [label], font_size=15, point_color='red')

        print(f"{label} is saved: {point}")
        current_pick_index += 1

        if current_pick_index == len(labels_to_pick):
            print("\nAll reference points have been selected, you can close this window.")
    else:
        print("All of the refence points have already been selected.")


p.enable_surface_point_picking(callback=callback, show_message=True)
print(f"Please select in given order: {labels_to_pick}")
p.show()

# Automatically assign the selected locations for downstream processing
try:
    nz_coord = mesh_elec['Nz']
    lpa_coord = mesh_elec['LPA']
    rpa_coord = mesh_elec['RPA']
    cz_coord = mesh_elec['Cz']
    oz_coord = mesh_elec['Oz']
except KeyError:
    print("ERROR: Not all points were selected. Please re-run this cell and pick all 5 points.")

#-------------------------------------------------------------

# Here we define a new function to align our imported mesh
def align_mesh(mesh, nz, lpa, rpa):
    # Origin is in the exactly middle of LPA and RPA, as determined by globally accepted standarts.
    origin = (np.array(lpa) + np.array(rpa)) / 2.0

    # X axis goes through RPA and LPA
    x_axis = np.array(rpa) - np.array(lpa)
    x_axis /= np.linalg.norm(x_axis)

    # A vector directed towards Nasion point from the origin. This is our temporary y axis.
    y_temp = np.array(nz) - origin
    y_temp /= np.linalg.norm(y_temp)

    # Finding the z axis
    z_axis = np.cross(x_axis, y_temp)
    z_axis /= np.linalg.norm(z_axis)

    # Finding the actual y axis
    y_axis = np.cross(z_axis, x_axis)

    # Creating a rotation matrix using our newly determined axises
    R = np.vstack([x_axis, y_axis, z_axis])
    # 4x4 Transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ origin

    aligned_mesh = mesh.copy()
    aligned_mesh.transform(T)
    return aligned_mesh, T


# We align our mesh with the defined function
aligned_mesh, transform_matrix = align_mesh(mesh, nz_coord, lpa_coord, rpa_coord)


def transform_points(points, T):
    # Converting (N, 3) points to (N, 4) homogeneous coordinates
    points_homogenous = np.column_stack([points, np.ones(len(points))])

    # Multiplying the points with the transformation matrix (T @ P.T).T
    transformed_points_homogenous = (T @ points_homogenous.T).T

    # Returning back to (N, 3) format
    return transformed_points_homogenous[:, :3]


# Converting our point to an array format
original_points = np.array([nz_coord, lpa_coord, rpa_coord, cz_coord, oz_coord])

# Calculating our new coordinates
new_coords = transform_points(original_points, transform_matrix)
R_labels = ["Nz", "LPA", "RPA", "Cz", "Oz"]

# Creating a dictionary of reference points
new_coords_dict = dict(zip(R_labels, new_coords))

#-------------------------------------------------------------

# Calculating the Gaussian Curvature value of every point on the aligned mesh, electrode sockets have much higher values than the usually-smooth scalp.
aligned_mesh["curv_data"] = aligned_mesh.curvature(curv_type='gaussian')

# We clip below the EEG cap of the mesh, aim here is to reduce unwanted noise.
# WARNING: The Z-coordinate in 'origin' (0, 0, 25) must be tuned manually for each specific scan to avoid clipping actual electrodes.
cap_mesh = aligned_mesh.clip(normal='z', origin=(0, 0, 25), invert=False)

# We apply a lateral (X-axis) clip based on the inter-aural distance (LPA to RPA).
# This ensures that we only process data within the head's width, effectively removing artifacts or scan noise outside the ears.
dist_x = np.linalg.norm(np.array(rpa_coord) - np.array(lpa_coord))
x_limit = (dist_x / 2.0) + 12.5  # Adding a 12.5mm tolerance to prevent clipping edge electrodes

# Applying a mask to eliminate noise
mask = (cap_mesh["curv_data"] > 0.05) & \
       (cap_mesh.points[:, 0] > -x_limit) & \
       (cap_mesh.points[:, 0] < x_limit)

# Determining potential electrode points.
electrode_candidates = cap_mesh.extract_points(mask)

# This is the progress so far:
p = pv.Plotter()
p.add_mesh(cap_mesh, color="white", opacity=0.1)
p.add_mesh(electrode_candidates, color="gold")
p.show()

#-------------------------------------------------------------

points = electrode_candidates.points

# Clustering electrode sites using DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
# This allows the identification of an arbitrary number of electrodes without prior knowledge.
# NOTE: 'eps' (search radius) and 'min_samples' (density threshold) are sensitive to mesh resolution.
db = DBSCAN(eps=5, min_samples=80).fit(points)
cluster_labels = db.labels_  # Extracting cluster labels for each point

# Identifying unique cluster labels to iterate through each detected electrode.
# 'set(labels)' removes duplicates, giving us a list of distinct cluster IDs.
unique_labels = set(cluster_labels)
centers = []

electrode_centers = []

for label in unique_labels:
    if label == -1: continue # -1 means noise, skip

    # We check every point and detect each cluster's center to refer it as an 'electrode'
    class_member_mask = (cluster_labels == label)
    cluster_points = points[class_member_mask]
    centers.append(cluster_points.mean(axis=0))

electrode_centers = np.array(centers)

#-------------------------------------------------------------

# Here we define a function to 'warp' the template electrode positions to our scanned mesh. Using selected reference points.
def warp_electrodes_to_model(template_dict, target_pts):

    # Reference points on the TEMPLATE dictionary
    src_pts = np.array([
        template_dict["LPA"],
        template_dict["RPA"],
        template_dict["Nz"],
        template_dict["Cz"],
        template_dict["Oz"],
    ])

    # Reference points of the mesh
    dst_pts = np.array([
        target_pts["LPA"],
        target_pts["RPA"],
        target_pts["Nz"],
        target_pts["Cz"],
        target_pts["Oz"],
    ])

    # Apply the Affine Transform Matrix (4x3 matrix)
    # Using padded coordinates: [x, y, z, 1]
    def pad(x): return np.hstack([x, np.ones((x.shape[0], 1))])

    # Find the best transform matrix using Least Squares method
    T, residuals, rank, s = np.linalg.lstsq(pad(src_pts), dst_pts, rcond=None)

    # Apply this matrix to all electrode points.
    warped_electrodes = {}
    for label, coords in template_dict.items():
        # Turn the point into a homogenous coordinate [x, y, z, 1]
        c_arr = np.array([coords[0], coords[1], coords[2], 1.0])
        # Calculating the new coordinates
        new_pos = c_arr @ T
        warped_electrodes[label] = new_pos.tolist()

    return warped_electrodes



# Warp the template coordinates to our mesh
final_coords = warp_electrodes_to_model(masterdict, new_coords_dict)

#-------------------------------------------------------------

labeled_electrodes = {}
noise_points = []

for found_point in electrode_centers:
    best_match = None
    min_dist = float('inf')

    for name, std_coord in final_coords.items():
        # Calculate the Euclidean distance
        dist = distance.euclidean(found_point, std_coord)

        if dist < min_dist:
            min_dist = dist
            best_match = name

    # Label the electrode if it's eligible
    if min_dist < 20:
        if best_match in labeled_electrodes and distance.euclidean(labeled_electrodes[best_match], final_coords[best_match]) >= min_dist:
            labeled_electrodes[best_match] = found_point
        elif best_match not in labeled_electrodes:
            labeled_electrodes[best_match] = found_point
        else:
            continue
    else:
        noise_points.append(found_point)

p = pv.Plotter()
p.add_mesh(aligned_mesh, color="antiquewhite", opacity=0.5)

# Add the matches with labels
for name, pos in labeled_electrodes.items():
    p.add_mesh(pv.Sphere(radius=4, center=pos), color="green")
    p.add_point_labels([pos], [name], font_size=12, point_color='green')

# Show 'false' electrodes, subtle of course
for pos in noise_points:
    p.add_mesh(pv.Sphere(radius=2, center=pos), color="gray", opacity=0.5)

p.show()

#-------------------------------------------------------------

with open("electrode_locations.txt", "w") as f:
    f.write("label\tx\ty\tz\n") # Title
    for name, pos in labeled_electrodes.items():
        f.write(f"{name}\t{pos[0]:.2f}\t{pos[1]:.2f}\t{pos[2]:.2f}\n")

print("Electrode coordinates are saved as 'electrode_locations.txt'")