import matplotlib.pyplot as plt
import numpy as np
import sys
from medpy.io import load
import cv2


'''
Returns an array of needle and image transformation matrices associated with each image
PARAMS:
- file_name: the path to the .mha file containing ultrasound image data
RETURNS: A list of needle transformation matrices and corresponding image transformation matrices 
'''
def get_transforms_from_MHA(file_name):
    needle_transforms = []
    image_transforms = []
    image_to_probe = np.array(
        [[-0.00916059, 0.100599, 0.00250033, -98.5887], [0.100611, 0.000965076, -0.00196802, -24.8155],
         [-0.00199131, 0.00248182, -0.100584, 16.0973], [0.0, 0.0, 0.0, 1.0]]) # A constant obtained from 3D Slicer

    # Parse needle nad image transform matrix data for each time step in the .mha file
    with open(file_name, errors='ignore') as f:
        line = f.readline()
        i = 0
        while line:
            if "StylusToReferenceTransform =" in line:
                elements = [float(n) for n in line[line.find('=') + 2:].split(' ')[:-1]]
                l = 0
                for k in range(0,3):
                    for j in range(0,4):
                        needle_transforms[i,k,j] = elements[l]
                        l = l + 1
                i = i + 1
            if "ProbeToReferenceTransform =" in line:
                elements = [float(n) for n in line[line.find('=') + 2:].split(' ')[:-1]]
                l = 0
                probe_to_reference = np.zeros((4,4))
                for k in range(0, 4):
                    for j in range(0, 4):
                        probe_to_reference[k, j] = elements[l]
                        l = l + 1
                image_to_reference = np.matmul(probe_to_reference, image_to_probe)
                image_transforms[i] = image_to_reference
            elif line.startswith("DimSize"):
                dim_size = line.split(' ')
                data_count = int(dim_size[-1]) # Get number of transforms
                needle_transforms = np.zeros((data_count, 3, 4))
                image_transforms = np.zeros((data_count, 4, 4))
            line = f.readline()
    return needle_transforms, image_transforms


'''
Computes the intersection of the needle and the plane of the ultrasound image at a particular time
PARAMS:
- needle_transforms: the transformation matrices of the needle
- image_transforms: the transformation matrices of the ultrasound images
- idx: the current timestep
RETURNS: An (x,y) coordinate representing the point at which the needle intersects with the ultrasound image
'''
def compute_intersection(needle_transforms, image_transforms, idx):
    # Compute image corners
    image_corners = np.zeros((4,4,1))

    # Get index coordinates of the image corners. In this experiment, the image is of size 356x589.
    ijk_top_right = np.array([[0],[588],[0],[1]])
    ijk_top_left = np.array([[0],[0],[0],[1]])
    ijk_bottom_right = np.array([[355],[588],[0],[1]])
    ijk_bottom_left = np.array([[355],[0],[0],[1]])

    # Transform image corner points to RAS space
    ras_top_right = np.matmul(image_transforms[idx],ijk_top_right)
    ras_top_left = np.matmul(image_transforms[idx],ijk_top_left)
    ras_bottom_right = np.matmul(image_transforms[idx],ijk_bottom_right)
    ras_bottom_left = np.matmul(image_transforms[idx],ijk_bottom_left)

    # Get the 3D points of the image corners
    image_corners = [ras_top_left, ras_top_right, ras_bottom_left, ras_bottom_right]

    # Compute line P1 + t*U, where U is a direction vector
    p1 = np.matmul(needle_transforms[idx], np.array([[0], [0], [0], [1]]))
    p2 = np.matmul(needle_transforms[idx], np.array([[0], [0], [-10], [1]]))
    U = p2 - p1

    # Compute plane. V is a point on the plane and n is its normal vector
    v = image_corners[0][:-1]
    N = np.cross(image_corners[1][:-1] - image_corners[0][:-1], image_corners[2][:-1] - image_corners[0][:-1], axis=0)
    d = -(N[0]*v[0] + N[1]*v[1] + N[2]*v[2])

    # Ensure line and plane are not parallel
    if np.vdot(U, N) == 0.0:
        return None

    # Compute the intersection point, i
    t = -(N[0]*p1[0] + N[1]*p1[1] + N[2]*p1[2] + d) / np.vdot(N, U)
    i = p1 + t*U
    i = np.append(i, 1.0) # Initially assume centroid is within the image

    # Transform i to the ijk space and round to nearest integer
    r = np.matmul(np.linalg.inv(image_transforms[idx]), i)
    r = np.rint(r)

    # Return a complete data label
    y = np.zeros((3,))
    y[0] = r[0]
    y[1] = r[1]
    y[2] = 1.0
    return y


'''
Resizes all of the ultrasound images.
PARAMS:
- raw_images: the image data from the .mha file
- w: the width (and height) of the desired output image
RETURNS:
- the list of images resized to the appropriate dimensions to be fed as input into the neural network
'''
def get_resized_images(raw_images, w):
    images = np.zeros((raw_images.shape[0], w, w, 1)) # Create a placeholder for square images of the new size
    for i in range(raw_images.shape[0]):
        images[i] = np.expand_dims(cv2.resize(raw_images[i], (w, w)), axis=3) # Resize an image
    return images



# COMMAND LINE ARGUMENTS:
# argv[0]: Image sequence path
# argv[1]: Append to or overwrite data
# Obtain paths to data from command line arguments
data_path = '../dataN3/data.mha'
append_flag = 'o'
if(len(sys.argv) >= 6):
    data_path = sys.argv[3]
    append_flag = sys.argv[5]

# Get raw image data
images, headers = load(data_path)
images = np.rollaxis(images, 2, 0)
images = np.expand_dims(images, axis=3)

# Get list of needle tip RAS transforms
needle_transforms, image_transforms = get_transforms_from_MHA(data_path, images)

# Compute the intersections of the images
intersections = np.zeros((images.shape[0], 3))
for j in range(images.shape[0]):
    i = compute_intersection(needle_transforms, image_transforms, j)
    if (i[0] < 0 or i[0] >= images.shape[1] or i[1] < 0 or i[1] >= images.shape[2]):
        i[2] = 0  # The fiducial is not in the image
    else:
        i[2] = 1
    intersections[j] = i

orig_w = images.shape[1]
orig_h = images.shape[2]
images = get_resized_images(images, 128) # Resize the images
images = images/255 # Each pixel in the image is represented by 1 byte
print(images.shape)
intersections[:,0] = intersections[:,0]/(orig_w - 1) # X-coordinate is a fraction of the total image width
intersections[:,1] = intersections[:,1]/(orig_h - 1) # Y-coordinate is a fraction of the total image height
intersections = 2.0*intersections - 1.0 # Normalizing of intersection coordinates to [-1, 1]

np.save("images", images) # Save list of images
np.save("intersections", intersections) # Save corresponding intersections, and a flag for whether an intersection exists

# Test an image-centroid pair
test_idx = 44
print("IJK space intersection: " + str(intersections[test_idx]));

# Display an image and its intersection to demonstrate correct intersection computation
sample_img = np.squeeze(images[test_idx], axis=2)
plt.imshow(sample_img, cmap="gray") # Display an image
plt.scatter(intersections[test_idx][1]*96, intersections[test_idx][0]*96, color='r', s=5)
plt.show()





