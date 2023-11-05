import argparse
import os

import cv2
import UI
import numpy as np
from scipy.spatial import Delaunay

# Output image video
SIZE = 256

def create_transformation_fields(landmarks_img1, landmarks_img2, num_frames):
    """
    :param str landmarks_img1: list of landmarks coordinates for image 1
    :param str landmarks_img2: list of landmarks coordinates for image 2
    :param str num_frames: Number of frames the final morphed video consist of
    :return: transformation field, of shape (num_frames, num_landmarks, 2).
     Every list represents one frame, and every element represents coordinate for
     that landmark.
    """
    # Ensure both lists have the same number of landmarks
    if len(landmarks_img1) != len(landmarks_img2):
        raise ValueError("Both lists of landmarks must have the same number of landmarks.")

    # Calculate t values for interpolation
    t_values = np.linspace(0, 1, num_frames)

    # Initialize a list to store transformation fields for each landmark
    fields = []

    # Loop over each landmark
    for i in range(len(landmarks_img1)):
        # Extract landmark coordinates for both images
        landmark_img1 = np.array(landmarks_img1[i])
        landmark_img2 = np.array(landmarks_img2[i])

        # Calculate the displacement vector between the two landmarks
        displacement = landmark_img2 - landmark_img1

        # Create a list to store the transformation field for the current landmark
        field = []

        # Interpolate the position of the landmark over the specified number of frames
        for t in t_values:
            # Calculate the intermediate position of the landmark
            intermediate_position = [int(element) for element in (landmark_img1 + (displacement * t))]

            # Append the intermediate position to the transformation field
            field.append(intermediate_position)

        # Append the transformation field for the current landmark to the list
        fields.append(field)
    # Put it into the right shape
    transposed_fields = np.array(fields).transpose(1, 0, 2)
    return transposed_fields


def shepard_interpolate(image, landmarks, landmark_diff):
    """
    :param image: input image
    :param landmarks: initial (non-transformed) landmarks. Value is different
    for 2 images
    :param landmark_diff: difference in new and initial landmark positions for
    each landmark
    :return:
    A mesh representing the interpolated coordinates for color mapping.
    """
    height, width, _ = image.shape

    # Generate grid of y and x coordinates
    y, x = np.ogrid[:height, :width]

    # Initialize arrays to store interpolated x and y change values and total weights
    # x_change and y_change values will be added to the original coordinates in the end
    x_change = np.zeros((height, width))
    y_change = np.zeros((height, width))
    total_weight = np.zeros((height, width))

    # Iterate through landmarks and difference values for interpolation
    for landmark, diff in zip(landmarks, landmark_diff):
        landmarkY, landmarkX = landmark

        # Calculate the distance between each pixel and the original landmark position
        distance = np.sqrt((x - landmarkX) ** 2 + (y - landmarkY) ** 2)

        # Change distance value 0 to 1, where encountered.
        # At the end of this function, these pixels will be reprocessed,
        # This is just to prevent division by zero
        distance[distance == 0] = 1

        # Calculate weights
        weights = 1 / (distance ** 2)
        total_weight += weights

        # Update x and y change values based on the differences and weights
        x_change += diff[1] * weights
        y_change += diff[0] * weights

    # Calculate final interpolated x and y coordinates considering the mask and total weights
    mask = (x_change == 0) & (y_change == 0)

    # If x_out and y_out are not zero, add the change divided by total_weight to the coordinates
    x_change = np.where(mask, x, x + x_change / total_weight)
    y_change = np.where(mask, y, y + y_change / total_weight)

    # Construct a mesh for the interpolated coordinates
    warped_space = np.empty((height, width), dtype=[('x_out', int), ('y_out', int)])
    warped_space['x_out'], warped_space['y_out'] = y_change, x_change

    # Landmark positions should be mapped to itself
    for landmark, diff in zip(landmarks, landmark_diff):
        landmarkY, landmarkX = landmark
        warped_space[landmarkY][landmarkX] = (diff[0] + landmarkY), (diff[1] + landmarkX)

    return warped_space


def applyMesh(image, mesh):
    """
    :param image: input image
    :param mesh: mesh for mapping
    :return: Transformed image
    """

    # Initialize the output image with zeros
    warped_image = np.zeros_like(image)
    height, width = mesh.shape

    # Iterate through every pixel
    for y in range(height):
        for x in range(width):
            # Extract coordinates and ensure they are in valid range
            warped_x, warped_y = mesh[x, y]
            warped_x = max(0, min(warped_x, width - 1))
            warped_y = max(0, min(warped_y, height - 1))

            # Map the color of the pixel using mesh coordinates
            warped_image[y, x] = image[warped_y, warped_x]
    return warped_image


def landmark_change(landmarks1, landmarks2):
    """
    :return: returns the difference between two landmarks for each landmark
    """

    # Check if the input lists have the same size
    if len(landmarks1) != len(landmarks2):
        raise ValueError("Input lists must have the same size")
    result = []

    # Iterate through the lists and compute the difference for each pair of 2-element lists
    for i in range(len(landmarks1)):
        if len(landmarks1[i]) != 2 or len(landmarks2[i]) != 2:
            raise ValueError("Each element in the input lists must be a 2-element list")
        diff = [landmarks1[i][0] - landmarks2[i][0], landmarks1[i][1] - landmarks2[i][1]]
        result.append(diff)

    return result


# This function subdivides triangles into smaller triangles by adding midpoints.

def subdivdie(tri):
    """
    :param tri: Contains information about edges and vertices of the triangles.
    :return: A list of vertices representing the resulting triangles after subdivision.
    """
    # Make a copy of the original vertices
    resulting_vertices = tri.points.copy()

    # Iterate through each triangle in the given triangles
    for simplex in tri.simplices:
        v0, v1, v2 = resulting_vertices[simplex]

        # Calculate midpoints for each edge of the triangle
        mid_points = [(v0 + v1) / 2, (v1 + v2) / 2, (v0 + v2) / 2]

        # Append the midpoints to the vertices
        resulting_vertices = np.append(resulting_vertices, mid_points, axis=0)

    # Convert the resulting vertices to integers and return as a list
    return resulting_vertices.astype(int).tolist()


# This function is similar to the 'subdivide' function but accepts coordinates of vertices separately.

def subdivide_with_triangles(tri, coordinates):
    """
    :param tri: Contains information about edges and vertices of the triangles.
    :param coordinates: Coordinates of vertices given separately.
    :return: A list of vertices representing the resulting triangles after subdivision.
    """
    # Convert the separate coordinates into an array
    resulting_vertices = np.asarray(coordinates.copy())

    # Iterate through each triangle in the given triangles
    for simplex in tri.simplices:
        v0, v1, v2 = resulting_vertices[simplex]

        # Calculate midpoints for each edge of the triangle and convert them to integers
        mid_points = [((v0 + v1) / 2).astype(int), ((v1 + v2) / 2).astype(int), ((v2 + v0) / 2).astype(int)]

        # Append the midpoints to the vertices
        resulting_vertices = np.append(resulting_vertices, mid_points, axis=0)

    # Convert the resulting vertices to integers and return as a list
    return resulting_vertices.astype(int).tolist()


# Densifies the landmarks by triangulating and subdividing them.

def densify_landmarks(landmarks1, landmarks2):
    """
    :param landmarks1: List of landmarks for the first image.
    :param landmarks2: List of landmarks for the second image.
    :return: Densified landmark points for both images.
    """
    # Apply Delaunay triangulation method to landmarks1
    tri = Delaunay(landmarks1)

    # Subdivide the triangles for landmarks1
    new_points = subdivdie(tri)

    # For landmarks2, triangles are the same as in landmarks1, but coordinates are different
    new_points2 = subdivide_with_triangles(tri, landmarks2)

    # Remove duplicate points between the resulting sets
    new_points1, new_points2 = removeDuplicates(new_points, new_points2)

    # Return densified landmark points for both sets
    return new_points1, new_points2


# Removes duplicate vertices after subdivision in both landmark lists

def removeDuplicates(landmark1, landmark2):
    # Copy both landmark lists
    landmark1_new = landmark1[:]
    landmark2_new = landmark2[:]
    i = 0

    # Iterate through landmarks
    while i < len(landmark1_new):
        count = 0
        j = 0

        # Search for a duplicate landmark
        while j < len(landmark1_new):

            # Check if landmark is duplicate for landmarks1
            if i != j and landmark1_new[i][0] == landmark1_new[j][0] and landmark1_new[i][1] == landmark1_new[j][1]:
                count += 1

                # Check if landmark is duplicate for landmarks2. The landmark is only
                # removed if it is duplicate in both landmarks to maintain the correlation between points
                if landmark2_new[i][0] == landmark2_new[j][0] or landmark2_new[i][1] == landmark2_new[j][1]:
                    # Remove landmark from both lists
                    landmark1_new.pop(i)
                    landmark2_new.pop(i)
                    i -= 1
                    break
            j += 1
        i += 1
    return landmark1_new, landmark2_new


if __name__ == '__main__':
    # Parser arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--img1_name', type=str, default=None,
        help='Name of the input image 1 from data directory')
    parser.add_argument(
        '--img2_name', type=str, default=None,
        help='Name of the input image 2 from data directory')
    parser.add_argument(
        '--size', type=int, default=256,
        help='Size of the output video.')
    parser.add_argument(
        '--custom', action='store_true',
        help='Set to true if the objects are non-face')

    opt = parser.parse_args()
    SIZE = opt.size
    # Retrieve image from the correct directory
    if not opt.custom:
        img1_path = os.path.join('data', 'faces', opt.img1_name)
        img2_path = os.path.join('data', 'faces', opt.img2_name)
    else:
        img1_path = os.path.join('data', 'custom', opt.img1_name)
        img2_path = os.path.join('data', 'custom', opt.img2_name)

    # Read and resize image
    image1 = cv2.imread(img1_path)
    resized_image1 = cv2.resize(image1, (SIZE, SIZE))
    image2 = cv2.imread(img2_path)
    resized_image2 = cv2.resize(image2, (SIZE, SIZE))

    # Obtain landmarks from the image through landmark detection algorithm and user input
    face1_landmarks, face2_landmarks = UI.runUI(img1_path, img2_path, custom=opt.custom, size=SIZE)
    print(len(face1_landmarks), " landmarks before densification")

    # Densify landmarks using triangle subdivision
    face1_landmarks, face2_landmarks = densify_landmarks(face1_landmarks, face2_landmarks)
    print(len(face1_landmarks), " landmarks after densification")

    # Number of intermediate frames
    num_frames = 20

    # Create intermediate landmarks per frame
    transformation_fields = create_transformation_fields(face1_landmarks, face2_landmarks, num_frames)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format

    # Output video configuration
    output_video = cv2.VideoWriter(os.path.join("output", opt.img1_name + "_" + opt.img2_name + ".avi"), fourcc, 5,
                                   (SIZE, SIZE))

    # Loop through intermediate frames
    for i in range(num_frames):
        print("iteration ", i + 1)

        # Create mesh for both images using Shepard interpolation
        warped_space1 = shepard_interpolate(resized_image1, face1_landmarks,
                                            landmark_change(face1_landmarks, transformation_fields[i]))
        warped_space2 = shepard_interpolate(resized_image2, face2_landmarks,
                                            landmark_change(face2_landmarks, transformation_fields[i]))

        # Apply mesh for both images
        warped_img1 = applyMesh(resized_image1, warped_space1)
        warped_img2 = applyMesh(resized_image2, warped_space2)
        blended_image = cv2.addWeighted(warped_img1, ((num_frames - 1) - i) / (num_frames - 1), warped_img2,
                                        i / (num_frames - 1), 0)

        # Write the image to the output video
        output_video.write(blended_image)
    output_video.release()

