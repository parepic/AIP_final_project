import copy
import glob
import os
import random
import subprocess
import tkinter as tk
import cv2
import dlib
import numpy as np
from PIL import Image, ImageTk

CANVAS_SIZE = 512
SIZE = 256


class PointMover:

    # Initialize object
    def __init__(self, root, image_path1, image_path2, landmarks1, landmarks2):
        """
        :param root: Tkinter root object
        :param image_path1: path for image 1
        :param image_path2: path for image 2
        :param landmarks1: landmark list for image 1
        :param landmarks2: landmark list for image 2
        """
        self.root = root
        self.image1Points = []
        self.image2Points = []
        self.finished = False
        self.root.title("Point Mover")

        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE)
        self.canvas.pack()

        self.points = []
        self.largestPoint = -1
        self.canvas.bind("<Button-1>", self.select_point)
        self.canvas.bind("<B1-Motion>", self.move_point)

        self.selected_point = None
        self.drag_data = {"x": 0, "y": 0}

        # Store original point appearance attributes
        self.point_appearance = {}

        # Load the image and display it on the canvas
        image = cv2.imread(image_path1)
        self.load_image(image)
        self.plot_landmarks(landmarks1)

        # Create a "Create Point" button
        self.create_point_button = tk.Button(root, text="Create Point", command=self.create_new_point)
        self.create_point_button.pack()

        self.root.protocol("WM_DELETE_WINDOW", lambda: self.on_closing(image_path2, landmarks2))

        # Create a "Delete Point" button
        self.delete_point_button = tk.Button(root, text="Delete Point", command=self.delete_selected_point)
        self.delete_point_button.pack()

    # This function is called when the canvas is closed

    def on_closing(self, image_path=None, landmarks=None):
        """
        :param image_path: image path. If the canvas is closed the second time, it is equal to None
        :param landmarks: landmarks list. If the canvas is closed the second time, it is equal to None
        """
        self.root.destroy()
        if not self.image1Points:
            self.root = tk.Tk()
            self.root.title("Point Mover 2")
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.canvas = tk.Canvas(self.root, width=CANVAS_SIZE, height=CANVAS_SIZE)
            self.canvas.pack()
            self.canvas.bind("<Button-1>", self.select_point)
            self.canvas.bind("<B1-Motion>", self.move_point)
            image = cv2.imread(image_path)
            self.load_image(image)
            self.selected_point = None
            self.image1Points = copy.deepcopy(self.points)
            self.points = self.remove_landmark_points(self.points)
            self.plot_landmarks(landmarks, copy.deepcopy(self.image1Points))
            self.drag_data = {"x": 0, "y": 0}
        else:
            self.image2Points = self.points
            self.finished = True

    # Loads image to the canvas

    def load_image(self, image):
        """
        :param image: Image to be loaded
        :return:
        """
        # Convert the OpenCV image to a PIL Image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        # Resize the image to fit the canvas size
        image = image.resize((CANVAS_SIZE, CANVAS_SIZE))

        # Display the resized image on the canvas
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def random_color(self):
        colors = ["cyan", "blue", "orange", "purple", "yellow", "green", "pink",
                  "magenta"]  # Define a list of possible colors
        color = random.choice(colors)  # Choose a random color from the list
        return color
    # Plots the landmarks obtained from landmark detection algorithm.
    # Additionally, plots points added points from image 1 if old_points is not None
    def plot_landmarks(self, landmarks, old_points=None):
        """
        :param landmarks: Landmarks from landmark detection algorithm
        :param old_points: Manually Added points from image 1
        """
        landmarkPoints = []
        for i in range(2, len(landmarks) + 2):
            x, y = landmarks[i - 2]
            if (old_points and self.contains_id(old_points, str(i) + "a")) or not old_points:
                landmarkPoints.append({"x": x, "y": y, "id": str(i) + "a"})
                if not old_points:
                    color = self.random_color()
                else:
                    color = self.point_appearance[str(i) + "a"]["color"]
                self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill=color, outline="black", width=2,
                                        tags=str(i) + "a")
                if i > self.largestPoint:
                    self.largestPoint = i
                # Store original appearance attributes for the new point
                self.point_appearance[str(i) + "a"] = {
                    "outline": "black",
                    "width": 2,
                    "color": color
                }
        if old_points:
            added_points = self.remove_landmark_points(old_points)
            for point in added_points:
                x = point["x"]
                y = point["y"]
                self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="red", outline="black", width=2,
                                        tags=point["id"])
            landmarkPoints.extend(added_points)

        self.points = landmarkPoints

    # Removes points that are not manually added (removes all green points)

    def remove_landmark_points(self, points):
        """
        :param points: landmark points
        """
        added_points = []
        for point in reversed(points):
            color = self.point_appearance[point["id"]]["color"]
            if color != "red":
                break
            added_points.insert(0, point)
        return added_points

    # Checks if point with the given id exists in a given list of points

    def contains_id(self, points, point_id):
        """
        :param points: input points
        :param point_id: point id
        """
        for point in points:
            if point_id == point["id"]:
                return True
        return False

    # Creates a new point in the canvas. Called when create button is clicked

    def create_point(self, x, y):
        """
        :param x: x coordinate
        :param y: y coordinate
=        """
        self.largestPoint += 1
        new_id = str(self.largestPoint)
        self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="red", outline="black", width=2, tags=new_id + "a")
        self.points.append({"x": x, "y": y, "id": new_id + "a"})
        # Store original appearance attributes for the new point
        self.point_appearance[new_id + "a"] = {
            "outline": "black",
            "width": 2,
            "color": "red"
        }

    # Called when point is selected

    def select_point(self, event):
        """
        :param event: Select event
        """
        all_tags = set()
        for item_id in self.canvas.find_all():
            tags = self.canvas.gettags(item_id)
            all_tags.update(tags)

        x, y = event.x, event.y
        for point in self.points:
            point_id = str(point["id"])
            if self.get_id_by_tag(point_id) and \
                    self.canvas.coords(self.get_id_by_tag(point_id))[0] <= x <= \
                    self.canvas.coords(self.get_id_by_tag(point_id))[2] \
                    and self.canvas.coords(self.get_id_by_tag(point_id))[
                1] <= y <= self.canvas.coords(self.get_id_by_tag(point_id))[3]:
                if self.selected_point is not None:
                    self.canvas.itemconfig(self.get_id_by_tag(self.selected_point),
                                           outline=self.point_appearance[self.selected_point]["outline"],
                                           width=self.point_appearance[self.selected_point]["width"])
                # Change the selected point's appearance
                self.selected_point = point_id
                self.canvas.itemconfig(point_id, outline="blue", width=3)
                self.drag_data["x"] = x
                self.drag_data["y"] = y
                break

    # Called when the point is dragged

    def move_point(self, event):
        """
        :param event: Move event
        """
        if self.selected_point is not None:

            x, y = event.x, event.y
            delta_x = x - self.drag_data["x"]
            delta_y = y - self.drag_data["y"]
            self.canvas.move(self.get_id_by_tag(self.selected_point), delta_x, delta_y)
            self.drag_data["x"] = x
            self.drag_data["y"] = y
            for point in self.points:
                if point.get("id") == self.selected_point:
                    point["x"] = x
                    point["y"] = y

    # Creates new point in the center. Called when create new point button is clicked

    def create_new_point(self):
        x, y = 250, 250
        self.create_point(x, y)

    # Removes the points from canvas, calls when delete button is pressed

    def delete_selected_point(self):
        if self.selected_point is not None:
            # Delete the point from the canvas
            self.canvas.delete(self.get_id_by_tag(self.selected_point))
            self.points = [point for point in self.points if point["id"] != self.selected_point]
            # Remove the stored appearance attributes for the deleted point
            del self.point_appearance[self.selected_point]
            self.selected_point = None

    # Transforms points into the suitable format

    def transform_list(self, points):
        """
        :param points: List of points
        :return: List of points in a new format
        """
        transformedPoints = []
        for point in points:
            x = point["x"]
            y = point["y"]
            transformedPoints.append([x, y])
        return transformedPoints

    # Gets the points id by tag

    def get_id_by_tag(self, tag):
        # Get a list of canvas item IDs with the specified tag
        item_ids = self.canvas.find_withtag(tag)
        if not item_ids:
            return None
        idd, *rest = item_ids

        return idd


# This function detects landmarks on a face within an image using dlib library.

def detect_landmarks(img_path):
    """
    :param show_img: Boolean flag indicating whether to display the image with detected landmarks.
    :return: A list of detected landmark coordinates.
    """
    # Load the shape predictor for 68 face landmarks
    landmark_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat")

    # Read the image and prepare for face detection
    img = cv2.imread(img_path)
    face_detector = dlib.get_frontal_face_detector()
    resized_image = cv2.resize(img, (CANVAS_SIZE, CANVAS_SIZE))
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    faces = face_detector(gray)
    landmarks_list = []
    # Loop through each detected face to find landmarks
    for face in faces:
        # Find 68 facial landmarks for each detected face
        landmarks = landmark_detector(gray, face)
        # Retrieve x and y coordinates for each landmark
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_list.append([x, y])
    # Return the list of detected landmark coordinates
    return landmarks_list

def draw_landmarks(img, landmarks, name):
    for i in range(len(landmarks)):
        cv2.circle(img, (landmarks[i][0], landmarks[i][1]), 1, (255, 150, 0), 2)
    cv2.imwrite(name, img)
    cv2.waitKey(0)
    return img



# Detects custom (non-face) landmarks on any type of objects
# It uses SuperGlue pretrained network to detect features

def detectCustomFeatures(img1_path, img2_path):
    """
    :param img1_path: image 1 path
    :param img2_path: image 2 path
    :return: output landmarks after running feature extractor
    """

    # Write paths to txt file to be used by SuperGlue feature extractor

    with open(os.path.join("data", "file.txt"), 'w') as file:
        file.write(f"{img1_path} {img2_path}\n")
    command = [
        "python",
        "SuperGluePretrainedNetwork/match_pairs.py",
        "--input_pairs",
        os.path.join("data", "file.txt"),
        "--input_dir",
        "./",
        "--output_dir",
        os.path.join("data", "custom"),
        "--match_threshold",
        "0.2",
    ]

    # Run the feature extractor in a seperate subprocess
    subprocess.run(command, check=True)
    npz_files = glob.glob("data/custom" + '/*.npz')
    result_list_path = npz_files[0]
    npz = np.load(result_list_path)
    landmarks1 = []
    landmarks2 = []
    # Convert matching features to suitable feature
    for i in range(len(npz['keypoints0'])):
        if npz['matches'][i] != -1:
            landmarks1.append(npz['keypoints0'][i].tolist())
            landmarks2.append(npz['keypoints1'][npz['matches'][i]].tolist())
    # Rescale landmarks to 256x256 range
    landmarks1 = rescaleCoords(CANVAS_SIZE / 480, CANVAS_SIZE / 640, landmarks1)
    landmarks2 = rescaleCoords(CANVAS_SIZE / 480, CANVAS_SIZE / 640, landmarks2)
    npz.close()
    os.remove(result_list_path)
    # return landmarks
    return landmarks1, landmarks2

# Multiplies x and y coordinates by a given factor

def rescaleCoords(x_factor, y_factor, landmarks):
    """
    :param x_factor: factor to multiply x coordinates with
    :param y_factor: factor to multiply y coordinates with
    :param landmarks: list of landmarks
    :return: rescaled coordinates
    """

    def multiply_and_convert(item):
        return int(item[0] * y_factor), int(item[1] * x_factor)

    result_generator = map(multiply_and_convert, landmarks)
    result = list(result_generator)
    return result


# detects landmarks and runs UI
def runUI(image_path1, image_path2, custom=False, size=256):
    """
    :param size:
    :param image_path1: path of image 1
    :param image_path2: path of image 2
    :param custom: true if the object to be morphed is object other than the face
    :return: landmark coordinates after user editing
    """
    SIZE = size
    root = tk.Tk()
    img = cv2.imread(image_path1)
    resized_image = cv2.resize(img, (SIZE, SIZE))
    img2 = cv2.imread(image_path2)
    resized_image2 = cv2.resize(img2, (SIZE, SIZE))
    # Run different feature extractor based on the value of custom
    if custom:
        landmarks1, landmarks2 = detectCustomFeatures(image_path1, image_path2)
        # draw_landmarks(resized_image, rescaleCoords(0.5, 0.5, landmarks1), "detector1.png")
        # draw_landmarks(resized_image2, rescaleCoords(0.5, 0.5, landmarks2), "detector2.png")

    else:
        landmarks1 = detect_landmarks(image_path1)
        # draw_landmarks(resized_image, rescaleCoords(0.5, 0.5, landmarks1), "detector1.png")

        landmarks2 = detect_landmarks(image_path2)
        # draw_landmarks(resized_image2, rescaleCoords(0.5, 0.5, landmarks2), "detector2.png")

    # Initialize and run UI
    app = PointMover(root, image_path1, image_path2, landmarks1, landmarks2)
    root.mainloop()

    # Wait until canvases are closed
    while not app.finished:
        pass

    # Transform landmarks into suitable format
    final_landmarks1 = app.transform_list(app.image1Points)
    final_landmarks2 = app.transform_list(app.image2Points)

    # Rescale and return the final landmark positions for both images
    return rescaleCoords(SIZE / CANVAS_SIZE, SIZE / CANVAS_SIZE, final_landmarks1), rescaleCoords(SIZE / CANVAS_SIZE,
                                                                                                SIZE / CANVAS_SIZE,
                                                                                                final_landmarks2)




