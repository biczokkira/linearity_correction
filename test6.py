import cv2
import numpy as np
from scipy.spatial import distance


def detect_grid_points(image):
    print(image.shape)
    corners = cv2.goodFeaturesToTrack(image, 0, 0.01, 4)
    corners = np.int0(corners)
    grid_points = []
    img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for corner in corners:
        x, y = corner.ravel()
        grid_points.append([x, y])
        cv2.circle(img_color, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow('Detect grid', img_color)
    print(np.array(grid_points))
    return np.array(grid_points)


def generate_theoretical_points(image_size):
    theoretical_points = []
    for y in range(0, image_size[0], 8):
        for x in range(0, image_size[1], 8):
            theoretical_points.append([x, y])
    return np.array(theoretical_points)


def calculate_grid_spacing(grid_points):
    grid_spacing_x = np.mean(np.diff(grid_points[:, 0]))
    grid_spacing_y = np.mean(np.diff(grid_points[:, 1]))
    return grid_spacing_x, grid_spacing_y


def calculate_correction_vector(theoretical_points, detected_points):
    print(theoretical_points.shape, detected_points.shape)
    correction_vector = detected_points - theoretical_points
    return correction_vector


def expand_detected_points(detected_points, theoretical_points):
    num_detected = len(detected_points)
    num_theoretical = len(theoretical_points)
    if num_detected < num_theoretical:
        expanded_points = detected_points
        theoretical_points = theoretical_points[:num_detected]
    else:
        expanded_points = detected_points[:num_theoretical]
    return expanded_points, theoretical_points


def find_nearest_theoretical_point(detected_point, theoretical_points):
    min_distance = float('inf')
    nearest_point = None
    for theoretical_point in theoretical_points:
        dist = distance.euclidean(detected_point, theoretical_point)
        if dist < min_distance:
            min_distance = dist
            nearest_point = theoretical_point
    return nearest_point


def visualize_correction_vectors(detected_points, nearest_points):
    img_color = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    img_color.fill(255)
    for i in range(len(detected_points)):
        x, y = detected_points[i]
        nearest_x, nearest_y = nearest_points[i]
        cv2.arrowedLine(img_color, (x, y), (int(nearest_x), int(nearest_y)), (0, 0, 255), 1)

    cv2.imshow('Correction Vectors', img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('LINE01.PGM', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Kalibrációs kép', img)

theoretical_points = generate_theoretical_points(img.shape)
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for point in theoretical_points:
    x, y = point
    cv2.circle(img_color, (x, y), 1, (0, 0, 255), -1)

# Eredmények megjelenítése
cv2.imshow('Elméletileg meghatározott pontok', img_color)

detected_points = detect_grid_points(img)

grid_spacing_x, grid_spacing_y = calculate_grid_spacing(detected_points)
print("Rácsállandó (x, y):", grid_spacing_x, grid_spacing_y)

#detected_points, theoretical_points = expand_detected_points(detected_points, theoretical_points)

nearest_points = []
for detected_point in detected_points:
    nearest_point = find_nearest_theoretical_point(detected_point, theoretical_points)
    nearest_points.append(nearest_point)

#correction_vector = calculate_correction_vector(theoretical_points, detected_points)

visualize_correction_vectors(detected_points, nearest_points)
cv2.waitKey(0)
cv2.destroyAllWindows()
