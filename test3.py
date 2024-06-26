import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def detect_grid_points(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 0, 0.05, 4)
    corners = np.int0(corners)

    grid_points = []
    img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for corner in corners:
        x, y = corner.ravel()
        grid_points.append((x, y))
        cv2.circle(img_color, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow('Detect grid', img_color)
    return grid_points

def calculate_grid_constant_row(img):
    mean_spacing = 0
    count_rows = 0

    height = img.shape[0]
    row = 0
    while row < height:
        if np.sum(img[row, :] == 1) == 0:
            row += 1
            continue

        current_mean = 0
        space_count = 0
        db = 0

        for col in range(img.shape[1]):
            if np.sum(img[row:row+3, col] == 1) != 0 and space_count == 0:
                space_count = col
                continue

            if np.sum(img[row:row+3, col] == 1) != 0 and space_count != 0:
                current_mean += (col - space_count)
                space_count = 0
                db += 1

        mean_spacing += current_mean / db
        count_rows += 1

        row += 3

    return mean_spacing / count_rows

def calculate_grid_constant(img, grid_points):
    m1 = calculate_grid_constant_row(img)
    m2 = calculate_grid_constant_row(img.T)

    return round((m1 + m2) / 2)


def bilinear_interpolation(x, y, image, grid_constant):
    x0 = int(x)
    x1 = min(x0 + 1, image.shape[1] - 1)
    y0 = int(y)
    y1 = min(y0 + 1, image.shape[0] - 1)

    Ia = image[y0, x0]
    Ib = image[y1, x0]
    Ic = image[y0, x1]
    Id = image[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return (wa * Ia + wb * Ib + wc * Ic + wd * Id) / (grid_constant**2)


def calculate_grid_displacement_vector(grid_points, grid_constant, image):
    correction_values = []
    theoretical = []
    theoretical2 = []
    paired_theoretical = []
    for y in range(0, image.shape[0], grid_constant):
        for x in range(0, image.shape[1], grid_constant):
            theoretical.append((x, y))
            theoretical2.append((x, y))

    x_coords = []
    y_coords = []
    delta_x_values = []
    delta_y_values = []

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if (x, y) in theoretical:
                corrected_x = round(x / grid_constant) * grid_constant
                corrected_y = round(y / grid_constant) * grid_constant
                min_distance = float('inf')
                nearest_theoretical = None
                for gridp in grid_points:
                    if gridp not in paired_theoretical:
                        distance = np.sqrt(
                            (gridp[0] - corrected_x) ** 2 + (gridp[1] - corrected_y) ** 2)
                        if distance < min_distance:
                            min_distance = distance
                            nearest_theoretical = gridp

                paired_theoretical.append(nearest_theoretical)  # Add nearest theoretical point to paired list
                corrected_x, corrected_y = nearest_theoretical  # Update corrected coordinates

                delta_x = corrected_x - x
                delta_y = corrected_y - y
                correction_values.append((delta_x, delta_y))
                x_coords.append(x)
                y_coords.append(y)
                delta_x_values.append(delta_x)
                delta_y_values.append(delta_y)
            else:
                correction_values.append((0, 0))

    grid_x, grid_y = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    grid_delta_x = griddata((x_coords, y_coords), delta_x_values, (grid_y, grid_x), method='linear')
    grid_delta_y = griddata((x_coords, y_coords), delta_y_values, (grid_y, grid_x), method='linear')

    return (grid_delta_x, grid_delta_y), theoretical, correction_values


def visualize_displacement_vector(image, grid_points, displacement_vector):
    height, width, _ = image.shape
    displacement_x, displacement_y = displacement_vector

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(displacement_x, cmap='jet', interpolation='none')
    plt.colorbar()
    plt.title('Displacement in X direction')

    plt.subplot(1, 2, 2)
    plt.imshow(displacement_y, cmap='jet', interpolation='none')
    plt.colorbar()
    plt.title('Displacement in Y direction')

    plt.show()


def visualize_displacement_vector_test(image, grid_points, displacement_vector, theoretical_points):
    image_vis = image.copy()

    for theo in theoretical_points:
        cv2.circle(image_vis, (theo[0], theo[1]), 1, (255, 0, 0), 1)

    for point in grid_points:
        cv2.circle(image_vis, (point[0], point[1]), 1, (0, 255, 255), 1)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if (x, y) in grid_points:
                point = (x, y)
                corrected_x = int(point[0] + displacement_vector[y * image.shape[1] + x][0])
                corrected_y = int(point[1] + displacement_vector[y * image.shape[1] + x][1])
                cv2.arrowedLine(image_vis, (int(point[0]), int(point[1])), (corrected_x, point[1]), (0, 255, 0),
                                1)  # Green for x component
                cv2.arrowedLine(image_vis, (int(point[0]), int(point[1])), (point[0], corrected_y), (0, 0, 255),
                                1)  # Red for y component

    cv2.imshow("Displacement Vectors", image_vis)


calibration_img = cv2.imread('LINE01.PGM')

grid_points = detect_grid_points(calibration_img)

grid_constant = calculate_grid_constant(calibration_img, grid_points)
print("Grid Constant:", grid_constant)

displacement_vector, theoretical_points, correction_values = calculate_grid_displacement_vector(grid_points, grid_constant, calibration_img)

visualize_displacement_vector_test(calibration_img, grid_points, correction_values, theoretical_points)
visualize_displacement_vector(calibration_img, grid_points, displacement_vector)

cv2.waitKey(0)
cv2.destroyAllWindows()