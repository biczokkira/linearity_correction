import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_grid_points(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 0, 0.01, 4)
    corners = np.int0(corners)

    grid_points = []
    img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for corner in corners:
        x, y = corner.ravel()
        grid_points.append((x, y))
        cv2.circle(img_color, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow('Detect grid', img_color)
    return grid_points


def calculate_grid_constant(grid_points):
    distances = []
    for i in range(len(grid_points)):
        for j in range(i + 1, len(grid_points)):
            distances.append(np.sqrt((grid_points[i][0] - grid_points[j][0]) ** 2 +
                                     (grid_points[i][1] - grid_points[j][1]) ** 2))

    if distances:
        grid_constant = np.median(distances)
    else:
        raise ValueError("No grid points detected.")

    return grid_constant


def calculate_correction_displacement_vector(grid_points, grid_constant, image):
    theoretical = []
    theoretical2 = []
    paired_theoretical = []
    grid_constant = 8
    for y in range(0, image.shape[0], grid_constant):
        for x in range(0, image.shape[1], grid_constant):
            theoretical.append((x, y))
            theoretical2.append((x, y))

    correction_values = []
    grid_constant = 8
    for point in grid_points:
        x, y = point
        corrected_x = round(x / grid_constant) * grid_constant
        corrected_y = round(y / grid_constant) * grid_constant
        delta_x = corrected_x - x
        delta_y = corrected_y - y
        correction_values.append((delta_x, delta_y))

    return correction_values, theoretical


def visualize_displacement_vector(image, grid_points, displacement_vector):
    displacement_x = np.zeros((256, 256))
    displacement_y = np.zeros((256, 256))

    for point, displacement in zip(grid_points, displacement_vector):
        x, y = point
        delta_x, delta_y = displacement
        displacement_x[y, x] = delta_x
        displacement_y[y, x] = delta_y

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(displacement_x, cmap='jet', interpolation='nearest')
    plt.colorbar()
    plt.title('Displacement in X direction')

    plt.subplot(1, 2, 2)
    plt.imshow(displacement_y, cmap='jet', interpolation='nearest')
    plt.colorbar()
    plt.title('Displacement in Y direction')

    plt.show()

def visualize_displacement_vector_test(image, grid_points, displacement_vector, theoretical_points):
    image_vis = image.copy()

    for theo in theoretical_points:
        cv2.circle(image_vis, (theo[0], theo[1]), 1, (255, 0, 0), 1)

    for point in grid_points:
        cv2.circle(image_vis, (point[0], point[1]), 1, (0, 255, 255), -1)

    for point, displacement in zip(grid_points, displacement_vector):
        corrected_x = int(point[0] + displacement[0])
        corrected_y = int(point[1] + displacement[1])
        cv2.arrowedLine(image_vis, (int(point[0]), int(point[1])), (corrected_x, point[1]), (0, 255, 0), 1)
        cv2.arrowedLine(image_vis, (int(point[0]), int(point[1])), (point[0], corrected_y), (0, 0, 255),
                        1)

    cv2.imshow("Displacement Vectors", image_vis)

calibration_img = cv2.imread('LINE01.PGM')

grid_points = detect_grid_points(calibration_img)

grid_constant = calculate_grid_constant(grid_points)
print("Grid Constant:", grid_constant)

displacement_vector, theoretical = calculate_correction_displacement_vector(grid_points, grid_constant, calibration_img)

visualize_displacement_vector_test(calibration_img, grid_points, displacement_vector, theoretical)
visualize_displacement_vector(calibration_img, grid_points, displacement_vector)

cv2.waitKey(0)
cv2.destroyAllWindows()
