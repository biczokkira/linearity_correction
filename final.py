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
    theoretical = []
    theoretical2 = []
    paired_theoretical = []
    used_grid_points = set()

    for y in range(0, image.shape[0], grid_constant):
        for x in range(0, image.shape[1], grid_constant):
            theoretical.append((x, y))
            theoretical2.append((x, y))

    x_coords = []
    y_coords = []
    delta_x_values = []
    delta_y_values = []

    def find_nearest_grid_point(tx, ty):
        nearest_point = None
        min_distance = float('inf')
        for gx, gy in grid_points:
            if (gx, gy) not in used_grid_points:
                distance = np.sqrt((gx - tx) ** 2 + (gy - ty) ** 2)
                if distance <= 6 and distance < min_distance:
                    min_distance = distance
                    nearest_point = (gx, gy)
        print((tx, ty), nearest_point)
        return nearest_point

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if (x, y) in theoretical:
                tx, ty = (x, y)
                nearest_grid_point = find_nearest_grid_point(tx, ty)

                if nearest_grid_point:
                    gx, gy = nearest_grid_point
                    used_grid_points.add(nearest_grid_point)

                    corrected_x = tx
                    corrected_y = ty

                    delta_x = corrected_x - gx
                    delta_y = corrected_y - gy

                    x_coords.append(gx)
                    y_coords.append(gy)

                    delta_x_values.append(delta_x)
                    delta_y_values.append(delta_y)

                    paired_theoretical.append((x, y))
                else:
                    x_coords.append(x)
                    y_coords.append(y)
                    delta_x_values.append(0)
                    delta_y_values.append(0)

    grid_x, grid_y = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    grid_delta_x = griddata((x_coords, y_coords), delta_x_values, (grid_y, grid_x), method='cubic')
    grid_delta_y = griddata((x_coords, y_coords), delta_y_values, (grid_y, grid_x), method='cubic')

    return (grid_delta_x, grid_delta_y), theoretical


def visualize_displacement_vector(image, grid_points, displacement_vector):
    height, width, _ = image.shape
    displacement_x, displacement_y = displacement_vector

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(displacement_x, cmap='jet', interpolation='bilinear')
    plt.colorbar()
    plt.title('Displacement in X direction')

    plt.subplot(1, 2, 2)
    plt.imshow(displacement_y, cmap='jet', interpolation='bilinear')
    plt.colorbar()
    plt.title('Displacement in Y direction')

    plt.show()


calibration_img = cv2.imread('LINE01.PGM')

grid_points = detect_grid_points(calibration_img)

grid_constant = calculate_grid_constant(calibration_img, grid_points)
print("Grid Constant:", grid_constant)

displacement_vector, theoretical_points = calculate_grid_displacement_vector(grid_points, grid_constant, calibration_img)

visualize_displacement_vector(calibration_img, grid_points, displacement_vector)

cv2.waitKey(0)
cv2.destroyAllWindows()