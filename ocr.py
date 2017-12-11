""" Simple OCR to read electric meter """

import argparse
import os.path
import math
from PIL import Image
import cv2
import numpy as np
import pytesseract

def main(image_path, canny_threshold, debug):
    """ Main function of the OCR """

    digits = image_processing(image_path, canny_threshold, debug)

    if digits:
        i = 0
        for digit in digits:
            cv2.imwrite("Digit" + str(i) + ".jpg", digit)
            text = pytesseract.image_to_string(Image.open("Digit" + str(i) + ".jpg"), \
                    config='--psm 10 -c tessedit_char_whitelist=0123456789')
            print(text)
            i = i + 1

    if debug:
        # Wait Esc key to stop
        while True:
            k = cv2.waitKey(20)
            if k == 27:
                cv2.destroyAllWindows()
                exit()

def image_processing(image_path, canny_threshold, debug):
    """ Take the image and return each digits of the meter """

    # Load image
    image = load_image(image_path)

    # Convert to gray and apply bilateral filter to smooth the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bilateral_filter = cv2.bilateralFilter(gray, 25, 10, 80)

    # Use Canny algorithm to detect edges
    if not canny_threshold:
        (canny_min, canny_max) = (60, 120)
    else:
        (canny_min, canny_max) = (canny_threshold[0], canny_threshold[1])

    edged = cv2.Canny(bilateral_filter, canny_min, canny_max)

    # Use Hough transfom to detect lines
    lines = cv2.HoughLines(edged, 1, np.pi/180, 120)

    # Filter lines that don't have a theta between 60° and 120°
    # So we keep only the horizontal lines or almost (+/- 30°)
    # And calculate the theta average of these filtered lines
    filtered_lines = []
    theta_min = 60 * np.pi/180
    theta_max = 120 * np.pi/180
    theta_average = 0
    theta_degree = 0

    if lines is not None:
        for line in lines:
            theta = line[0][1]
            if theta > theta_min and theta < theta_max:
                filtered_lines.append(line)
                theta_average += theta

    if filtered_lines:
        theta_average /= len(filtered_lines)
        theta_degree = (theta_average / (np.pi/180)) - 90

    # Rotate the image according to theta average to align the digits as mush as
    # possible horizontally in order to optimize the recognition of them
    # So we need to make a affine transfomation to do that
    rotation = edged
    image_rotation = image

    if theta_degree != 0:
        # Get the size of the image to determine its center
        height, width = edged.shape
        (x_center, y_center) = (width / 2, height / 2)

        # Calculate the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((x_center, y_center), theta_degree, 1)

        # Make the affine transformation
        rotation = cv2.warpAffine(edged, rotation_matrix, (width, height))
        image_rotation = cv2.warpAffine(image, rotation_matrix, (width, height))

    contours = cv2.findContours(rotation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    # Keep only contours that have the size of a digit
    filtered_bounding_rects = []
    if contours:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (w > 20 and w < 50) and (h > 45 and h < 80):
                filtered_bounding_rects.append([x, y, w, h])

    # Keep only bounding rects that are digits
    bounding_digits = []
    if len(filtered_bounding_rects) > 1:
        for i in range(0, len(filtered_bounding_rects)):
            temp = [filtered_bounding_rects[i]]

            i_x, i_y, i_w, i_h = filtered_bounding_rects[i]
            for j in range(i + 1, len(filtered_bounding_rects)):
                j_x, j_y, j_w, j_h = filtered_bounding_rects[j]
                if abs(i_y - j_y) < 20 and abs(i_h - j_h) < 10:
                    temp.append(filtered_bounding_rects[j])

            if len(temp) > len(bounding_digits):
                bounding_digits = temp

    # Sort bounding digits by x position
    bounding_digits.sort(key=lambda x: x[0])

    # Extract all digits from the image
    digits = []
    space = 7
    for bounding_digit in bounding_digits:
        x, y, w, h = bounding_digit
        roi = bilateral_filter[y - space : y + h + space, x - space : x + w + space]
        roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        digits.append(roi)

    # Show images if debug
    if debug:
        # Print the lines of the hough transformation in the source image
        if lines is not None:
            for line in lines:
                rho = line[0][0]
                theta = line[0][1]
                cos = math.cos(theta)
                sin = math.sin(theta)
                x_0 = cos * rho
                y_0 = sin * rho
                point_1 = (int(x_0 + 1000*(-sin)), int(y_0 + 1000*(cos)))
                point_2 = (int(x_0 - 1000*(-sin)), int(y_0 - 1000*(cos)))
                cv2.line(image, point_1, point_2, (0, 255, 0), 1)

        # Print the filtered lines of the hough transformation also
        if filtered_lines is not None:
            for filtered_line in filtered_lines:
                rho = filtered_line[0][0]
                theta = filtered_line[0][1]
                cos = math.cos(theta)
                sin = math.sin(theta)
                x_0 = cos * rho
                y_0 = sin * rho
                point_1 = (int(x_0 + 1000*(-sin)), int(y_0 + 1000*(cos)))
                point_2 = (int(x_0 - 1000*(-sin)), int(y_0 - 1000*(cos)))
                cv2.line(image, point_1, point_2, (0, 0, 255), 1)

        # Print the bounding rects digits
        if bounding_digits:
            for digit in bounding_digits:
                x, y, w, h = digit
                cv2.rectangle(image_rotation, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Original", image)
        cv2.imshow("Gray", gray)
        cv2.imshow("BilateralFilter", bilateral_filter)
        cv2.imshow("Canny", edged)
        cv2.imshow("Rotation", rotation)
        cv2.imshow("Original Rotation", image_rotation)

        if digits:
            i = 1
            for digit in digits:
                cv2.imshow("Digit " + str(i), digit)
                i = i+1

    return digits

def load_image(image_path):
    """ Read the image and return a 3-dimensional matrix """

    if not os.path.isfile(image_path):
        print("We have passed an invalid image")
        exit()

    return cv2.imread(image_path)

def parse_arguments():
    """ Setup up a parser to read command line arguments"""

    parser = argparse.ArgumentParser(
        description='Simple OCR to read electric meter')
    parser.add_argument('-i', '--image',
                        action='store',
                        dest='image',
                        required=True,
                        help='path of the image to read')
    parser.add_argument('--threshold',
                        action='store',
                        dest='canny_threshold',
                        nargs=2,
                        type=int,
                        help='set the min an max thresholds for canny function')
    parser.add_argument('-d', '--debug',
                        action='store_true',
                        dest='debug',
                        help='debug flag to display all transition images')
    parser.add_argument('-v', '--version',
                        action='version',
                        version='%(prog)s 0.0.1')
    return parser.parse_args()

if __name__ == '__main__':
    arguments = parse_arguments()
    main(arguments.image, arguments.canny_threshold, arguments.debug)
