import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


class iris_detection():
    def __init__(self, image, image_name):
        self.img_masked = None
        self.img_gray = None
        self.img_cropped = None
        self.img_cleaned = None
        self._img = image
        self.img_name = image_name
        self.img_type = "png"

    def detect_contours(self):

        # reduce noise (image smoothing)
        self._img = cv2.bilateralFilter(self._img, d=9, sigmaColor=150, sigmaSpace=20)
        # self._img = cv2.GaussianBlur(self._img, (5, 5), cv2.BORDER_DEFAULT)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        # convert image to gray scale
        self.convert_to_gray_scale(self._img)

        # calculate and plot histogram
        self.create_histogram(self.img_gray)

        # cut eyebrows
        self.cut_eyebrows(self._img)

        #####################################
        # Edge Features
        #####################################
        # calculate thresholded image
        _, thresholded = cv2.threshold(self.img_cleaned, 100, 255, cv2.THRESH_TOZERO_INV)
        store_path = "data/thresholded/{}_thresholded.{}".format(self.img_name, self.img_type)
        self.store_image(thresholded, store_path)

        # perform canny edge detection
        canny = cv2.Canny(self.img_cleaned, 40, 40)
        store_path = "data/canny/{}_canny.{}".format(self.img_name, self.img_type)
        self.store_image(canny, store_path)

        #####################################
        # Contours
        #####################################
        # Dilate and erode image
        closed = cv2.erode(cv2.dilate(canny, kernel, iterations=1), kernel, iterations=1)

        # find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        drawing = np.copy(self.img_cleaned)

        # draw all detected contours in blue
        cv2.drawContours(drawing, contours, -1, (255, 0, 0), 2)

        #####################################
        # Contour analysis and decision
        #####################################
        # goal: draw ellipse + center in green
        for contour in contours:

            area = cv2.contourArea(contour)
            bounding_box = cv2.boundingRect(contour)

            extend = area / (bounding_box[2] * bounding_box[3])

            # reject the contours that are too small
            if area < 5000 or area > 9500:
                continue

            # reject the contours with big extend
            if extend > 0.9:
                continue

            # compute the convex hull of the contours
            contour = cv2.convexHull(contour)

            # measure of roundness
            circumference = cv2.arcLength(contour, True)
            circularity = circumference ** 2 / (4 * math.pi * area)

            if circularity > 3:
                continue

            # calculate contour center and draw a dot there
            m = cv2.moments(contour)
            if m['m00'] != 0:
                center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
                cv2.circle(drawing, center, 3, (0, 255, 0), -1)

                # draw white circle to mask pupil
                self.mask_pupil(self.img_cleaned, center)

                # crop image
                self.crop_image(self.img_cleaned, contour)

                # store image
                store_path = "data/cropped/{}_cropped.{}".format(self.img_name, self.img_type)
                self.store_image(self.img_cropped, store_path)

            # fit an ellipse around the contour and draw it into the image
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(drawing, box=ellipse, color=(0, 255, 0))

            img = np.copy(drawing)
            self.mask_iris(img, contour)

        self._img = drawing
        store_path = "data/analyzed/{}_analyzed.{}".format(self.img_name, self.img_type)

        self.store_image(self._img, store_path)

    #############################################
    # Helper Methods
    #############################################
    # Load image as numpy array
    def load_image(self, path):
        self._img = cv2.imread(path)

        # If the image doesn't exists or is not valid then imread returns None
        if type(self._img) is type(None):
            return False
        else:
            return True

    def convert_to_gray_scale(self, image):
        self.img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def create_histogram(self, image):
        fig = plt.figure(figsize=(6, 4))
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        plt.title("Grayscale Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")
        plt.plot(hist)
        plt.xlim([0, 256])
        store_path = "data/histograms/{}_histogram.{}".format(self.img_name, self.img_type)
        fig.savefig(store_path)

    def cut_eyebrows(self, image):
        height, width = image.shape[:2]
        eyebrow_h = int(height / 5)
        self.img_cleaned = image[eyebrow_h:height, 0:width]

    def crop_image(self, image, contour):
        x, y, w, h = cv2.boundingRect(contour)

        self.img_cropped = image[y:y + h, x:x + w]

    def mask_pupil(self, image, center):
        cv2.circle(image, center, 30, (255, 255, 255), -1)

    def mask_iris(self, drawing, contour):
        ellipse = cv2.fitEllipse(contour)
        self.img_masked = cv2.ellipse(drawing, box=ellipse, color=(0, 0, 0), thickness=-1)

        store_path = "data/masked/{}_masked.{}".format(self.img_name, self.img_type)

        self.store_image(self.img_masked, store_path)

    def store_image(self, image, path):
        cv2.imwrite(path, image)
