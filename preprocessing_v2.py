import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from data_engineering import dataset_creation


class iris_detection_v2():
    def __init__(self, image, image_name, dataset):
        self.img_mask = None
        self.img_gray = None
        self.img_crop = None
        self.img_cut = None
        self.dataset = dataset
        self._img = image
        self.img_name = image_name
        self.img_type = "png"

    def preprocess_image(self):

        # reduce noise (image smoothing)
        self.img_smooth = cv2.bilateralFilter(self._img, d=9, sigmaColor=150, sigmaSpace=20)
        # self._img = cv2.GaussianBlur(self._img, (5, 5), cv2.BORDER_DEFAULT)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        # calculate and plot histogram
        self.create_histogram(self.img_smooth)

        self.convert_to_gray_scale(self.img_smooth)

        #####################################
        # Edge Features
        #####################################
        # calculate thresholded image
        _, thresholded = cv2.threshold(self.img_gray, 100, 255, cv2.THRESH_TOZERO_INV)
        store_path = "data/thresholded/{}_thresholded.{}".format(self.img_name, self.img_type)
        self.store_image(thresholded, store_path)

        # perform canny edge detection
        canny = cv2.Canny(self.img_gray, 15, 15)
        store_path = "data/canny/{}_canny.{}".format(self.img_name, self.img_type)
        self.store_image(canny, store_path)

        #####################################
        # Contours
        #####################################
        # Dilate and erode image
        closed = cv2.erode(cv2.dilate(canny, kernel, iterations=1), kernel, iterations=1)

        # find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        drawing = np.copy(self.img_smooth)

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
            if area < 1000 or area > 5000:
                continue

            # reject the contours with big extend
            if extend > 0.9:
                continue

            # compute the convex hull of the contours
            contour = cv2.convexHull(contour)

            # measure of roundness
            circumference = cv2.arcLength(contour, True)
            circularity = circumference ** 2 / (4 * math.pi * area)

            if circularity > 1.2:
                continue

            # calculate contour center and draw a dot there
            m = cv2.moments(contour)
            if m['m00'] != 0:
                center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
                cv2.circle(drawing, center, 3, (0, 255, 0), -1)

                # draw black circle to mask pupil
                self.mask_pupil(self._img, center)

                # crop image
                self.img_crop = self.crop_image(self._img, center)

                # store image as .png
                store_path = "data/cropped/{}_cropped.{}".format(self.img_name, self.img_type)
                self.store_image(self.img_crop, store_path)

                # store image in dataset
                dataset_creation.add_to_dataset(self.img_crop, self.dataset)

            # fit an ellipse around the contour and draw it into the image
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(drawing, box=ellipse, color=(0, 255, 0))

            img = np.copy(drawing)
            self.mask_iris(img, contour)

        self._img = drawing
        store_path = "data/analyzed/{}_analyzed.{}".format(self.img_name, self.img_type)

        self.store_image(self._img, store_path)

        return self.dataset

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

    def crop_image(self, image, center):
        x = center[0] - 60
        y = center[1] - 60
        h = 115
        w = 115
        return image[y:y + h, x:x + w]

    def mask_pupil(self, image, center):
        cv2.circle(image, center, 30, (0, 0, 0), -1)

    def mask_iris(self, drawing, contour):
        ellipse = cv2.fitEllipse(contour)
        self.img_mask = cv2.ellipse(drawing, box=ellipse, color=(0, 0, 0), thickness=-1)

        store_path = "data/masked/{}_masked.{}".format(self.img_name, self.img_type)

        self.store_image(self.img_mask, store_path)

    def store_image(self, image, path):
        cv2.imwrite(path, image)
