import cv2
import numpy as np
import math


class iris_detection():
    def __init__(self, image_path, image_type):
        self.img_masked = None
        self.img_gray = None
        self.img_cropped = None
        self.img_path = image_path
        self.image_type = image_type
        self.full_path = "{}.{}".format(image_path, image_type)
        self._img = None

    def detect_contours(self):
        # load image
        self.load_image(self.full_path)

        # reduce noise (image smoothing)
        #self._img = cv2.bilateralFilter(self._img, d=9, sigmaColor=150, sigmaSpace=20)
        self._img = cv2.GaussianBlur(self._img, (5, 5), cv2.BORDER_DEFAULT)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        self.convert_to_gray_scale(self._img)

        _, thresholded = cv2.threshold(self.img_gray, 100, 255, cv2.THRESH_TOZERO_INV)

        closed = cv2.erode(cv2.dilate(thresholded, kernel, iterations=1), kernel, iterations=1)

        contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        drawing = np.copy(self._img)

        # draw all detected contours in blue
        cv2.drawContours(drawing, contours, -1, (255, 0, 0), 2)

        # analyze contours and draw ellipse + center in green
        for contour in contours:

            area = cv2.contourArea(contour)
            bounding_box = cv2.boundingRect(contour)

            extend = area / (bounding_box[2] * bounding_box[3])

            # reject the contours that are too small
            if area < 15000:
                continue

            # reject the contours with big extend
            if extend > 0.8:
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

                # crop image
                self.crop_image(self._img, contour)

                # store image
                store_path = "{}_cropped.{}".format(self.img_path, self.image_type)
                self.store_image(self.img_cropped, store_path)

            # fit an ellipse around the contour and draw it into the image
            try:
                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(drawing, box=ellipse, color=(0, 255, 0))

                img = np.copy(drawing)
                self.mask_pupil(img, contour)
            except:
                pass

        self._img = drawing
        store_path = "{}_analyzed.{}".format(self.img_path, self.image_type)

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

    def crop_image(self, image, contour):
        x, y, w, h = cv2.boundingRect(contour)

        self.img_cropped = image[y:y + h, x:x + w]

    def mask_pupil(self, drawing, contour):
        ellipse = cv2.fitEllipse(contour)
        self.img_masked = cv2.ellipse(drawing, box=ellipse, color=(0, 0, 0), thickness=-1)

        store_path = "{}_masked.{}".format(self.img_path, self.image_type)

        self.store_image(self.img_masked, store_path)

    def store_image(self, image, path):
        cv2.imwrite(path, image)
