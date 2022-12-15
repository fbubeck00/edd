import cv2
import numpy as np
import math


class iris_detection():
    def __init__(self, image_path, typ):
        self.img_test = None
        self.img1 = None
        self.img2 = None
        self.img3 = None
        self.img4 = None
        self.img5 = None
        self.img6 = None
        self.img_path = image_path
        self.typ = typ
        self.full_path = "{}.{}".format(image_path, typ)
        self._img = None
        self._pupil = None

    # Load image as numpy array
    def load_image(self):
        self._img = cv2.imread(self.full_path)

        # If the image doesn't exists or is not valid then imread returns None
        if type(self._img) is type(None):
            return False
        else:
            return True

    def convert_to_gray_scale(self):
        self._img = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)

    # perform Canny-Edge-Detection
    def detect_edges(self):

        self.img1 = cv2.Canny(self._img, 10, 150)
        self.img2 = cv2.Canny(self._img, 50, 150)
        self.img3 = cv2.Canny(self._img, 150, 150)
        self.img4 = cv2.Canny(self._img, 50, 10)
        self.img5 = cv2.Canny(self._img, 50, 50)
        self.img6 = cv2.Canny(self._img, 50, 150)

    def cut_eyebrows(self):
        height, width = self._img.shape[:2]
        print("height: ", height)
        print("width: ", width)
        eyebrow_h = int(height / 4)
        self._img = self._img[eyebrow_h:height, 0:width]

    # ToDo
    def detect_pupil(self):
        # reduce noise
        self._img = cv2.medianBlur(self._img, 5)

        circles = cv2.HoughCircles(self._img, cv2.HOUGH_GRADIENT, dp=1, minDist=self._img.shape[0] / 2,
                                   param1=100, param2=100, minRadius=0, maxRadius=0)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv2.circle(self._img, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv2.circle(self._img, center, radius, (255, 0, 255), 3)

    def detect_contours(self):
        # reduce noise
        self._img = cv2.medianBlur(self._img, 5)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        gray = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
        retval, thresholded = cv2.threshold(gray, 80, 255, 0)

        closed = cv2.erode(cv2.dilate(thresholded, kernel, iterations=1), kernel, iterations=1)
        #closed = cv2.morphologyEx(close, cv2.MORPH_CLOSE, kernel)

        contours, hierarchy = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        drawing = np.copy(self._img)
        cv2.drawContours(drawing, contours, -1, (255, 0, 0), 2)

        for contour in contours:

            area = cv2.contourArea(contour)
            bounding_box = cv2.boundingRect(contour)

            extend = area / (bounding_box[2] * bounding_box[3])

            # removing candidates that are too small
            if area < 600:
                continue

            # reject the contours with big extend
            if extend > 0.8:
                continue

            # computing the convex hull of the contours
            contour = cv2.convexHull(contour)

            # measure of roundness
            circumference = cv2.arcLength(contour, True)
            circularity = circumference ** 2 / (4 * math.pi * area)

            if circularity > 1.3:
                continue

            # calculate countour center and draw a dot there
            m = cv2.moments(contour)
            if m['m00'] != 0:
                center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
                cv2.circle(drawing, center, 3, (0, 255, 0), -1)

            # fit an ellipse around the contour and draw it into the image
            try:
                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(drawing, box=ellipse, color=(0, 255, 0))
            except:
                pass


        self._img = drawing


    def store_image(self):

        store_path = "{}_analyzed.{}".format(self.img_path, self.typ)

        cv2.imwrite(store_path, self._img)

        cv2.imwrite("data/canny-1.png", self.img1)
        cv2.imwrite("data/canny-2.png", self.img2)
        cv2.imwrite("data/canny-3.png", self.img3)
        cv2.imwrite("data/canny-4.png", self.img4)
        cv2.imwrite("data/canny-5.png", self.img5)
        cv2.imwrite("data/canny-6.png", self.img6)

    def start_detection(self):
        self.load_image()
        #self.convert_to_gray_scale()
        #self.cut_eyebrows()
        self.detect_edges()
        #self.detect_pupil()
        self.detect_contours()
        self.store_image()
