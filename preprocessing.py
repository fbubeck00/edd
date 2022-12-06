import cv2
import numpy as np


class iris_detection():
    def __init__(self, image_path):
        self.tight = None
        self.mid = None
        self.wide = None
        self._img_path = image_path
        self._img = None
        self._pupil = None

    # Load image as numpy array
    def load_image(self):
        self._img = cv2.imread(self._img_path)
        print(self._img)

        # If the image doesn't exists or is not valid then imread returns None
        if type(self._img) is type(None):
            return False
        else:
            return True

    def convert_to_gray_scale(self):
        self._img = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
        print(self._img)

    # perform Canny-Edge-Detection
    def detect_edges(self):

        self.wide = cv2.Canny(self._img, 10, 200)
        self.mid = cv2.Canny(self._img, 30, 150)
        self.tight = cv2.Canny(self._img, 240, 250)

    def cut_eyebrows(self):
        height, width = self._img.shape[:2]
        eyebrow_h = int(height / 4)
        self._img = self._img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)

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


    def store_image(self):
        cv2.imwrite("data/as-test-gray.png", self._img)

        # cv2.imwrite("data/canny-wide.png", self.wide)
        # cv2.imwrite("data/canny-mid.png", self.mid)
        # cv2.imwrite("data/canny-tight.png", self.tight)

    def start_detection(self):
        self.load_image()
        self.convert_to_gray_scale()
        #self.cut_eyebrows()
        # self.detect_edges()
        self.detect_pupil()
        self.store_image()
