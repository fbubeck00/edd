import cv2
import numpy


class iris_detection():
    def __init__(self, image_path):
        self.tight = None
        self.mid = None
        self.wide = None
        self._img_path = image_path
        self._img = None

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

    def store_image(self):
        cv2.imwrite("data/as-test-gray.png", self._img)

        cv2.imwrite("data/canny-wide.png", self.wide)
        cv2.imwrite("data/canny-mid.png", self.mid)
        cv2.imwrite("data/canny-tight.png", self.tight)

    def start_detection(self):
        self.load_image()
        self.convert_to_gray_scale()
        self.detect_edges()
        self.store_image()
