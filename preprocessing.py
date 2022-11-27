import cv2
import numpy


class iris_detection():
    def __init__(self, image_path):
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

    def store_image(self):
        cv2.imwrite("data/test-img-gray.png", self._img)

    def start_detection(self):
        self.load_image()
        self.convert_to_gray_scale()
        self.store_image()
