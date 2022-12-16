import cv2


class testing:
    def __init__(self, image_path, typ):
        self.img_gray = None
        self.img_cropped = None
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

    def detect_circles(self):
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
