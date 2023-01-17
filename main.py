import os
import preprocessing
import preprocessing_v2
import glob
import cv2


def main():
    print("Starting...")

    for img in glob.glob("data/image_dataset/*.bmp"):
        # extract file name with regex
        text = os.path.basename(img)
        image_name, _, _ = text.partition('.')

        # read image from folder
        image = cv2.imread(img)

        # start preprocess image
        detector = preprocessing_v2.iris_detection_v2(image, image_name)
        detector.detect_contours()

        print("preprocessed image: {}".format(image_name))


if __name__ == '__main__':
    main()
