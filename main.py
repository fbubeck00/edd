import os
import preprocessing
import preprocessing_v2
import data_engineering
import glob
import cv2


def main():
    print("Starting...")

    # Dataset init
    dataset = data_engineering.data_engineering()

    # iterate over files in folder and call preprocessing method
    for img in glob.glob("data/image_dataset/*.bmp"):
        # extract file name with regex
        text = os.path.basename(img)
        image_name, _, _ = text.partition('.')

        # read image from folder
        image = cv2.imread(img)

        # start preprocess image
        detector = preprocessing_v2.iris_detection_v2(image, image_name)
        img_result = detector.detect_contours()
        print("preprocessed image: {}".format(image_name))

        dataset.add_to_dataset(img_result)

    # store dataset
    dataset.store_dataset()


if __name__ == '__main__':
    main()
