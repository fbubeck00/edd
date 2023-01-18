import os

import numpy as np

import preprocessing
import preprocessing_v2
from data_engineering import dataset_creation
import glob
import cv2


def main():
    print("Starting...")

    temp_dataset = []

    # iterate over files in folder and call preprocessing method
    for img in glob.glob("data/image_dataset/*.bmp"):
        # extract file name with regex
        text = os.path.basename(img)
        image_name, _, _ = text.partition('.')

        # read image from folder
        image = cv2.imread(img)

        # start preprocess image
        print("preprocess image: {}".format(image_name))
        detector = preprocessing_v2.iris_detection_v2(image, image_name, temp_dataset)
        image_dataset = detector.preprocess_image()

    ###############################
    # Image Dataset
    ###############################
    dataset = np.array(image_dataset)
    print("Dataset format: {}".format(dataset.shape))
    print("Image Format: {}".format(dataset[0].shape))
    print("Number of Images: {}".format(dataset.shape[0]))



if __name__ == '__main__':
    main()
