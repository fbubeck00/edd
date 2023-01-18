import numpy as np


class data_engineering:
    def __init__(self):
        self.dataset = []

    def add_to_dataset(self, img):
         self.dataset.append(img)

    def store_dataset(self):
        dataset = np.asarray(self.dataset)

        # store as csv
        dir = "data/dataset.csv"
        dataset.tofile(dir,  sep=';')

        print("----> Dataset stored in {}".format(dir))
