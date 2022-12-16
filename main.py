import preprocessing


def main():
    print("Starting...")

    image_path1 = "data/as-test1"
    image_path2 = "data/as-test2"
    image_path3 = "data/as-test3"
    image_path4 = "data/as-test4"
    typ = "jpg"

    id = preprocessing.iris_detection(image_path1, typ)
    id.detect_contours()

    id2 = preprocessing.iris_detection(image_path2, typ)
    id2.detect_contours()

    id3 = preprocessing.iris_detection(image_path3, typ)
    id3.detect_contours()

    id4 = preprocessing.iris_detection(image_path4, typ)
    id4.detect_contours()


if __name__ == '__main__':
    main()
