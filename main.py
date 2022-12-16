import preprocessing


def main():
    print("Starting...")

    image_name1 = "as-test1"
    image_name2 = "as-test2"
    image_name3 = "as-test3"
    image_name4 = "as-test4"
    image_type = "jpg"

    id = preprocessing.iris_detection(image_name1, image_type)
    id.detect_contours()

    id2 = preprocessing.iris_detection(image_name2, image_type)
    id2.detect_contours()

    id3 = preprocessing.iris_detection(image_name3, image_type)
    id3.detect_contours()

    id4 = preprocessing.iris_detection(image_name4, image_type)
    id4.detect_contours()


if __name__ == '__main__':
    main()
