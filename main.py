import preprocessing


def main():
    print("Starting...")

    image_path1 = "data/as-test1"
    image_path2 = "data/as-test2"
    image_path3 = "data/as-test3"
    typ = "jpg"

    id = preprocessing.iris_detection(image_path1, typ)
    id.start_detection()

    id2 = preprocessing.iris_detection(image_path2, typ)
    id2.start_detection()

    id3 = preprocessing.iris_detection(image_path3, typ)
    id3.start_detection()


if __name__ == '__main__':
    main()
