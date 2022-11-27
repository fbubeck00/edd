import preprocessing


def main():
    print("Starting...")
    image_path = "data/as-test.jpg"
    id = preprocessing.iris_detection(image_path)
    id.start_detection()


if __name__ == '__main__':
    main()
