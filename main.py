import os

from util.util_data import load_path


def main():
    paths = load_path(root_path)
    print(paths.sample(10))


if __name__ ==  "__main__":
    print("Hello, world!")
    root_path = os.getcwd()
    main()
