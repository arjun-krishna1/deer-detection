import os
import random
import shutil

PATH_0 = "data/img/0"
PATH_0_PREFIX = "0_"

PATH_1 = "data/img/1"
PATH_1_PREFIX = "1_"

FILE_EXTENSION = ".jpg"

BASE_PATH = "data/img"
TRAIN_0_PATH = BASE_PATH + "train/0"

TRAIN_PERCENT = 0.65
VALIDATE_PERCENT = 0.15
TEST_PERCENT = 0.2


def rename_files(path, prefix, ext):
    os.chdir(path)

    old_names = os.listdir(path)
    counter = 0
    for name in old_names:
        os.rename(name, prefix+str(counter)+ext)
        counter += 1


def make_dirs(base_path):
    new_dirs = ["/train/", "/train/0", "/train/1",
                "/validate/", "/validate/0", "/validate/1",
                "/test/", "/test/0", "/test/1"]

    for dir in new_dirs:
        os.mkdir(base_path + dir)


def move_img(base_path):
    train0, val0, test0 = get_new_fnames(base_path, "/0")
    train1, val1, test1 = get_new_fnames(base_path, "/1")
    move_files(train0, "data/img/0/", "data/img/train/0/")
    move_files(train1, "data/img/1/", "data/img/train/1/")

    move_files(val0, "data/img/0/", "data/img/validate/0/")
    move_files(val1, "data/img/1/", "data/img/validate/1/")

    move_files(test0, "data/img/0/", "data/img/test/0/")
    move_files(test1, "data/img/1/", "data/img/test/1/")


def move_files(file_names, from_path, to_path):
    for file_name in file_names:
        shutil.move(from_path + file_name, to_path + file_name)


def get_new_fnames(base_path, path_suffix):
    path = base_path + path_suffix
    file_names = os.listdir(path)
    random.Random(42069).shuffle(file_names)

    train_end = int(TRAIN_PERCENT*len(file_names))
    validate_end = train_end + int(VALIDATE_PERCENT*len(file_names))
    test_end = len(file_names)

    train_names = [file_names[i] for i in range(train_end)]
    validate_names = [file_names[i] for i in range(train_end, validate_end)]
    test_names = [file_names[i] for i in range(validate_end, test_end)]

    return train_names, validate_names, test_names


if __name__ == "__main__":
    rename_files(PATH_0, PATH_0_PREFIX, FILE_EXTENSION)
    rename_files(PATH_1, PATH_1_PREFIX, FILE_EXTENSION)
    make_dirs(BASE_PATH)
    move_img(BASE_PATH)
