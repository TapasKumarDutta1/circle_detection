from typing import NamedTuple, Optional, Tuple, Generator
import os
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from skimage.draw import circle_perimeter_aa
from utils import (
    CircleParams,
    draw_circle,
    noisy_circle,
    show_circle,
    generate_examples,
)


def create_if_not_present(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)


create_if_not_present("train")
create_if_not_present("train/images")
create_if_not_present("train/labels")
create_if_not_present("valid")
create_if_not_present("valid/images")
create_if_not_present("valid/labels")
create_if_not_present("test")
create_if_not_present("test/images")
create_if_not_present("test/labels")


def save(dir, en, img, label):
    np.save(dir + "/images/" + str(en) + ".npy", img)
    circle_instance = label
    with open(dir + "/labels/" + str(en) + ".txt", "w") as file:
        file.write(str(circle_instance.col) + "\n")
        file.write(str(circle_instance.row) + "\n")
        file.write(str(circle_instance.radius) + "\n")


def generate(num):
    a = generate_examples()
    for en, data in tqdm(enumerate(a)):
        if en < num:
            dir = "/content/train"
            save(dir, en, data[0], data[1])
        elif (en >= num) and (en < int(1.1 * num)):
            dir = "/content/valid"
            save(dir, en, data[0], data[1])
        elif (en >= int(1.1 * num)) and (en < int(1.3 * num)):
            dir = "/content/test"
            save(dir, en, data[0], data[1])
            if en == (1.3 * num) - 1:
                break
