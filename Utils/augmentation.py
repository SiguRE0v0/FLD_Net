from Utils.traversal import file_traversal
from PIL import Image
import os

dir_img = '../data/training'


def add_suffix(image_path, suffix):
    base, ext = os.path.splitext(image_path)
    new_image_path = f"{base}_{suffix}{ext}"
    return new_image_path


def rotation(dir_list):
    for path in dir_list:
        img = Image.open(path)
        for degree in [90, 180, 270]:
            rotate_img = img.rotate(degree, expand=True)
            suffix = str(degree)
            rotate_dir = add_suffix(path, suffix)
            rotate_img.save(rotate_dir)


def horizontal_flip(dir_list):
    for path in dir_list:
        img = Image.open(path)
        flipped_image = img.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_dir = add_suffix(path, 'flip')
        flipped_image.save(flipped_dir)


if __name__ == '__main__':
    img_dir_list, _ = file_traversal(dir_img)
    rotation(img_dir_list)
    img_dir_list, _ = file_traversal(dir_img)
    horizontal_flip(img_dir_list)
