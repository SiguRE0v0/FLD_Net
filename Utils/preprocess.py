import numpy as np
import cv2
import random

def neg_image(img):
    img = np.array(img)
    img_max = np.max(img)
    img_min = np.min(img)
    img_normalized = (img - img_min) / (img_max - img_min)
    neg_img = 1.0 - img_normalized
    return neg_img

def cal_grav(img, img_size):
    moments = cv2.moments(img)
    if moments['m00'] == 0:
        cx, cy = img_size//2, img_size//2
        return cx, cy
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    return cx, cy


def get_patch(image, cx, cy, width=128):
    start_row = max(0, cy - width // 2)
    end_row = min(image.shape[0], cy + width // 2)
    start_col = max(0, cx - width // 2)
    end_col = min(image.shape[1], cx + width // 2)

    patch = image[start_row:end_row, start_col:end_col]
    if patch.shape[0] < width:
        padding_top = (width - patch.shape[0]) // 2
        padding_bottom = width - patch.shape[0] - padding_top
        patch = np.pad(patch, ((padding_top, padding_bottom), (0, 0)), "constant", constant_values=0)
    if patch.shape[1] < width:
        padding_left = (width - patch.shape[1]) // 2
        padding_right = width - patch.shape[1] - padding_left
        patch = np.pad(patch, ((0, 0), (padding_left, padding_right)), "constant", constant_values=0)

    return patch


def patch(img, img_size):
    neg_img = neg_image(img)
    cx, cy = cal_grav(neg_img, img_size)
    patch_img = get_patch(neg_img, cx, cy, img_size)
    return patch_img
