import numpy as np
import cv2
from matplotlib import pyplot as plt

image_shape = (800, 600)

class watermask_module:
    def __init__(self):
        self.ori_backgd = np.array([])
        self.ori_watermask = np.array([])
        self.__embed_image = np.array([])
        self.__extract_backgd = np.array([])
        self.__extract_watermask = np.array([])

    def embed_wm_in_img(self, background, watermask, emd_bits, img_shape = image_shape):
        background = cv2.resize(background, img_shape)
        watermask = cv2.resize(watermask, img_shape)
        self.ori_backgd = background
        self.ori_watermask = watermask
        self.embed_image = np.zeros(img_shape[::-1], dtype=np.uint8)

        for i in range(img_shape[1]):
            for j in range(img_shape[0]):
                background_b_pixel = "{:08b}".format(background[i][j])
                watermask_b_pixel = "{:08b}".format(watermask[i][j])
                embed_b_img = background_b_pixel[:(8-emd_bits)] + watermask_b_pixel[:emd_bits]
                self.embed_image[i][j] = int(embed_b_img,2)
        return self.embed_image

    def extract_wm_in_img(self, embeded_img, emd_bits, img_shape = image_shape):
        self.__extract_backgd = np.zeros(img_shape[::-1], dtype=np.uint8)
        self.__extract_watermask = np.zeros(img_shape[::-1], dtype=np.uint8)
        
        for i in range(img_shape[1]):
            for j in range(img_shape[0]):
                embeded_b_pixel = "{:08b}".format(embeded_img[i][j])
                extract_background_b_pixel = "{:0<8}".format(embeded_b_pixel[:(8-emd_bits)])
                extract_watermask_b_pixel = "{:0<8}".format(embeded_b_pixel[-emd_bits:])
                self.__extract_backgd[i][j] = int(extract_background_b_pixel,2)
                self.__extract_watermask[i][j] = int(extract_watermask_b_pixel,2)
        return self.__extract_backgd, self.__extract_watermask
