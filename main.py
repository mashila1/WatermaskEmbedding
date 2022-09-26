import numpy as np
import cv2
from matplotlib import pyplot as plt
from WM_module import watermask_module

def show_image(desc, image):
    cv2.imshow(str(desc), image)
    # 按下任意鍵則關閉所有視窗
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

'''def display_result(ori_bg, ori_wm, embed, ext_bg, ext_wm):
    imgs = np.hstack([one_bit_embed.ori_backgd, embed_1b_image])
    show_image('1b origin(right) and embeded(left) background', imgs)
    imgs = np.hstack([one_bit_embed.ori_backgd, ext_1b_ori_backbg])
    show_image('1b origin(right) and extract(left) background', imgs)
    imgs = np.hstack([one_bit_embed.ori_watermask, ext_1b_ori_watermask])
    show_image('1b origin(right) and extract(left) watermask', imgs)'''

img_shape = (800, 600)
ori_img = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
watermask = cv2.imread('watermask1.jpg', cv2.IMREAD_GRAYSCALE)

# one bit embeding
one_bit_embed = watermask_module()
embed_1b_image = one_bit_embed.embed_wm_in_img(ori_img, watermask, 1)
ext_1b_ori_backbg, ext_1b_ori_watermask = one_bit_embed.extract_wm_in_img(embed_1b_image, 1)

imgs = np.hstack([one_bit_embed.ori_backgd, embed_1b_image])
show_image('1b origin(right) and embeded(left) background', imgs)
imgs = np.hstack([one_bit_embed.ori_backgd, ext_1b_ori_backbg])
show_image('1b origin(right) and extract(left) background', imgs)
imgs = np.hstack([one_bit_embed.ori_watermask, ext_1b_ori_watermask])
show_image('1b origin(right) and extract(left) watermask', imgs)

# two bit embeding
two_bit_embed = watermask_module()
embed_2b_image = two_bit_embed.embed_wm_in_img(ori_img, watermask, 2)
ext_2b_ori_backbg, ext_2b_ori_watermask = two_bit_embed.extract_wm_in_img(embed_2b_image, 2)

imgs = np.hstack([two_bit_embed.ori_backgd, embed_2b_image])
show_image('2b origin(right) and embeded(left) background', imgs)
imgs = np.hstack([two_bit_embed.ori_backgd, ext_2b_ori_backbg])
show_image('2b origin(right) and extract(left) background', imgs)
imgs = np.hstack([two_bit_embed.ori_watermask, ext_2b_ori_watermask])
show_image('2b origin(right) and extract(left) watermask', imgs)

# three bit embeding
three_bit_embed = watermask_module()
embed_3b_image = three_bit_embed.embed_wm_in_img(ori_img, watermask, 3)
ext_3b_ori_backbg, ext_3b_ori_watermask = three_bit_embed.extract_wm_in_img(embed_3b_image, 3)

imgs = np.hstack([three_bit_embed.ori_backgd, embed_3b_image])
show_image('3b origin(right) and embeded(left) background', imgs)
imgs = np.hstack([three_bit_embed.ori_backgd, ext_3b_ori_backbg])
show_image('3b origin(right) and extract(left) background', imgs)
imgs = np.hstack([three_bit_embed.ori_watermask, ext_3b_ori_watermask])
show_image('3b origin(right) and extract(left) watermask', imgs)
