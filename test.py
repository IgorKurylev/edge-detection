import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tools import *
from lane import Lane
import os
import numpy as np
import cv2

if __name__ == "__main__":

    image = mpimg.imread('test_images/solidWhiteRight.jpg')
    # image = mpimg.imread('test_images/solidYellowLeft.jpg')
    Lane.purge()
    plt.imshow(image, cmap='gray')
    plt.show()
    res = image_pipeline(image)
    plt.imshow(res)
    plt.show()

    plt.imsave('res.jpg', res)

    # VERBOSE = False
    # for img in os.listdir("test_images/"):
    #     if img.endswith('jpg'):
    #         Lane.purge()
    #         plt.figure(figsize=(12, 8))
    #         image = mpimg.imread('test_images/{}'.format(img))
    #         plt.imshow(image_pipeline(image))
    #         plt.show()