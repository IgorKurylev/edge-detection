import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tools import *
from lane import Lane
import os
import numpy as np
import cv2

#reading in an image
# image = mpimg.imread('test_images/challenge.jpg')
#
# #printing out some stats and plotting
# print('This image is:', type(image), 'with dimesions:', image.shape)
# plt.imshow(image, cmap='gray')
# plt.show()
# if you wanted to show a single
#color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

if __name__ == "__main__":

    image = mpimg.imread('test_images/challenge2.jpg')

    Lane.purge()
#    plt.imshow(image_pipeline(image))
#    plt.show()

    VERBOSE = False
    for img in os.listdir("test_images/"):
        if img.endswith('jpg'):
            Lane.purge()
            plt.figure(figsize=(12, 8))
            image = mpimg.imread('test_images/{}'.format(img))
            plt.imshow(image_pipeline(image))
            plt.show()
