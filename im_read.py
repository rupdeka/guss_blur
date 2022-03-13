"""
Implementation of gaussian filter algorithm
"""
from itertools import product
import cv2 as cv2
from cv2 import COLOR_BGR2GRAY, cvtColor, imread, imshow, waitKey
from numpy import dot, exp, mgrid, pi, ravel, square, uint8, zeros
import numpy as np

import matplotlib.pyplot as plt

def gen_gaussian_kernel(k_siZe, sig):
    c = k_siZe // 2
    x, y = mgrid[0 - c : k_siZe - c, 0 - c : k_siZe - c]
    g = 1 / (2 * pi * sig) * exp(-(square(x) + square(y)) / (2 * square(sig)))
    return g


def gaussian_filter(img, k_siZe, sig):
    height, width = img.shape[0], img.shape[1]
    
    dheight = height - k_siZe + 1
    dwidth = width - k_siZe + 1

    
    img_array = zeros((dheight * dwidth, k_siZe * k_siZe))
    row = 0
    for i, j in product(range(dheight), range(dwidth)):
        window = ravel(img[i : i + k_siZe, j : j + k_siZe])
        img_array[row, :] = window
        row += 1

    #  turn the kernel into shape(k*k, 1)
    gaus_kernel = gen_gaussian_kernel(k_siZe, sig)
    filter_array = ravel(gaus_kernel)

    # reshape and get the dst image
    dst = dot(img_array, filter_array).reshape(dheight, dwidth).astype(uint8)

    return dst


if __name__ == "__main__":
    # read original image
    img = imread('16487_SerenaWilliams_33_f.jpg')
    # turn image in gray scale value
    gray = cvtColor(img, COLOR_BGR2GRAY)

    # get values with two different mask size
    gaussian31x31_1 = gaussian_filter(gray, 31, sig=1)
    histogram1, bin_edges = np.histogram(gaussian31x31_1, bins=256, range=(0, 1))
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixel count")
    plt.xlim([0.0, 1.0])  # <- named arguments do not work here
    plt.plot(bin_edges[0:-1], histogram1)  # <- or here
    plt.show()
    cv2.imwrite('histogram1.jpg', histogram1)
    
    gaussian31x31_2 = gaussian_filter(gray, 31, sig=2)
    histogram2, bin_edges = np.histogram(gaussian31x31_2, bins=256, range=(0, 1))
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixel count")
    plt.xlim([0.0, 1.0])  # <- named arguments do not work here
    plt.plot(bin_edges[0:-1], histogram2)  # <- or here
    plt.show()
    cv2.imwrite('histogram2.jpg', histogram2)
    
    gaussian31x31_3 = gaussian_filter(gray, 31, sig=3)
    histogram3, bin_edges = np.histogram(gaussian31x31_3, bins=256, range=(0, 1))
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixel count")
    plt.xlim([0.0, 1.0])  # <- named arguments do not work here
    plt.plot(bin_edges[0:-1], histogram3)  # <- or here
    plt.show()
    cv2.imwrite('histogram3.jpg', histogram3)
    
    gaussian31x31_4 = gaussian_filter(gray, 31, sig=4)
    histogram4, bin_edges = np.histogram(gaussian31x31_4, bins=256, range=(0, 1))
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixel count")
    plt.xlim([0.0, 1.0])  # <- named arguments do not work here
    plt.plot(bin_edges[0:-1], histogram4)  # <- or here
    plt.show()
    cv2.imwrite('histogram4.jpg', histogram4)
    
    gaussian31x31_5 = gaussian_filter(gray, 31, sig=5)
    histogram5, bin_edges = np.histogram(gaussian31x31_5, bins=256, range=(0, 1))
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixel count")
    plt.xlim([0.0, 1.0])  # <- named arguments do not work here
    plt.plot(bin_edges[0:-1], histogram5)  # <- or here
    plt.show()
    cv2.imwrite('histogram5.jpg', histogram5)

    
    print(np.linalg.det(np.corrcoef(gaussian31x31_1,gaussian31x31_2)))
    print(np.linalg.det(np.corrcoef(gaussian31x31_1,gaussian31x31_3)))
    print(np.linalg.det(np.corrcoef(gaussian31x31_1,gaussian31x31_4)))
    print(np.linalg.det(np.corrcoef(gaussian31x31_1,gaussian31x31_5)))

    print(np.linalg.det(np.corrcoef(gaussian31x31_2,gaussian31x31_3)))
    print(np.linalg.det(np.corrcoef(gaussian31x31_2,gaussian31x31_4)))
    print(np.linalg.det(np.corrcoef(gaussian31x31_2,gaussian31x31_5)))

    print(np.linalg.det(np.corrcoef(gaussian31x31_3,gaussian31x31_4)))
    print(np.linalg.det(np.corrcoef(gaussian31x31_3,gaussian31x31_5)))

    print(np.linalg.det(np.corrcoef(gaussian31x31_4,gaussian31x31_5)))

    
    # show result images
    imshow("gaussian filter with 31x31 mask", gaussian31x31_1)
    cv2.imwrite('gaussian31x31_1.jpg', gaussian31x31_1)
    imshow("gaussian filter with 31x31 mask", gaussian31x31_2)
    cv2.imwrite('gaussian31x31_2.jpg', gaussian31x31_2)
    imshow("gaussian filter with 31x31 mask", gaussian31x31_3)
    cv2.imwrite('gaussian31x31_3.jpg', gaussian31x31_3)
    imshow("gaussian filter with 31x31 mask", gaussian31x31_4)
    cv2.imwrite('gaussian31x31_4.jpg', gaussian31x31_4)
    imshow("gaussian filter with 31x31 mask", gaussian31x31_5)
    cv2.imwrite('gaussian31x31_5.jpg', gaussian31x31_5)
    waitKey()
