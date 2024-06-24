import numpy as np
from scipy import ndimage
import cv2


# Question B1
def conv2d(image, kernel):
    # zero padding the image from both sides, so we could apply the filter at the edges
    conv_ready = np.pad(image, (int(kernel.shape[0]/2), int(kernel.shape[1]/2)))
    result = np.zeros_like(image)  # for python to not override the original image
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            # as learned in exercise and lecture
            result[row, col] = np.sum(conv_ready[row:row + kernel.shape[0], col:col + kernel.shape[1]] * kernel)
    # we handle the images with 255 levels of gray, so should return it as uint-8-bit
    return result.astype(np.uint8)


def main():
    # Question B2.a - Derivative Filter
    img_I = cv2.imread("../images/I.jpg", cv2.IMREAD_GRAYSCALE)  # read the image without BGR, only gray scale
    img_I_n = cv2.imread("../images/I_n.jpg", cv2.IMREAD_GRAYSCALE)  # read the image without BGR, only gray scale

    derive_filter = np.array([[-1, 0, 1]])  # the 1d derivative filter

    # # if this section is commented, uncomment for the 2d filter (to the "end of 2d filter")
    #
    # derive_filter_2d = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])  # 2d filter for horizontal derivative
    # img_I_filtered = conv2d(img_I, derive_filter_2d)
    # img_I_n_filtered = conv2d(img_I_n, derive_filter_2d)
    #
    # # end of 2d filter


    # if this section is commented, uncomment for the 1d filter (to the "end of 1d filter")

    img_I_filtered = np.zeros(img_I.shape)
    # using new loops for 1d more efficient computation
    for i in range(img_I.shape[0]):
        for j in range(img_I.shape[1]):
            # condition added to avoid the edges instead of zero padding
            if j == 0:
                img_I_filtered[i, j] = np.sum(img_I[i, :1] * derive_filter[0, 1:])
            elif j == (img_I.shape[1] - 1):
                img_I_filtered[i, j] = np.sum(img_I[i, j - 1:] * derive_filter[0, 0:-1])
            else:
                img_I_filtered[i, j] = np.sum(img_I[i, j - 1: j + 2] * derive_filter[0, :])

    img_I_n_filtered = np.zeros(img_I_n.shape)
    # same as previous loops, only for I_n image
    for i in range(img_I_n.shape[0]):
        for j in range(img_I_n.shape[1]):
            if j == 0:
                img_I_n_filtered[i, j] = np.sum(img_I_n[i, :1] * derive_filter[0, 1:])
            elif j == (img_I.shape[1] - 1):
                img_I_n_filtered[i, j] = np.sum(img_I_n[i, j - 1:] * derive_filter[0, 0:-1])
            else:
                img_I_n_filtered[i, j] = np.sum(img_I_n[i, j - 1: j + 2] * derive_filter[0, :])

    # end of 1d filter

    cv2.imshow("I filtered", img_I_filtered)
    cv2.waitKey(0)

    cv2.imshow("I_n filtered", img_I_n_filtered)
    cv2.waitKey(0)


    # Question B2.b - Gaussian Filter
    I_dn = ndimage.gaussian_filter(img_I_n, 3)  # second argument is the sigma as shown in exercise
    cv2.imshow("Gaussian Filtered I_n", I_dn)
    cv2.waitKey(0)


    # Question B2.c - Sobel Filter

    # function: ddepth - data type of the image (uint8), dx/dy tells it in which axis to apply the derivative,
    # ksize is the size of the filter (K*K)
    I_dn2 = cv2.Sobel(img_I_n, ddepth=cv2.CV_8U, dx=1, dy=0, ksize=3)

    cv2.imshow('Original Image', img_I_n)
    cv2.imshow('Sobel Filtered', I_dn2)
    cv2.waitKey(0)

    cv2.imwrite("I_dn.jpg", I_dn)
    cv2.imwrite("I_dn2.jpg", I_dn2)


if __name__ == "__main__":
    main()