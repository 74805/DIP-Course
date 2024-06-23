import numpy as np
import cv2
import matplotlib.pyplot as plt


def fourier_transform(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f) # shift the zero frequency component to the center
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    return f, fshift, magnitude_spectrum


def main():
    # open I.jpg and I_n.jpg
    img = cv2.imread('I.jpg', cv2.IMREAD_GRAYSCALE)
    img_n = cv2.imread('I_n.jpg', cv2.IMREAD_GRAYSCALE)

    # compute the Fourier Transform of the images
    f, fshift, magnitude_spectrum = fourier_transform(img)
    f_n, fshift_n, magnitude_spectrum_n = fourier_transform(img_n)


if __name__ == '__main__':
    main()