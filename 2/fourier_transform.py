import numpy as np
import cv2
import matplotlib.pyplot as plt


def fourier_transform(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)  # shift the zero frequency component to the center
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    phase_spectrum = np.angle(fshift)

    return f, magnitude_spectrum, phase_spectrum


def plot_fourier_transform(title, magnitude_spectrum, phase_spectrum):
    plt.subplot(121), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title(title + ' Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(phase_spectrum, cmap='gray')
    plt.title(title + ' Phase Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()


def main():
    # open I.jpg and I_n.jpg
    img = cv2.imread('../images/I.jpg', cv2.IMREAD_GRAYSCALE)
    img_n = cv2.imread('../images/I_n.jpg', cv2.IMREAD_GRAYSCALE)  # noisy image

    # compute the Fourier Transform of the images
    f, magnitude_spectrum, phase_spectrum = fourier_transform(img)
    f_n, magnitude_spectrum_n, phase_spectrum_n = fourier_transform(img_n)

    # plot the Fourier Transform of the images
    plot_fourier_transform('I.jpg - Original Image', magnitude_spectrum, phase_spectrum)
    plot_fourier_transform('I_n.jpg - Noisy Image', magnitude_spectrum_n, phase_spectrum_n)


if __name__ == '__main__':
    main()