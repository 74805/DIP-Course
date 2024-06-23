import numpy as np
import cv2
import matplotlib.pyplot as plt


def fourier_transform(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)  # shift the zero frequency component to the center
    amplitude = np.abs(fshift)
    phase = np.angle(fshift)

    return f, amplitude, phase

def plot_fourier_transform(title, amplitude, phase):
    plt.subplot(131), plt.imshow(amplitude, cmap='gray')
    plt.title(title + ' Amplitude Spectrum'), plt.xticks([]), plt.yticks([])

    # log scale
    plt.subplot(132), plt.imshow(np.log(1 + amplitude), cmap='gray')
    plt.title(title + ' Log Amplitude Spectrum'), plt.xticks([]), plt.yticks([])

    plt.subplot(133), plt.imshow(phase, cmap='gray')
    plt.title(title + ' Phase Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()



def main():
    # a) Compute the Fourier Transform of the images I.jpg and I_n.jpg and plot the magnitude and phase spectrum of each image.
    # open I.jpg and I_n.jpg
    img = cv2.imread('../images/I.jpg', cv2.IMREAD_GRAYSCALE)
    img_n = cv2.imread('../images/I_n.jpg', cv2.IMREAD_GRAYSCALE)  # noisy image

    # compute the Fourier Transform of the images
    f, amplitude, phase = fourier_transform(img)
    f_n, amplitude_n, phase_n = fourier_transform(img_n)

    # plot the Fourier Transform of the images
    plot_fourier_transform('I.jpg - Original Image', amplitude, phase)
    plot_fourier_transform('I_n.jpg - Noisy Image', amplitude_n, phase_n)




if __name__ == '__main__':
    main()