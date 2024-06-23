import numpy as np
import cv2
import matplotlib.pyplot as plt


def fourier_transform(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)  # shift the zero frequency component to the center
    amplitude = np.abs(fshift)
    phase = np.angle(fshift)

    return f, amplitude, phase

def plot_fourier_transform(title, amplitude, phase, fig_num):
    plt.figure(fig_num, figsize=(15, 5))
    plt.subplot(131), plt.imshow(amplitude, cmap='gray')
    plt.title(title + ' Amplitude Spectrum'), plt.xticks([]), plt.yticks([])

    # log scale
    plt.subplot(132), plt.imshow(np.log(1 + amplitude), cmap='gray')
    plt.title(title + ' Log Amplitude Spectrum'), plt.xticks([]), plt.yticks([])

    plt.subplot(133), plt.imshow(phase, cmap='gray')
    plt.title(title + ' Phase Spectrum'), plt.xticks([]), plt.yticks([])

def resize_image(img, shape):
    return cv2.resize(img, shape, interpolation=cv2.INTER_LINEAR)

def main():
    # a) Compute the Fourier Transform of the images I.jpg and I_n.jpg and plot the magnitude and phase spectrum of each image.
    # open I.jpg and I_n.jpg
    img = cv2.imread('../images/I.jpg', cv2.IMREAD_GRAYSCALE)
    img_n = cv2.imread('../images/I_n.jpg', cv2.IMREAD_GRAYSCALE)  # noisy image

    # compute the Fourier Transform of the images
    f, amplitude, phase = fourier_transform(img)
    f_n, amplitude_n, phase_n = fourier_transform(img_n)

    # plot the Fourier Transform of the images
    plot_fourier_transform('I.jpg - Original Image', amplitude, phase, 1)
    plot_fourier_transform('I_n.jpg - Noisy Image', amplitude_n, phase_n, 2)


    # b) Compute the amplitude difference between the Fourier Transform of the images plot
    # compute amplitude difference
    amplitude_diff = np.abs(amplitude - amplitude_n)

    # plot amplitude difference
    plt.figure(3, figsize=(10, 5))
    plt.subplot(121), plt.imshow(amplitude_diff, cmap='gray')
    plt.title('Amplitude Difference'), plt.xticks([]), plt.yticks([])

    # log scale
    plt.subplot(122), plt.imshow(np.log(1 + amplitude), cmap='gray')
    plt.title('Log Amplitude Spectrum'), plt.xticks([]), plt.yticks([])


    # c) Compute the Fourier Transform of the images chita.jpeg and zebra.jpeg and plot the amplitude and phase spectrum of them respectively
    # open chita.jpeg and zebra.jpeg
    img_chita = cv2.imread('../images/chita.jpeg', cv2.IMREAD_GRAYSCALE)
    img_zebra = cv2.imread('../images/zebra.jpeg', cv2.IMREAD_GRAYSCALE)

    # resize images to the same shape
    target_shape = (256, 256)
    img_chita_resized = resize_image(img_chita, target_shape)
    img_zebra_resized = resize_image(img_zebra, target_shape)

    # compute the Fourier Transform of the images
    f_chita, amplitude_chita, phase_chita = fourier_transform(img_chita_resized)
    f_zebra, amplitude_zebra, phase_zebra = fourier_transform(img_zebra_resized)

    # plot the amplitude of chita.jpeg
    plt.figure(4, figsize=(15, 5))
    plt.subplot(121), plt.imshow(amplitude_chita, cmap='gray')
    plt.title('chita.jpeg - Amplitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(np.log(1 + amplitude_chita), cmap='gray')
    plt.title('chita.jpeg - Log Amplitude Spectrum'), plt.xticks([]), plt.yticks([])

    # plot the phase of zebra.jpeg
    plt.figure(5, figsize=(15, 5))
    plt.subplot(111), plt.imshow(phase_zebra, cmap='gray')
    plt.title('zebra.jpeg - Phase Spectrum'), plt.xticks([]), plt.yticks([])


    # d) reconstuct a new image using the amplitude of chita.jpeg and the phase of zebra.jpeg and plot the new image
    # reconstruct the image
    f_new = amplitude_chita * np.exp(1j * phase_zebra)
    img_new = np.fft.ifft2(np.fft.ifftshift(f_new)).real

    # plot the new image
    plt.figure(6, figsize=(10, 5))
    plt.imshow(img_new, cmap='gray')
    plt.title('Reconstructed Image'), plt.xticks([]), plt.yticks([])

    plt.show()

if __name__ == '__main__':
    main()