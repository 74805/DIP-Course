import numpy as np
import cv2
import matplotlib.pyplot as plt


def fourier_transform(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    amplitude = 20 * np.log(np.abs(fshift))
    phase = np.angle(fshift)

    return f, amplitude, phase

def plot_fourier_transform(title, amplitude, phase, fig_num):
    plt.figure(fig_num, figsize=(15, 5))
    plt.subplot(121), plt.imshow(amplitude, cmap='gray')
    plt.title(title + ' Amplitude Spectrum'), plt.xticks([]), plt.yticks([])

    # log scale
    # plt.subplot(132), plt.imshow(np.log(1 + amplitude), cmap='gray')
    # plt.title(title + ' Log Amplitude Spectrum'), plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(phase, cmap='gray')
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
    plt.subplot(111), plt.imshow(amplitude_diff, cmap='gray')
    plt.title('Amplitude Difference'), plt.xticks([]), plt.yticks([])

    # log scale
    # plt.subplot(122), plt.imshow(np.log(1 + amplitude), cmap='gray')
    # plt.title('Log Amplitude Spectrum'), plt.xticks([]), plt.yticks([])


    # c) Compute the Fourier Transform of the images chita.jpeg and zebra.jpeg and plot the amplitude and phase spectrum of them respectively
    # open chita.jpeg and zebra.jpeg
    img_cheetah = cv2.imread('../images/chita.jpeg', cv2.IMREAD_GRAYSCALE)
    img_zebra = cv2.imread('../images/zebra.jpeg', cv2.IMREAD_GRAYSCALE)

    # compute the Fourier Transform of the images
    f_cheetah, amplitude_cheetah, phase_cheetah = fourier_transform(img_cheetah)
    f_zebra, amplitude_zebra, phase_zebra = fourier_transform(img_zebra)

    # plot the amplitude of chita.jpeg
    plt.figure(4, figsize=(15, 5))
    plt.subplot(121), plt.imshow(amplitude_cheetah, cmap='gray')
    plt.title('chita.jpeg - Amplitude Spectrum'), plt.xticks([]), plt.yticks([])

    # plot the phase of zebra.jpeg
    plt.subplot(122), plt.imshow(phase_zebra, cmap='gray')
    plt.title('zebra.jpeg - Phase Spectrum'), plt.xticks([]), plt.yticks([])


    # d) reconstuct a new image using the amplitude of chita.jpeg and the phase of zebra.jpeg and plot the new image
    # resize the images to the same shape
    img_zebra_resized = cv2.resize(img_zebra, (225, 225), interpolation=cv2.INTER_AREA)
    img_cheetah_resized = cv2.resize(img_cheetah, (225, 225), interpolation=cv2.INTER_AREA)

    # compute the Fourier Transform of the resized images
    f_zebra_resized, amplitude_zebra_resized, phase_zebra_resized = fourier_transform(img_zebra_resized)
    f_cheetah_resized, amplitude_cheetah_resized, phase_cheetah_resized = fourier_transform(img_cheetah_resized)

    # combine the amplitude of the cheetah with the phase of the zebra
    combined_spectrum = np.abs(np.fft.fftshift(f_cheetah_resized)) * np.exp(1j * phase_zebra_resized)
    
    # inverse FFT to get the reconstructed image
    combined_ifft_resized = np.fft.ifftshift(combined_spectrum)
    img_new = np.abs(np.fft.ifft2(combined_ifft_resized))
    
    # plot the new image
    plt.figure(6, figsize=(10, 5))
    plt.imshow(img_new, cmap='gray')
    plt.title('Reconstructed Image'), plt.xticks([]), plt.yticks([])

    plt.show()

if __name__ == '__main__':
    main()