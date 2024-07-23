import numpy as np
import cv2
import matplotlib.pyplot as plt


def main():
    # Question A2.a - Histogram plot
    img = cv2.imread("../images/leafs.jpg", cv2.IMREAD_GRAYSCALE)  # read the image without BGR, only gray scale
    histogram, bin_edges = np.histogram(img, 256, range=(0, 256))  # separate to 256, so we count each gray level

    # histogram plot
    plt.bar(bin_edges[:-1], histogram, width=1, edgecolor='black')  # additional style parameters
    plt.xlabel('Intensity')
    plt.ylabel('Count')
    plt.title('Histogram')
    plt.xlim([0, 256])
    plt.show()

    # Question A2.b - Contrast Stretch
    # extract f_min and f_max
    histogram_nonzero = np.nonzero(histogram)[0]
    min_intensity = histogram_nonzero[0]  # first non-zero element is f_min
    max_intensity = histogram_nonzero[-1]  # last non-zero element is f_max

    # Apply the equation from the HW1
    multiplication_factor = 255 / (max_intensity - min_intensity)
    contrast_img = np.round(multiplication_factor * (img - min_intensity)).astype(np.uint8)

    # comparison between the first and last leafs images - by using a concat of images
    img_compare = cv2.hconcat([img, contrast_img])

    # simply add a title to each image at the right location
    # function args: image, title, coordinates of the title, font, font size, font color (BGR), thickness of the font
    cv2.putText(img_compare, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img_compare, 'Contrast Enhanced', (img.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


    cv2.imshow("Original Vs. Contrast Stretched Leafs", img_compare)
    cv2.waitKey(0)

    # new histogram (after contrast stretching) - same as above
    contrast_histogram, new_bin_edges = np.histogram(contrast_img, 256, range=(0, 256))
    plt.bar(new_bin_edges[:-1], contrast_histogram, width=1, edgecolor='black')
    plt.xlabel('Intensity')
    plt.ylabel('Count')
    plt.title('Contrast Stretched Histogram')
    plt.xlim([0, 256])
    plt.show()


    # Question A2.c - Histogram Equalization
    # Calculate the CDF of the histogram
    cumulative = np.cumsum(histogram)  # summation of the histogram
    normal_cumulative = cumulative / np.sum(histogram)  # normalize the cumulative curve, so we won't cross 255 after it
    mult_factor = 255 / cumulative[-1]  # we'll multiply each cell by this value to get from lowest to highest levels

    mapping = np.zeros(256)
    for i in range(mapping.size):
        mapping[i] = int(round(float(mult_factor * cumulative[i])))  # the mapping array of previous level to the new one
    mapping = mapping.astype(np.uint8)  # keep it as uint8, not sure if it necessary at this point

    equal_img = mapping[img]  # the mapping happen here, all at once

    # comparison as above
    compare_everything = cv2.hconcat([img, contrast_img, equal_img])

    # titles as above
    cv2.putText(compare_everything, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(compare_everything, 'Contrast Enhanced', (img.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(compare_everything, 'Equalized', (img.shape[1] + contrast_img.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imshow('Balanced Image', compare_everything)
    cv2.waitKey(0)

    # New Histogram (after histogram equalization) - as above
    equal_histogram, equal_bin_edges = np.histogram(equal_img, 256, range=(0, 256))
    plt.bar(equal_bin_edges[:-1], equal_histogram, width=1, edgecolor='black')
    plt.xlabel('Intensity')
    plt.ylabel('Count')
    plt.title('Histogram Equalization')
    plt.xlim([0, 256])
    plt.show()

    # Histogram Comparison - side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Original Histogram
    ax1.bar(bin_edges[:-1], histogram, alpha=0.7, color='blue', width=1, edgecolor='black')  # just style parameters
    ax1.set_xlabel('Intensity')
    ax1.set_ylabel('Count')
    ax1.set_title('Original Histogram')
    ax1.set_xlim([0, 256])

    # Contrast Stretched Histogram
    ax2.bar(new_bin_edges[:-1], contrast_histogram, alpha=0.7, color='green', width=1, edgecolor='black')
    ax2.set_xlabel('Intensity')
    ax2.set_ylabel('Count')
    ax2.set_title('Contrast Enhanced Histogram')
    ax2.set_xlim([0, 256])

    # Histogram Equalization
    ax3.bar(equal_bin_edges[:-1], equal_histogram, alpha=0.7, color='red', width=1, edgecolor='black')
    ax3.set_xlabel('Intensity')
    ax3.set_ylabel('Count')
    ax3.set_title('Histogram Equalization')
    ax3.set_xlim([0, 256])

    plt.tight_layout()  # makes sure the titles are without overlapping
    plt.show()

if __name__ == "__main__":
    main()
