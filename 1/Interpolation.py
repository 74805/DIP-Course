import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def bilinear_interpolation(image_array, scale_factor):
    original_height, original_width = image_array.shape
    new_height, new_width = int(original_height * scale_factor), int(original_width * scale_factor)
    new_image_array = np.zeros((new_height, new_width))

    for i in range(new_height):
        for j in range(new_width):
            # coordinates of the original image
            x = i / scale_factor
            y = j / scale_factor

            # floor of x and y
            x1 = int(x)
            y1 = int(y) 

            # ceiling of x and y
            x2 = min(x1 + 1, original_height - 1)
            y2 = min(y1 + 1, original_width - 1)

            # pixel values of the four corners
            r1 = (y2 - y) * image_array[x1, y1] + (y - y1) * image_array[x1, y2]
            r2 = (y2 - y) * image_array[x2, y1] + (y - y1) * image_array[x2, y2]
            new_image_array[i, j] = (x2 - x) * r1 + (x - x1) * r2

    return new_image_array


def main():
    image = Image.open('../images/peppers.jpg').convert('L')
    image_array = np.array(image)

    # scale by factor of 2
    scaled_image_array_2x = bilinear_interpolation(image_array, 2)
    scaled_image_array_2x = np.clip(scaled_image_array_2x, 0, 255).astype(np.uint8)
    scaled_image_2x = Image.fromarray(scaled_image_array_2x)
    scaled_image_2x.save('output_image_2x.png')

    # scale by factor of 8 (factor of 2, three times)
    scaled_image_array_8x = bilinear_interpolation(image_array, 2)
    scaled_image_array_8x = bilinear_interpolation(scaled_image_array_8x, 2)
    scaled_image_array_8x = bilinear_interpolation(scaled_image_array_8x, 2)
    scaled_image_array_8x = np.clip(scaled_image_array_8x, 0, 255).astype(np.uint8)
    scaled_image_8x = Image.fromarray(scaled_image_array_8x)
    scaled_image_8x.save('output_image_8x.png')

    # displaying the original and scaled images
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image_array, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("2x Scaled Image")
    plt.imshow(scaled_image_array_2x, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("8x Scaled Image")
    plt.imshow(scaled_image_array_8x, cmap='gray')
    plt.axis('off')

    plt.show()


if __name__ == '__main__':
    main()
