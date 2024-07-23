import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_matches(image1, image2):
    # Load the images
    img1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for both images
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Initialize the FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Perform matching
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return good_matches, keypoints1, keypoints2

def dlt(matrix1, matrix2):
    A = np.zeros((8, 9))
    # each two coordinates gives two equations that comes from the homography relation.
    # it comes from the analytic solution:
    # -x*h1 - y*h2 - 1*h3 - 0*h4 - 0*h5 - 0*h6 + xu*h7 + yu*h8 + u*h9 = 0
    # -0*h1 - 0*h2 - 0*h3 - x*h4 - y*h5 - 1*h6 + xv*h7 + yv*h8 + v*h9 = 0
    for i in range(4):
        x, y = matrix1[i]
        u, v = matrix2[i]
        A[2 * i] = [-x, -y, -1, 0, 0, 0, x * u, y * u, u]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, x * v, y * v, v]

    # extract the SVD form
    U, S, Vt = np.linalg.svd(A)

    # the last row of Vt is the solution vector
    h = Vt[-1].reshape(3, 3)

    # normalize the matrix by the index [2,2, so index [2,2] would be 1
    h = h / h[2, 2]

    return h

# from the lecture:
# step (i): randomly select a sample of 4 data points from S.
# step (ii): determine the set of data points Si which are within a distance threshold t of the model.
# step (iii): if the the number of inliers is greater than some threshold T
#             re-estimate the model using all the points in Si.
# step (iv): if the size of Si is less than T select a new subset and repeat steps 1-4.
# step (v): After N trials select the largest consensus set Si, re-estimate the model using all the points in the subset Si.
def RANSAC(coordinates1, coordinates2, threshold, max_iterations=1000):
    best_homography = None
    max_inliers = 0

    for _ in range(max_iterations):
        # randomly select 4 points
        idx = np.random.choice(len(coordinates1), 4, replace=False)
        pts1 = coordinates1[idx]
        pts2 = coordinates2[idx]

        # compute the homography using DLT
        homography = dlt(pts1, pts2)

        # apply the homography to all points
        coordinates1_homogeneous = np.column_stack((coordinates1, np.ones(coordinates1.shape[0])))
        coordinates2_projected = (homography @ coordinates1_homogeneous.T).T
        coordinates2_projected /= coordinates2_projected[:, 2][:, np.newaxis]

        # calculate the distance between the projected points and the actual points
        distances = np.linalg.norm(coordinates2 - coordinates2_projected[:, :2], axis=1)

        # determine the number of inliers
        inliers = distances < threshold
        num_inliers = np.sum(inliers)

        # update the best homography if the current one has more inliers
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_homography = homography

    return best_homography, max_inliers

def stitch_images(image1, image2, homography):
    # Get images height and width , turn them to 1X2 vectors
    h1, w1 = cv2.imread(image1).shape[:2]
    h2, w2 = cv2.imread(image2).shape[:2]
    corners1 = np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32).reshape(-1, 1, 2)
    corners2 = np.array([[0, 0], [0, h2], [w2, h2], [w2, 0]], dtype=np.float32).reshape(-1, 1, 2)
    # Translate the homography to take under account the field of view of the second image
    corners2_transformed = cv2.perspectiveTransform(corners2, np.linalg.inv(homography).astype(np.float32))
    all_corners = np.concatenate((corners1, corners2_transformed), axis=0)
    x, y, w, h = cv2.boundingRect(all_corners)

    # Adjust the homography matrix to map from img2 to img1
    H_adjusted = np.linalg.inv(homography)

    # Warp the images
    img1_warped = cv2.warpPerspective(cv2.imread(image1), np.eye(3), (w, h))
    img2_warped = cv2.warpPerspective(cv2.imread(image2), H_adjusted, (w, h))

    # Combine the warped images into a single output image
    output = cv2.addWeighted(img1_warped, 0.5, img2_warped, 0.5, 0)
    plt.imshow(img1_warped)
    plt.figure()
    plt.imshow(img2_warped)

    # Create a mask for the overlapping region
    mask1 = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask1, [np.int32(corners1)], (255))
    mask2 = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask2, [np.int32(corners2_transformed)], (255))
    overlap_mask = cv2.bitwise_and(mask1, mask2)
    not_overlap_img2_mask = cv2.bitwise_and(cv2.bitwise_not(overlap_mask), mask2)

    # Blend only the overlapping region

    blended = cv2.addWeighted(img1_warped, 0.5, img2_warped, 0.5, 0)

    # Copy img1_warped and img2_warped to blended using the overlap_mask
    blended = cv2.bitwise_and(blended, blended, mask=overlap_mask)
    blended += cv2.bitwise_and(img1_warped, img1_warped, mask=cv2.bitwise_not(overlap_mask))
    blended += cv2.bitwise_and(img2_warped, img2_warped, mask=not_overlap_img2_mask)

    plt.figure()
    plt.imshow(blended, cmap='gray')
    cv2.imwrite('panoramic_image.jpg', blended)

# Main
# Make sure you use the right path
image1 = 'Hanging1.png'
image2 = 'Hanging2.png'

# Calculate good matches between the images and obtain keypoints
matches, keypoints1, keypoints2 = calculate_matches(image1, image2)

# Extract coordinates of the keypoints
coordinates1 = np.float32([keypoints1[match.queryIdx].pt for match in matches])
coordinates2 = np.float32([keypoints2[match.trainIdx].pt for match in matches])

# RANSAC to find the best homography
homography, inliers = RANSAC(coordinates1, coordinates2, threshold=5)

# Stitch the images together
stitch_images(image1, image2, homography)
