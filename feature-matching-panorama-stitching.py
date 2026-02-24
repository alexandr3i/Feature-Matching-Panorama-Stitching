import cv2
import numpy as np
import matplotlib.pyplot as plt

picture_a = cv2.imread(r"assets/24.png")
picture_b = cv2.imread(r"assets/11.png")
picture_c = cv2.imread(r"assets/42.png")

im = [picture_a, picture_b, picture_c]
im_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in im]

plt.figure(figsize=(10, 6))
plt.title('Images')
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(picture_a, cv2.COLOR_BGR2RGB))
plt.title('Image 1')
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(picture_b, cv2.COLOR_BGR2RGB))
plt.title('Image 2')
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(picture_c, cv2.COLOR_BGR2RGB))
plt.title('Image 3')
plt.show()

MAX_FEATURES = 750
GOOD_MATCH_PERCENT = 0.20
orb = cv2.ORB_create(MAX_FEATURES)

kp1, d1 = orb.detectAndCompute(im_gray[0], None)
kp2, d2 = orb.detectAndCompute(im_gray[1], None)
kp3, d3 = orb.detectAndCompute(im_gray[2], None)

matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

matches32 = matcher.match(d3, d2, None)
matches32 = sorted(matches32, key=lambda x: x.distance)
numGoodMatches32 = int(len(matches32) * GOOD_MATCH_PERCENT)
good_matches = matches32[:numGoodMatches32]
imMatches32 = cv2.drawMatches(im_gray[2], kp3, im_gray[1], kp2, good_matches, None)

matches21 = matcher.match(d2, d1, None)
matches21 = sorted(matches21, key=lambda x: x.distance)
numGoodMatches21 = int(len(matches21) * GOOD_MATCH_PERCENT)
good_matches = matches21[:numGoodMatches21]
imMatches21 = cv2.drawMatches(im_gray[1], kp2, im_gray[0], kp1, good_matches, None)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(imMatches21[:, :, ::-1])
plt.title('Match 1-2')
plt.subplot(1, 2, 2)
plt.imshow(imMatches32[:, :, ::-1])
plt.title('Match 2-3')
plt.show()


def keypoints_des(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors


def match_keypoints(descriptors1, descriptors2, good_match_percent=0.17):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    num_good_matches = int(len(matches) * good_match_percent)
    good_matches = matches[:num_good_matches]
    return good_matches


def stitch_images(image1, image2, keypoints1, keypoints2, matches):
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
    if homography is None:
        print("The homography could not be calculated.")
        return None

    height1, width1, _ = image1.shape
    height2, width2, _ = image2.shape
    panorama_width = width1 + width2
    panorama_height = max(height1, height2)

    result = cv2.warpPerspective(image2, homography, (panorama_width, panorama_height))
    result[0:height1, 0:width1] = image1

    gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    ret, binary_threshold = cv2.threshold(gray_result, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cropped_result = result[y:y + h, x:x + w]
    return cropped_result


im = [picture_a, picture_b, picture_c]
im_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in im]

keypoints_list = []
descriptors_list = []
for gray_image in im_gray:
    keypoints, descriptors = keypoints_des(gray_image)
    keypoints_list.append(keypoints)
    descriptors_list.append(descriptors)

stitched_image = im[0]
current_keypoints = keypoints_list[0]
current_descriptors = descriptors_list[0]

for i in range(1, len(im)):
    matches = match_keypoints(current_descriptors, descriptors_list[i])
    stitched_result = stitch_images(stitched_image, im[i], current_keypoints, keypoints_list[i], matches)

    if stitched_result is not None:
        stitched_image = stitched_result

        plt.figure(figsize=(10, 10))
        plt.imshow(stitched_image[..., ::-1])
        plt.title(f'Stitched Image {i}')
        plt.axis('off')
        plt.show()

        current_keypoints, current_descriptors = keypoints_des(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY))

gray_panorama = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)
ret, binary_threshold = cv2.threshold(gray_panorama, 1, 255, cv2.THRESH_BINARY)
contours, hier = cv2.findContours(binary_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

max_contour_area = 0
largest_contour = None
for contour in contours:
    contour_area = cv2.contourArea(contour)
    if contour_area > max_contour_area:
        max_contour_area = contour_area
        largest_contour = contour

x, y, width, height = cv2.boundingRect(largest_contour)
new = stitched_image[y:y + height, x:x + width]

plt.figure()
plt.imshow(new[..., ::-1])
plt.title('Final Panorama')
plt.show()
