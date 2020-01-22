import time
from statistics import mean

import scipy
import numpy
import cv2

THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

source = cv2.imread("images/zoom-in.jpg")
target = cv2.imread("images/zoom-out.jpg")

transform = cv2.xfeatures2d.SIFT_create()

source_key_points, source_descriptors = transform.detectAndCompute(source, None)
target_key_points, target_descriptors = transform.detectAndCompute(target, None)

for threshold in THRESHOLDS:
    start = time.time()
    matcher = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})

    matches = matcher.knnMatch(source_descriptors, target_descriptors, k=2)

    good_matches = list(filter(lambda match_list: match_list[0].distance < threshold * match_list[1].distance, matches))

    match_count = len(good_matches)

    result = cv2.drawMatchesKnn(source, source_key_points, target, target_key_points, good_matches, None, flags=2)

    good_source_points = numpy.float32([source_key_points[match[0].queryIdx].pt for match in good_matches]) \
        .reshape(-1, 1, 2)
    good_target_points = numpy.float32([target_key_points[match[0].trainIdx].pt for match in good_matches]) \
        .reshape(-1, 1, 2)

    homography_matrix, mask = cv2.findHomography(good_target_points, good_source_points, cv2.RANSAC, 5.0)

    height, width, channels = source.shape

    target_realigned = cv2.warpPerspective(target, homography_matrix, (width, height))

    print("{}\t{}\t{}".format(threshold, match_count, time.time() - start))
    cv2.imwrite("./output/THRESHOLD_{}_MATCH_{}.jpg".format(threshold, match_count), result)
    #cv2.imwrite("./output/PERSPECTIVE_THRESHOLD_{}_MATCH_{}.jpg".format(threshold, match_count), target_realigned)
