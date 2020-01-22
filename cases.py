import numpy
import cv2

source = cv2.imread("images/source.jpg")
hidden = cv2.imread("images/hidden.jpg")
double = cv2.imread("images/double.jpg")
none = cv2.imread("images/none.jpg")

transform = cv2.xfeatures2d.SIFT_create()

source_key_points, source_descriptors = transform.detectAndCompute(source, None)
hidden_key_points, hidden_descriptors = transform.detectAndCompute(hidden, None)
# double_key_points, double_descriptors = transform.detectAndCompute(double, None)
# none_key_points, none_descriptors = transform.detectAndCompute(none, None)

matcher = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 60})

hidden_matches = matcher.knnMatch(source_descriptors, hidden_descriptors, k=2)
# double_matches = matcher.knnMatch(source_descriptors, double_descriptors, k=2)
# none_matches = matcher.knnMatch(source_descriptors, none_descriptors, k=2)

good_matches_hidden = \
    list(filter(lambda match_list: match_list[0].distance < 0.75 * match_list[1].distance, hidden_matches))
# good_matches_double = \
#     list(filter(lambda match_list: match_list[0].distance < 0.75 * match_list[1].distance, double_matches))
# good_matches_none = \
#     list(filter(lambda match_list: match_list[0].distance < 0.75 * match_list[1].distance, none_matches))

hidden_match_count = len(good_matches_hidden)
# double_match_count = len(good_matches_double)
# none_match_count = len(good_matches_none)

hidden_result = cv2.drawMatchesKnn(
    source, source_key_points, hidden, hidden_key_points, good_matches_hidden, None, flags=2)
# double_result = cv2.drawMatchesKnn(
#     source, source_key_points, double, double_key_points, good_matches_double, None, flags=2)
# none_result = cv2.drawMatchesKnn(
#     source, source_key_points, none, none_key_points, good_matches_none, None, flags=2)

cv2.imwrite("./output/HIDDEN_MATCH_{}.jpg".format(hidden_match_count), hidden_result)
# cv2.imwrite("./output/DOUBLE_MATCH_{}.jpg".format(double_match_count), double_result)
# cv2.imwrite("./output/NONE_MATCH_{}.jpg".format(none_match_count), none_result)
print("{}".format(hidden_match_count))
# print("{}".format(double_match_count))
# print("{}".format(none_match_count))

# for matches, key_points, target, matchCount, name in zip(
#         [good_matches_hidden, good_matches_double, good_matches_none],
#         [hidden_key_points, double_key_points, none_key_points],
#         [hidden, double, none],
#         [hidden_match_count, double_match_count, none_match_count],
#         ["HIDDEN", "DOUBLE", "NONE"],
# ):

for matches, key_points, target, matchCount, name in zip(
        [good_matches_hidden],
        [hidden_key_points],
        [hidden],
        [hidden_match_count],
        ["HIDDEN"],
):

    good_source_points = numpy.float32([source_key_points[match[0].queryIdx].pt for match in matches]) \
        .reshape(-1, 1, 2)
    good_target_points = numpy.float32([key_points[match[0].trainIdx].pt for match in matches]) \
        .reshape(-1, 1, 2)

    homography_matrix, mask = cv2.findHomography(good_target_points, good_source_points, cv2.RANSAC, 5.0)

    height, width, channels = source.shape

    target_realigned = cv2.warpPerspective(target, homography_matrix, (width, height))

    #cv2.imwrite("./output/PERSPECTIVE_{}_MATCH_{}.jpg".format(name, matchCount), target_realigned)
