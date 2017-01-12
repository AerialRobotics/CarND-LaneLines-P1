####################################################################
#
# Andrew Baker
# Self Driving Car Engineer
# Project 1:  Finding Lane Lines on the Road
#
# File Name:  P1.py
# Due Date:  12/06/16
#
# Description:  In this project you will develop code to identify
#  lane lines on the road.  The pipeline will be developed on a
#  series of individual images.  Later the result will be applied
#  to a video stream.  The final result should look like
#  "raw-lines-example.mp4.
#
#
#####################################################################

# Imports
import numpy as np
import cv2
from moviepy.editor import VideoFileClip


def grayscale(image):
    """
    :param image: original image to be converted
    Applies the Grayscale transform
    This will return an image with only one color channel
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gaussian_blur(image, kernel):
    """Applies a Gaussian Noise kernel
    :param image:  image to blur
    :param kernel:  size of gaussian kernel
    :return: blurred image
    """
    return cv2.GaussianBlur(image, (kernel, kernel), 0)


def canny(image, sigma=0.33):
    """Applies the Canny transform
    :param image:  image to apply canny algorithm
    :param sigma:  sigma value used to compute canny
    :return:  edge image
    """
    # compute the media of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)


def region_of_interest(image, vert):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    :param image:  image to apply mask on.
    :param vert:  points of the 4 sided mask polygon
    :return:  image mask
    """
    # defining a blank mask to start with
    mask = np.zeros_like(image)

    # defining a 3 channel or 1 channel color to fill the mask with depending on
    # the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vert, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def hough_lines(image, f_rho, f_theta, f_threshold, f_min_line_len, f_max_line_gap):
    """
    `image` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    :param image: image to apply hough transform on
    :param f_rho: distance resolution in pixels of the Hough grid
    :param f_theta: angular resolution in radians of the Hough grid
    :param f_threshold: minimum number of votes (intersections in Hough grid cell
    :param f_min_line_len: minimum number of pixels making up a line
    :param f_max_line_gap: maximum gap in pixels between connectible line segments
    :return: line image
    """
    lines = cv2.HoughLinesP(image,
                            f_rho,
                            f_theta,
                            f_threshold,
                            np.array([]),
                            minLineLength=f_min_line_len,
                            maxLineGap=f_max_line_gap)
    line_img = np.zeros((*image.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def draw_lines(image, lines, color=(255, 0, 255), thickness=7):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    :param image:  hough image to draw lines on.
    :param lines:  lines found by the hough transform
    :param color:  color to draw the lines
    :param thickness:  line thickness
    """

    # separate lines by their slopes to group left and right lane lines
    right = []
    left = []

    for x1, y1, x2, y2 in lines[:, 0]:
        m = (float(y2) - y1) / (x2 - x1)
        b = float(y1) - m * x1

        # filter the slope between 0.55 and 0.70.  If the slope falls out of this
        # range that particular line is not included in the dataset.

        if (m >= 0.45) and (m <= 0.75):
            right.append([x1, y1, x2, y2, m, b])
        elif (m >= -0.85) and (m <= -0.35):
            left.append([x1, y1, x2, y2, m, b])

    # this defines how long our lines will be.  The lines should start
    # at the bottom of the drawing and extend up to the top of the masked
    # region.
    y_bottom = image.shape[0]
    y_top = 322  # 460

    # calculate and draw the right lane
    if right:
        right_slope_ave = np.mean(np.array(right)[:, 4])
        right_b = np.mean(np.array(right)[:, 5])
        x_right_bottom = int((y_bottom - right_b) / right_slope_ave)
        x_right_top = int((y_top - right_b) / right_slope_ave)
        cv2.line(image, (x_right_bottom, y_bottom), (x_right_top, y_top), color, thickness)

    # calculate and draw the left lane
    if left:
        left_slope_ave = np.mean(np.array(left)[:, 4])
        left_b = np.mean(np.array(left)[:, 5])
        x_left_bottom = int((y_bottom - left_b) / left_slope_ave)
        x_left_top = int((y_top - left_b) / left_slope_ave)
        cv2.line(image, (x_left_bottom, y_bottom), (x_left_top, y_top), color, thickness)


def weighted_img(image, initial_img, α=0.8, β=1.0, λ=0.):
    """
    `image` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    :param image: hough line image
    :param initial_img: original image
    :param α: alpha blending value
    :param β: beta blending value
    :param λ: gamma value
    """
    return cv2.addWeighted(initial_img, α, image, β, λ)

'''
##########################
# Main Graphics Pipeline #
##########################
for file in os.listdir("test_images/"):
    # Read in each image one at a time
    img = cv2.imread("test_images/" + file)

    # display original image
    cv2.imshow("image", img)
    cv2.waitKey(0)

    # convert image to grayscale
    img_gray = grayscale(img)

    # display grayscale image
    cv2.imshow("gray image", img_gray)
    cv2.waitKey(0)

    # apply Gaussian blur to get rid of high frequencies within image
    kernel_size = 5             # size of gaussian kernel
    img_blur = gaussian_blur(img_gray, kernel_size)

    # display blurred image
    cv2.imshow("blurred image", img_blur)
    cv2.waitKey(0)

    # Use canny edge function to detect potential edge points
    img_canny = canny(img_blur, sigma=0.33)

    # display edge image
    cv2.imshow("edge", img_canny)
    cv2.waitKey(0)

    # Create a masked edges image
    # Assume the camera is located in the same place for each image.
    ysize = img_canny.shape[0]
    xsize = img_canny.shape[1]
    vertices = np.array([[(130, ysize),
                        (xsize / 2 - 30, 322),
                        (xsize / 2 + 25, 322),
                        (xsize - 40, ysize)]],
                        dtype=np.int32)
    img_masked = region_of_interest(img_canny, vertices)

    # display edge image mask
    cv2.imshow("edge image mask", img_masked)
    cv2.waitKey(0)

    # Now find the lines on the masked edge image using hough transform
    # Define the Hough transform parameters
    rho = 1               # distance resolution in pixels of the Hough grid
    theta = np.pi / 180   # angular resolution in radians of the Hough grid
    threshold = 15        # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 40     # maximum gap in pixels between connectible line segments
    img_hough = hough_lines(img_masked,
                            rho,
                            theta,
                            threshold,
                            min_line_length,
                            max_line_gap)

    # display line image
    cv2.imshow("line image", img_hough)
    cv2.waitKey(0)

    # merge the line image over the original image
    img_final = weighted_img(img_hough, img, α=0.8, β=1.0, λ=0.)

    # display final image
    cv2.imshow("final image", img_final)
    cv2.waitKey(0)

    # save final image
    cv2.imwrite("final_images/" + file, img_final)

    cv2.destroyAllWindows()
'''


def process_image(image):

    output = False

    if output:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("original", image)
        cv2.waitKey(0)

    # convert image to grayscale
    img_gray = grayscale(image)

    # apply Gaussian blur to get rid of high frequencies within image
    kernel_size = 5             # size of gaussian kernel
    img_blur = gaussian_blur(img_gray, kernel_size)

    year = 2017
    print("Happy New {}!!!!".format(year))

    if output:
        cv2.imshow("img_gray_blur", img_blur)
        cv2.waitKey(0)

    # Use canny edge function to detect potential edge points
    img_canny = canny(img_blur, sigma=0.33)

    if output:
        cv2.imshow("img_edge", img_canny)
        cv2.waitKey(0)

    # Create a masked edges image
    # Assume the camera is located in the same place for each image.
    ysize = img_canny.shape[0]
    xsize = img_canny.shape[1]

    # use these vertices for the white.mp4 and yellow.mp4 videos
    vertices = np.array([[(100, ysize),
                        (xsize / 2 - 30, 322),
                        (xsize / 2 + 30, 322),
                        (xsize - 40, ysize)]],
                        dtype=np.int32)

    '''
    # use these vertices for the extra.mp4 video
    vertices = np.array([[(100, ysize - 0),
                        (xsize / 2 - 30, 400),
                        (xsize / 2 + 50, 400),
                        (xsize - 40, ysize - 0)]],
                        dtype=np.int32)
    '''
    img_masked = region_of_interest(img_canny, vertices)

    if output:
        cv2.imshow("mask", img_masked)
        cv2.waitKey(0)

    # Now find the lines on the masked edge image using hough transform
    # Define the Hough transform parameters
    rho = 1               # distance resolution in pixels of the Hough grid
    theta = np.pi / 180   # angular resolution in radians of the Hough grid
    threshold = 15        # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 30  # minimum number of pixels making up a line
    max_line_gap = 50     # maximum gap in pixels between connectible line segments
    img_hough = hough_lines(img_masked,
                            rho,
                            theta,
                            threshold,
                            min_line_length,
                            max_line_gap)

    if output:
        cv2.imshow("lines", img_hough)
        cv2.waitKey(0)

    # merge the line image over the original image
    img_final = weighted_img(img_hough, image, α=0.8, β=1.0, λ=0.)

    if output:
        cv2.imshow("final", img_final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # return img_final
    # img_final = weighted_img(img_canny, image, α=0.8, β=1.0, λ=0.)
    img_final = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)
    cv2.imshow("final", img_final)
    #cv2.waitKey(0)
    return img_final

'''
white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
'''


yellow_output = 'yellow.mp4'
clip2 = VideoFileClip('solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)

'''
challenge_output = 'extra.mp4'
clip2 = VideoFileClip('challenge.mp4')
challenge_clip = clip2.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)
'''