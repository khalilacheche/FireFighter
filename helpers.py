import cv2
import numpy as np
import math
import copy
from IPython.display import display, Image


def angle_modulo(input_angle):
    revolutions = int((input_angle + np.sign(input_angle) * 180) / (180))

    p1 = truncated_remainder(input_angle + np.sign(input_angle) * 180, 360)
    p2 = (
        np.sign(
            np.sign(input_angle)
            + 2
            * (
                np.sign(
                    math.fabs((truncated_remainder(input_angle + 180, 360)) / (360))
                )
                - 1
            )
        )
    ) * 180

    output_angle = p1 - p2

    return output_angle, revolutions


def truncated_remainder(dividend, divisor):
    divided_number = dividend / divisor
    divided_number = (
        -int(-divided_number) if divided_number < 0 else int(divided_number)
    )

    remainder = dividend - divisor * divided_number

    return remainder


def overlay_image_in_bounds(original_image_shape, overlay_image_shape, position):
    """
    Verify if the overlay image is inside the picture or not
    """
    # x_index,
    return True  # original_image_shape[:, :]


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def overlayPNG(imgBack, imgFront, pos=[0, 0]):
    hf, wf, cf = imgFront.shape
    hb, wb, cb = imgBack.shape
    *_, mask = cv2.split(imgFront)
    maskBGRA = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
    maskBGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    imgRGBA = cv2.bitwise_and(imgFront, maskBGRA)
    imgRGB = cv2.cvtColor(imgRGBA, cv2.COLOR_BGRA2BGR)

    top_pos = pos[1]
    bottom_pos = pos[1] + hf
    right_pos = pos[0] + wf
    left_pos = pos[0]
    clamped_top_idx = clamp(top_pos, 0, hb)
    clamped_bottom_idx = clamp(bottom_pos, 0, hb)
    clamped_left_idx = clamp(left_pos, 0, wb)
    clamped_right_idx = clamp(right_pos, 0, wb)

    offset_top = clamped_top_idx - top_pos
    offset_bottom = hf - (bottom_pos - clamped_bottom_idx)
    offset_left = clamped_left_idx - left_pos
    offset_right = wf - (right_pos - clamped_right_idx)

    imgMaskFull = np.zeros((hb, wb, cb), np.uint8)
    imgMaskFull[
        clamped_top_idx:clamped_bottom_idx, clamped_left_idx:clamped_right_idx, :
    ] = imgRGB[offset_top:offset_bottom, offset_left:offset_right, :]
    imgMaskFull2 = np.ones((hb, wb, cb), np.uint8) * 255
    maskBGRInv = cv2.bitwise_not(maskBGR)
    imgMaskFull2[
        clamped_top_idx:clamped_bottom_idx, clamped_left_idx:clamped_right_idx, :
    ] = maskBGRInv[offset_top:offset_bottom, offset_left:offset_right, :]
    imgBack = cv2.bitwise_and(imgBack, imgMaskFull2)

    imgBack = cv2.bitwise_or(imgBack, imgMaskFull)

    return imgBack


def rotateImage(img, angle, scale=1):
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale)
    img = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(w, h))
    return img


def overlay_robot_coordinates_and_angle(image, robot_pos, robot_angle, contours=None):
    x, y = robot_pos
    angle_degs = robot_angle
    angle_rads = math.radians(robot_angle)
    if contours is not None:
        cv2.drawContours(image, contours, -1, (255, 255, 0), 3)
    ##### Draw robot position
    cv2.circle(image, (x, y), radius=10, color=(0, 0, 255), thickness=-1)

    ##### Draw robot angle direction
    length = 30
    cv2.line(
        image,
        (x, y),
        (
            round(x + length * math.cos(angle_rads)),
            round(y + length * math.sin(angle_rads)),
        ),
        (0, 0, 255),
        2,
    )
    ##### Draw angle 0 direction
    cv2.line(
        image,
        (x, y),
        (
            round(x + length * math.cos(0)),
            round(y + length * math.sin(0)),
        ),
        (0, 0, 255),
        1,
    )
    ##### Display angle value
    cv2.putText(
        image,
        "{:.2f}".format(angle_degs),
        (x + 20, y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )


def overlay_target(image, target_pos):
    cv2.circle(image, target_pos, 10, (0, 0, 255), 10)


def overlay_obstacles(image, scaled_contours, orig_contours=None):
    if orig_contours is not None:
        cv2.drawContours(image, orig_contours, -1, (255, 255, 0))
    cv2.drawContours(image, scaled_contours, -1, (0, 255, 0), 3)


def draw_angled_rec(x0, y0, width, height, angle, img):
    """Draws a rotated angle on the image
    Args:
      x0: the x position of the rect
      y0: the y position of the rect
      width: the width of the rectangle
      height: the height of the rectangle
      angle: the angle of the rectangle
      image: the image to draw on
    Returns: the image with a rotated rectangle drawn on
    """
    _angle = angle * math.pi / 180.0
    b = math.cos(_angle) * 0.5
    a = math.sin(_angle) * 0.5
    pt0 = (int(x0 - a * height - b * width), int(y0 + b * height - a * width))
    pt1 = (int(x0 + a * height - b * width), int(y0 - b * height - a * width))
    pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
    pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

    cv2.line(img, pt0, pt1, (255, 255, 255), 3)
    cv2.line(img, pt1, pt2, (255, 255, 255), 3)
    cv2.line(img, pt2, pt3, (255, 255, 255), 3)
    cv2.line(img, pt3, pt0, (255, 255, 255), 3)


def scale_contours(contours, scale_factor=2.2):
    """Scales the contours by a factor
    Args:
      contours: the contours to scale
      scale_factor (optionnal: default is 2.5): the scale to factor on
    Returns: the scaled version of the contours
    """
    res = copy.deepcopy(contours)
    for i, contour in enumerate(contours):
        M = cv2.moments(contour)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        for j, point in enumerate(contour):
            x_ = point[0][0]
            y_ = point[0][1]
            x_new = (scale_factor * (x_ - cx)) + cx
            y_new = (scale_factor * (y_ - cy)) + cy
            res[i][j][0] = [x_new, y_new]
    return res


def display_image_in_notebook(image):
    _, frame = cv2.imencode(".jpeg", image)
    display(Image(data=frame.tobytes()))
