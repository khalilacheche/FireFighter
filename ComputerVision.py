import cv2
import math
import numpy as np
from threading import Thread
import helpers

window_name = "Firefighter"
HSV_LOW = [57, 74, 136]
HSV_HIGH = [144, 251, 243]
BLUE_MIN = np.array(HSV_LOW, np.uint8)
BLUE_MAX = np.array(HSV_HIGH, np.uint8)

HSV_LOW = [143, 120, 81]
HSV_HIGH = [183, 177, 180]
RED_MIN = np.array([143, 120, 81], np.uint8)
RED_MAX = np.array([183, 177, 180], np.uint8)
hsvs_red = [
    [np.array([143, 120, 81], np.uint8), np.array([183, 177, 180], np.uint8)],
    [np.array([113, 92, 106], np.uint8), np.array([219, 199, 231], np.uint8)],
    [np.array([143, 26, 156], np.uint8), np.array([253, 238, 245], np.uint8)],
]
MAX_DISTANCE_ROBOT_CENTER = 38
MIN_DISTANCE_ROBOT_CENTER = 36.5


def find_area(image, overlay=False):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_thresholded = cv2.threshold(img_gray, 150, 200, 0)
    image_thresholded = cv2.morphologyEx(
        image_thresholded, cv2.MORPH_CLOSE, kernel=np.ones((9, 9), np.uint8)
    )
    contours, _ = cv2.findContours(
        image_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    contours = [
        cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        for contour in contours
        if cv2.contourArea(contour) > 1000
    ]
    res = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        (x, y), (w, h), r = cv2.minAreaRect(approx)
        x, y, w, h = math.ceil(x), math.ceil(y), math.ceil(w), math.ceil(h)
        if overlay:
            helpers.draw_angled_rec(x, y, w, h, r, image)
        res.append((x, y, w, h, r))
    res = sorted(res, key=lambda x: (x[2] * x[3]), reverse=True)
    return res[0]


def get_area(image, crop_params=None):
    """Calculates the parameters to crop the image on the play area (if no crop parameters are given) and performs the crop
    Args:
      image: the image to process
      crop_params (optionnal): the crop parameters to perform on the image (x,y,width,height,rotation in degrees)
    Returns: the cropped image, a tuple containing the crop parameters if they were not given (x,y,width,height,rotation in degrees)
    If the area was not found, we return the default crop parameters
    """

    if crop_params is None:
        # get the area
        res = find_area(image)
        x, y, w, h, r = (
            (
                image.shape[0] // 2,
                image.shape[1] // 2,
                image.shape[0],
                image.shape[1],
                0,
            )
            if res is None
            else res
        )
    else:
        x, y, w, h, r = crop_params

    # rotate it
    center = (x, y)
    M = cv2.getRotationMatrix2D(center, r, 1.0)
    rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    # crop it
    cropped_image = rotated_image[
        y - (h // 2) : y + (h // 2), x - (w // 2) : x + (w // 2)
    ]
    cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_90_CLOCKWISE)
    if crop_params is None:
        return cropped_image, (x, y, w, h, r)
    else:
        return cropped_image


def find_obstacles(image, overlay=False, verbose=False):
    """
    Retrieve the obstacles of the map from the given image
    Params:
        image: the image to find the obstacles in
        display_image: (optionnal, default False) shows a window of the processed image
        verbose: (optionnal, default False) prints info about what is happening
    Returns:
        obstacles (list of list of list of ints): first dimension represents the obstacle index,
        the second dimension represents the edge index,
        the third dimension represents the x and y components (0 for x, 1 for y)
    """
    image_processing = image.copy()
    img_gray = cv2.cvtColor(image_processing, cv2.COLOR_BGR2GRAY)
    _, image_thresholded = cv2.threshold(img_gray, 60, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        image_thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = [cv2.convexHull(cnt) for cnt in contours]
    contours = [
        cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        for cnt in contours
        if cv2.contourArea(cnt) > 1000
        and cv2.contourArea(cnt) < (0.9 * image.shape[0] * image.shape[1])
    ]
    orig_contours = contours.copy()
    contours = helpers.scale_contours(contours)
    if verbose:
        print("Number of obstacles found = " + str(len(contours)))
    image_processing = np.zeros([image.shape[0], image.shape[1], 3], dtype=np.uint8)
    image_processing.fill(0)
    for contour in contours:
        im = np.zeros([image.shape[0], image.shape[1], 3], dtype=np.uint8)
        im.fill(0)
        cv2.fillPoly(im, pts=[contour], color=(255, 255, 255))
        image_processing = cv2.bitwise_or(im, image_processing)
    img_gray = cv2.cvtColor(image_processing, cv2.COLOR_BGR2GRAY)
    _, image_thresholded = cv2.threshold(img_gray, 50, 100, 0)
    contours, _ = cv2.findContours(
        image_thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = [
        cv2.approxPolyDP(cnt, 0.008 * cv2.arcLength(cnt, True), True)
        for cnt in contours
        if cv2.contourArea(cnt) > 1000
    ]
    if verbose:
        print("Number of obstacles found = " + str(len(contours)))
    if overlay:
        helpers.overlay_obstacles(image, contours, orig_contours)
    return (contours, orig_contours)


def find_target(image, overlay=False):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    final_frame_threshed = np.zeros(
        (hsv_img.shape[0], hsv_img.shape[1], 1), dtype=np.uint8
    )
    for HSV_values in hsvs_red:
        frame_threshed = cv2.inRange(hsv_img, HSV_values[0], HSV_values[1])
        frame_threshed = cv2.morphologyEx(
            frame_threshed, cv2.MORPH_CLOSE, kernel=np.ones((9, 9), np.uint8)
        )
        final_frame_threshed = cv2.bitwise_or(frame_threshed, final_frame_threshed)
    contours, _ = cv2.findContours(
        final_frame_threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    contours = [
        cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        for cnt in contours
        if cv2.contourArea(cnt) > 500
    ]
    contours.sort(key=cv2.contourArea, reverse=True)
    if len(contours) < 1:
        return None
    center, _ = cv2.minEnclosingCircle(contours[0])
    center = (int(center[0]), int(center[1]))
    if overlay:
        helpers.overlay_target(image, center)
    return center


def find_robot(image, verbose=False, overlay=False):
    """
    Retrieve the robot in the map from the given image
    Params:
        image: the image to find the robot in
        overlay: (optionnal, default False) overlays the robot contours and coordinates in the image
        verbose: (optionnal, default False) prints info about what is happening
    Returns:
        robot_coordinates (list of 2 int): the x y coordinates of the robot
        angle_degs: the angle of the robot with the (0,1) vector of the map (note that the evolution of the angle is clockwise)
    """
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    frame_threshed = cv2.inRange(hsv_img, BLUE_MIN, BLUE_MAX)
    frame_threshed = cv2.morphologyEx(
        frame_threshed, cv2.MORPH_CLOSE, kernel=np.ones((9, 9), np.uint8)
    )
    contours, _ = cv2.findContours(
        frame_threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    contours = [
        cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
        for cnt in contours
        if cv2.contourArea(cnt) > 100
    ]
    contours.sort(key=cv2.contourArea)
    if len(contours) < 2:
        if verbose:
            print(
                "Robot not found, number of objects:",
                len(contours),
                "number of contours for each object",
                [len(cnt) for cnt in contours],
            )
        return None, None
    coords_tri, _ = cv2.minEnclosingCircle(contours[0])
    coords_circle, _ = cv2.minEnclosingCircle(contours[1])
    distance_center = math.dist(coords_circle, coords_tri)
    if (
        distance_center < MIN_DISTANCE_ROBOT_CENTER
        or distance_center > MAX_DISTANCE_ROBOT_CENTER
    ):
        # return None, None
        pass
    angle_rads = math.atan2(
        coords_circle[1] - coords_tri[1], coords_circle[0] - coords_tri[0]
    )
    angle_degs = math.degrees(angle_rads)
    x = round((coords_tri[0] + coords_circle[0]) / 2)
    y = round((coords_tri[1] + coords_circle[1]) / 2)
    if verbose:
        print(x, y, angle_degs)
    if overlay:
        helpers.overlay_robot_coordinates_and_angle(image, (x, y), angle_degs, contours)
    return [x, y], angle_degs


def process_image(image):
    # image = cv2.bilateralFilter(image, 9, 75, 75)
    image = cv2.GaussianBlur(image, (11, 11), 0, 0)
    return image


class ComputerVisionManager(Thread):
    def __init__(self):
        self.stop_stream = False
        Thread.__init__(self)
        self.crop_params = None
        self.obstacles = None
        self.cap = cv2.VideoCapture(0)
        self.robot_coordinates = None
        self.target = None
        self.stop_loop = False
        self.cropped_image = None
        self.est_coords = None, None

    def get_area_elements(self):
        # get area
        _, frame = self.cap.read()
        cropped_image, self.crop_params = get_area(frame)
        cropped_image = process_image(cropped_image)
        # find obstacles
        self.obstacles, orig_obs = find_obstacles(cropped_image, overlay=True)
        # find robot
        self.robot_coordinates = find_robot(cropped_image, overlay=True)
        # find target
        self.target = find_target(cropped_image, overlay=True)
        return (
            cropped_image,
            self.robot_coordinates,
            self.target,
            self.obstacles,
            orig_obs,
        )

    def get_robot_coordinates(self):
        res = (None, None)
        res = self.robot_coordinates
        return res

    def stop(self):
        self.cap.release()
        self.stop_loop = True

    def run(self):
        while not self.stop_loop:
            if self.crop_params is None:
                continue
            # update robot's position
            _, frame = self.cap.read()
            if frame is None:
                continue
            self.cropped_image = get_area(frame, self.crop_params)
            self.robot_coordinates = find_robot(self.cropped_image)
