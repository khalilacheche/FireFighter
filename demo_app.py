import LocalNavigation as ln
import MotionControl as mc
import KalmanFilter as kf
import GlobalNavigation as gn
import ComputerVision as cv

import cv2
import threading
import asyncio
import numpy as np
import helpers

from tdmclient import ClientAsync

OBSTACLE_TRESHOLD = 100  # threshold to consider an obstacle
SPEED = 150  # speed of the robot
PERIOD = 0.1  # period of the main loop in seconds
SECURITYD = 50  # security distance

(pos0, robot_angle), goal, obstacles, path = (None, None), None, None, None

global currentNode
currentNode = [0, 0]
init_path = None
cvm = cv.ComputerVisionManager()
cvm.start()
cond = True
while cond:
    (
        area_image,
        (pos0, robot_angle),
        goal,
        obstacles,
        orig_obstacles,
    ) = cvm.get_area_elements()
    # show area, wait for confirmation
    cv2.imshow("FireFighter", area_image)
    key = cv2.waitKey()
    cond = key == ord("r")
    if key == ord("r"):
        continue
    path = gn.find_path(
        pos0,
        obstacles,
        goal,
        (cvm.crop_params[2], cvm.crop_params[3]),
        display_graph=False,
    )
    init_path = path.copy()


async def main(robot_coords, goal, path):
    global currentNode, pos, robot_angle
    client = ClientAsync()
    node = await client.wait_for_node()
    await node.lock()
    (pos0, robot_angle) = robot_coords

    pos = pos0
    nextP = path.pop(0)
    await node.wait_for_variables()  # wait for the variables to be updated
    kfm = kf.KalmanFilterManager(robot_angle, pos0, node)

    while (not mc.checkPoint(pos, goal, SECURITYD)) or len(path) != 0:
        # check if there is an obstacle in front of the robot
        if await ln.seeObs(node, OBSTACLE_TRESHOLD):
            await ln.avoidObs(node)
        else:
            if mc.checkPoint(pos, nextP, SECURITYD):
                nextP = path.pop(0)
                currentNode = nextP
            await mc.moveToPoint(node, pos, robot_angle, nextP, SPEED)

        pos, robot_angle = cvm.get_robot_coordinates()
        pos, robot_angle = kfm.update_position((robot_angle, pos))
        cvm.est_coords = (pos, robot_angle)
    try:
        cvm.stop()
        await node.stop()
    except:
        pass


robot_control_thread = threading.Thread(
    target=lambda: asyncio.run(main((pos0, robot_angle), goal, path))
)
robot_control_thread.start()
img_firetruck = cv2.imread("res/ft.png", cv2.IMREAD_UNCHANGED)
img_fire = cv2.imread("res/fire.png", cv2.IMREAD_UNCHANGED)
cond = True
while cond:
    if cvm.cropped_image is None:
        continue
    image_webcam = cvm.cropped_image
    w = image_webcam.shape[0]
    h = image_webcam.shape[1]
    image_game = np.zeros([w, h, 3], dtype=np.uint8)
    image_game.fill(255)
    # image = #cvm.cropped_image.copy()

    # Overlay goal
    image_game = helpers.overlayPNG(
        image_game,
        img_fire,
        [
            goal[0] - (img_fire.shape[0] // 2),
            goal[1] - (img_fire.shape[1] // 2),
        ],
    )
    # Overlay path
    for point1, point2 in zip(init_path, init_path[1:]):
        cv2.line(
            image_game,
            point1,
            point2,
            [0, 255, 0] if point2 == currentNode else [0, 0, 0],
            2,
        )
    for pp in init_path:
        cv2.circle(image_game, (pp[0], pp[1]), radius=10, color=(0, 0, 0), thickness=-1)
    cv2.circle(
        image_game,
        (currentNode[0], currentNode[1]),
        radius=10,
        color=(255, 0, 255),
        thickness=-1,
    )
    # overlay obstacles
    cv2.drawContours(image_game, orig_obstacles, -1, (0, 0, 0), 3)
    cv2.fillPoly(image_game, pts=orig_obstacles, color=(0, 0, 0))

    robot_pos, angle = cvm.est_coords  # get_robot_coordinates()
    if robot_pos is not None and helpers.overlay_image_in_bounds(
        (w, h), img_firetruck.shape, robot_pos
    ):
        # overlay firetruck
        img_firetruck_rotated = helpers.rotateImage(img_firetruck, -90 - angle)
        image_game = helpers.overlayPNG(
            image_game,
            img_firetruck_rotated,
            [
                robot_pos[0] - (img_firetruck_rotated.shape[0] // 2),
                robot_pos[1] - (img_firetruck_rotated.shape[1] // 2),
            ],
        )
    combined = np.hstack((image_game, image_webcam))
    cv2.imshow("FireFighter", combined)
    key = cv2.waitKey(30)
    cond = key != ord("q")

robot_control_thread.join()
