# This module is used to control the motion of the robot
import math
import helpers

Kp = 1


async def setSpeed(node, speed):
    """
    Set the speed of the robot
    Input : node : the async var of the robot
            speed : the speed of the robot [left wheel, right wheel]
    Output : None
    """
    await node.set_variables(
        {"motor.left.target": [int(speed[0])], "motor.right.target": [int(speed[1])]}
    )

async def moveToPoint(node, pos, orientation, path, speed):
    """
    Move the robot to a point
    Input : node : the async var of the robot
            pos : the position of the robot [x,y]
            orientation : the orientation of the robot (angle with the x axis)
            path : the point [x, y] in the path that the robot need to reach
            speed : the speed of the robot
    Output : None
    """
    # compute the direction vector of the path
    direction = [path[0] - pos[0], path[1] - pos[1]]

    # compute the angle between the direction vector and the x axis
    dirAngle = math.atan2(direction[1], direction[0])
    # compute the angle between the direction vector and the orientation of the robot
    theta = math.degrees(dirAngle) - orientation
    theta, _ = helpers.angle_modulo(theta)

    await setSpeed(node, [speed + Kp * theta, speed - Kp * theta])


def checkPoint(pos, path, securityD):
    """
    Check if the robot is close enough to the point in the path
    Input : pos : the position of the robot [x,y]
            path : the point [x, y] in the path that the robot need to reach
            threshold : the threshold to consider that the robot is close enough to the point
    Output : True if the robot is close enough to the point, False otherwise
    """
    # compute the distance between the robot and the point
    dist = math.dist(pos, path)
    if dist < securityD:
        return True
    return False
