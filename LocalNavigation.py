import MotionControl as mc

scale = 600


# check if there is an obstacle in front of the robot
async def seeObs(node, trehsold):
    """
    Check if there is an obstacle in front of the robot
    Input : node : the async var of the robot
            threshold : the threshold to consider an obstacle
    Output : True if there is an obstacle, False otherwise
    """

    if any([x > trehsold for x in node["prox.horizontal"]]):
        return True
    return False


# Local avoidance function allowing the robot to avoid obstacles
async def avoidObs(node):
    """
    Local avoidance function allowing the robot to avoid obstacles
    Input : node : the async var of the robot
    Output : None
    """

    wL = [40, 20, -20, -20, -40, 30, -10]
    wR = [-40, -20, -20, 20, 40, -10, 30]

    yl, yr = 0, 0

    for i in range(7):
        yl = wL[i] * node["prox.horizontal"][i] + yl
        yr = wR[i] * node["prox.horizontal"][i] + yr

    await mc.setSpeed(node, [yl // scale, yr // scale])
