from pickle import TRUE
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

MAP_BORDER = 100

# functions to check if two segments intersect


def onSegment(px, py, qx, qy, rx, ry):
    """
    Given three collinear points p, q, r, the function checks if
    point q lies on line segment 'pr'
    Input : coordinates of p, q and r points
    Output : True if q on segment pr and False otherwise
    """
    if (
        (qx <= max(px, rx))
        and (qx >= min(px, rx))
        and (qy <= max(py, ry))
        and (qy >= min(py, ry))
    ):
        return True
    return False


def orientation(px, py, qx, qy, rx, ry):
    """
    to find the orientation of an ordered triplet (p,q,r)
    function returns the following values:
    0 : Collinear points
    1 : Clockwise points
    2 : Counterclockwise

    See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
    for details of below formula.
    """

    val = (float(qy - py) * (rx - qx)) - (float(qx - px) * (ry - qy))
    if val > 0:

        # Clockwise orientation
        return 1
    elif val < 0:

        # Counterclockwise orientation
        return 2
    else:

        # Collinear orientation
        return 0


def doIntersect(p1x, p1y, q1x, q1y, p2x, p2y, q2x, q2y):
    """
    Main function to check for intersection
    Input : coordinates of points defining segments p1q1 and p2q2
    Output : True if the line segment 'p1q1' and 'p2q2' intersect False otherwise
    """

    # Find the 4 orientations required for
    # the general and special cases
    o1 = orientation(p1x, p1y, q1x, q1y, p2x, p2y)
    o2 = orientation(p1x, p1y, q1x, q1y, q2x, q2y)
    o3 = orientation(p2x, p2y, q2x, q2y, p1x, p1y)
    o4 = orientation(p2x, p2y, q2x, q2y, q1x, q1y)

    # General case
    if (o1 != o2) and (o3 != o4):
        return True

    # Special Cases

    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if (o1 == 0) and onSegment(p1x, p1y, p2x, p2y, q1x, q1y):
        return True

    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if (o2 == 0) and onSegment(p1x, p1y, q2x, q2y, q1x, q1y):
        return True

    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if (o3 == 0) and onSegment(p2x, p2y, p1x, p1y, q2x, q2y):
        return True

    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if (o4 == 0) and onSegment(p2x, p2y, q1x, q1y, q2x, q2y):
        return True

    # If none of the cases
    return False


def distance(px, py, qx, qy):
    """
    Input : coordinates of the two points
    Output : euclidian distance between two points
    """
    return ((px - qx) ** 2 + (py - qy) ** 2) ** 0.5


def same_point(px, py, qx, qy):
    """
    checks if p and q are the same point
    Input : coordinates of p and q
    Output : True if p=q and False otherwise
    """
    if (px == qx) and (py == qy):
        return True
    else:
        return False


def same_segment(p1x, p1y, q1x, q1y, p2x, p2y, q2x, q2y):
    """
    checks if p1-q1 and p2-q2 are the same segment
    Input : coordinates extremeties of both segments
    Output : True if p1-q1=p2-q2 and False otherwise
    """
    if (same_point(p1x, p1y, p2x, p2y) and same_point(q1x, q1y, q2x, q2y)) or (
        same_point(p1x, p1y, q2x, q2y) and same_point(q1x, q1y, p2x, p2y)
    ):
        return True
    else:
        return False


def point_on_border(px, py, dim):
    """
    checks if point p is on the border of the map
    Input : coordinates p=(px,py)
    Output : True if on the map border and False otherwise
    """
    if (
        px < MAP_BORDER
        or py < MAP_BORDER
        or py > dim[0] - MAP_BORDER
        or px > dim[1] - MAP_BORDER
    ):
        return True
    else:
        return False


def segment_on_map_border(px, py, qx, qy, dim):
    """
    checks if segment p-q on the border of the map
    Input : coordinates of the extremeties of the segment
    Output : True if segment on the map border and False otherwise
    """
    if point_on_border(px, py, dim) and point_on_border(qx, qy, dim):
        return True
    else:
        return False


def formatting_data(obs_coord):
    """
    Formats what is received by the computer vision    
    Input : list of list of list of np array with the coordinates of the corners of the obstacles 
    Output : list of list of list with the coordinates of the corners of the obstacles
    """
    m = len(obs_coord)
    n = 0
    obs_coord1 = [[0 for x in range(n)] for x in range(m)]
    for obstacle in range(len(obs_coord)):
        for corner in range(len(obs_coord[obstacle])):
            x = obs_coord[obstacle][corner][0][0]
            y = obs_coord[obstacle][corner][0][1]
            obs_coord1[obstacle].append([x, y])

    return obs_coord1


def find_path(start, obs_coord, target, dim, display_graph=True):
    """
    Input : coordinates of start point, corners of obstacles and target point
    Output : shortest path, as a list of point to go through
    """
    obs_coord = formatting_data(obs_coord)
    # initializing node graph
    G = nx.Graph()

    # adding the start and target node
    G.add_node("S", pos=(start[0], start[1]))
    G.add_node("T", pos=(target[0], target[1]))

    # adding all the nodes from the obstacle's corners
    for obstacle in range(len(obs_coord)):
        for corner in range(len(obs_coord[obstacle])):
            G.add_node(
                str(obstacle) + str(corner),
                pos=(obs_coord[obstacle][corner][0], obs_coord[obstacle][corner][1]),
            )

    # first checking for connection to the starting point
    for obstacle in range(len(obs_coord)):
        for corner in range(len(obs_coord[obstacle])):

            visible = True

            for i in range(len(obs_coord)):
                for j in range(len(obs_coord[i]) - 1):
                    if (
                        doIntersect(
                            start[0],
                            start[1],
                            obs_coord[obstacle][corner][0],
                            obs_coord[obstacle][corner][1],
                            obs_coord[i][j][0],
                            obs_coord[i][j][1],
                            obs_coord[i][j + 1][0],
                            obs_coord[i][j + 1][1],
                        )
                        and not (
                            same_point(
                                obs_coord[obstacle][corner][0],
                                obs_coord[obstacle][corner][1],
                                obs_coord[i][j][0],
                                obs_coord[i][j][1],
                            )
                        )
                        and not (
                            same_point(
                                obs_coord[obstacle][corner][0],
                                obs_coord[obstacle][corner][1],
                                obs_coord[i][j + 1][0],
                                obs_coord[i][j + 1][1],
                            )
                        )
                    ):
                        visible = False

                if (
                    doIntersect(
                        start[0],
                        start[1],
                        obs_coord[obstacle][corner][0],
                        obs_coord[obstacle][corner][1],
                        obs_coord[i][-1][0],
                        obs_coord[i][-1][1],
                        obs_coord[i][0][0],
                        obs_coord[i][0][1],
                    )
                    and not (
                        same_point(
                            obs_coord[obstacle][corner][0],
                            obs_coord[obstacle][corner][1],
                            obs_coord[i][-1][0],
                            obs_coord[i][-1][1],
                        )
                    )
                    and not (
                        same_point(
                            obs_coord[obstacle][corner][0],
                            obs_coord[obstacle][corner][1],
                            obs_coord[i][0][0],
                            obs_coord[i][0][1],
                        )
                    )
                ):
                    visible = False

            if visible:
                G.add_edge(
                    "S",
                    str(obstacle) + str(corner),
                    weight=distance(
                        start[0],
                        start[1],
                        obs_coord[obstacle][corner][0],
                        obs_coord[obstacle][corner][1],
                    ),
                )

    # then connecting corners of each obstacle
    for obstacle in range(len(obs_coord)):
        for corner in range(len(obs_coord[obstacle])):
            if corner == (len(obs_coord[obstacle]) - 1):
                G.add_edge(
                    str(obstacle) + str(0),
                    str(obstacle) + str(corner),
                    weight=distance(
                        obs_coord[obstacle][0][0],
                        obs_coord[obstacle][0][1],
                        obs_coord[obstacle][corner][0],
                        obs_coord[obstacle][corner][1],
                    ),
                )
                if segment_on_map_border(
                    obs_coord[obstacle][0][0],
                    obs_coord[obstacle][0][1],
                    obs_coord[obstacle][corner][0],
                    obs_coord[obstacle][corner][1],
                    dim,
                ):
                    G.remove_edge(str(obstacle) + str(0), str(obstacle) + str(corner))
                break

            G.add_edge(
                str(obstacle) + str(corner),
                str(obstacle) + str(corner + 1),
                weight=distance(
                    obs_coord[obstacle][corner][0],
                    obs_coord[obstacle][corner][1],
                    obs_coord[obstacle][corner + 1][0],
                    obs_coord[obstacle][corner + 1][1],
                ),
            )
            if segment_on_map_border(
                obs_coord[obstacle][corner][0],
                obs_coord[obstacle][corner][1],
                obs_coord[obstacle][corner + 1][0],
                obs_coord[obstacle][corner + 1][1],
                dim,
            ):
                G.remove_edge(
                    str(obstacle) + str(corner), str(obstacle) + str(corner + 1)
                )

    # connecting corner of obstacles together
    for obstacle1 in range(len(obs_coord)):
        for corner1 in range(len(obs_coord[obstacle1])):

            for obstacle2 in range(len(obs_coord)):
                if obstacle1 == obstacle2:
                    break
                for corner2 in range(len(obs_coord[obstacle2])):

                    visible = True

                    for i in range(len(obs_coord)):
                        for j in range(len(obs_coord[i]) - 1):
                            if (
                                doIntersect(
                                    obs_coord[obstacle1][corner1][0],
                                    obs_coord[obstacle1][corner1][1],
                                    obs_coord[obstacle2][corner2][0],
                                    obs_coord[obstacle2][corner2][1],
                                    obs_coord[i][j][0],
                                    obs_coord[i][j][1],
                                    obs_coord[i][j + 1][0],
                                    obs_coord[i][j + 1][1],
                                )
                                and not (
                                    same_point(
                                        obs_coord[obstacle1][corner1][0],
                                        obs_coord[obstacle1][corner1][1],
                                        obs_coord[i][j][0],
                                        obs_coord[i][j][1],
                                    )
                                )
                                and not (
                                    same_point(
                                        obs_coord[obstacle1][corner1][0],
                                        obs_coord[obstacle1][corner1][1],
                                        obs_coord[i][j + 1][0],
                                        obs_coord[i][j + 1][1],
                                    )
                                )
                                and not (
                                    same_point(
                                        obs_coord[obstacle2][corner2][0],
                                        obs_coord[obstacle2][corner2][1],
                                        obs_coord[i][j][0],
                                        obs_coord[i][j][1],
                                    )
                                )
                                and not (
                                    same_point(
                                        obs_coord[obstacle2][corner2][0],
                                        obs_coord[obstacle2][corner2][1],
                                        obs_coord[i][j + 1][0],
                                        obs_coord[i][j + 1][1],
                                    )
                                )
                            ):
                                visible = False

                        if (
                            doIntersect(
                                obs_coord[obstacle1][corner1][0],
                                obs_coord[obstacle1][corner1][1],
                                obs_coord[obstacle2][corner2][0],
                                obs_coord[obstacle2][corner2][1],
                                obs_coord[i][-1][0],
                                obs_coord[i][-1][1],
                                obs_coord[i][0][0],
                                obs_coord[i][0][1],
                            )
                            and not (
                                same_point(
                                    obs_coord[obstacle1][corner1][0],
                                    obs_coord[obstacle1][corner1][1],
                                    obs_coord[i][-1][0],
                                    obs_coord[i][-1][1],
                                )
                            )
                            and not (
                                same_point(
                                    obs_coord[obstacle1][corner1][0],
                                    obs_coord[obstacle1][corner1][1],
                                    obs_coord[i][0][0],
                                    obs_coord[i][0][1],
                                )
                            )
                            and not (
                                same_point(
                                    obs_coord[obstacle2][corner2][0],
                                    obs_coord[obstacle2][corner2][1],
                                    obs_coord[i][-1][0],
                                    obs_coord[i][-1][1],
                                )
                            )
                            and not (
                                same_point(
                                    obs_coord[obstacle2][corner2][0],
                                    obs_coord[obstacle2][corner2][1],
                                    obs_coord[i][0][0],
                                    obs_coord[i][0][1],
                                )
                            )
                        ):
                            visible = False

                    if visible:
                        G.add_edge(
                            str(obstacle1) + str(corner1),
                            str(obstacle2) + str(corner2),
                            weight=distance(
                                obs_coord[obstacle1][corner1][0],
                                obs_coord[obstacle1][corner1][1],
                                obs_coord[obstacle2][corner2][0],
                                obs_coord[obstacle2][corner2][1],
                            ),
                        )

    # lastly checking for connection to the target point
    for obstacle in range(len(obs_coord)):
        for corner in range(len(obs_coord[obstacle])):

            visible = True

            for i in range(len(obs_coord)):
                for j in range(len(obs_coord[i]) - 1):
                    if (
                        doIntersect(
                            target[0],
                            target[1],
                            obs_coord[obstacle][corner][0],
                            obs_coord[obstacle][corner][1],
                            obs_coord[i][j][0],
                            obs_coord[i][j][1],
                            obs_coord[i][j + 1][0],
                            obs_coord[i][j + 1][1],
                        )
                        and not (
                            same_point(
                                obs_coord[obstacle][corner][0],
                                obs_coord[obstacle][corner][1],
                                obs_coord[i][j][0],
                                obs_coord[i][j][1],
                            )
                        )
                        and not (
                            same_point(
                                obs_coord[obstacle][corner][0],
                                obs_coord[obstacle][corner][1],
                                obs_coord[i][j + 1][0],
                                obs_coord[i][j + 1][1],
                            )
                        )
                    ):
                        visible = False

                if (
                    doIntersect(
                        target[0],
                        target[1],
                        obs_coord[obstacle][corner][0],
                        obs_coord[obstacle][corner][1],
                        obs_coord[i][-1][0],
                        obs_coord[i][-1][1],
                        obs_coord[i][0][0],
                        obs_coord[i][0][1],
                    )
                    and not (
                        same_point(
                            obs_coord[obstacle][corner][0],
                            obs_coord[obstacle][corner][1],
                            obs_coord[i][-1][0],
                            obs_coord[i][-1][1],
                        )
                    )
                    and not (
                        same_point(
                            obs_coord[obstacle][corner][0],
                            obs_coord[obstacle][corner][1],
                            obs_coord[i][0][0],
                            obs_coord[i][0][1],
                        )
                    )
                ):
                    visible = False

            if visible:
                G.add_edge(
                    "T",
                    str(obstacle) + str(corner),
                    weight=distance(
                        target[0],
                        target[1],
                        obs_coord[obstacle][corner][0],
                        obs_coord[obstacle][corner][1],
                    ),
                )
    # drawings debug
    if display_graph:
        pos = nx.get_node_attributes(G, "pos")
        plt.figure(2, figsize=(12, 12))
        nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
        # edge_labels = nx.get_edge_attributes(G, "weight")
        # nx.draw_networkx_edge_labels(G, pos, edge_labels)
        nx.draw(G, pos)
        plt.show()
    # end debug

    path = nx.shortest_path(
        G, source="S", target="T", weight="weight"
    )  # returns list of node's name we want coordinates
    if display_graph:
        print(path)
    pos = nx.get_node_attributes(G, "pos")
    # converting path to list of coordinates
    for i in range(len(path)):
        path[i] = pos[str(path[i])]

    return path
