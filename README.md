# Basics of Mobile Robotic : Project Firefighter
__Group members :__ <br>
- Khalil Haroun Achache (300350) <br>
- Barbara Joanne De Groot (296815) <br>
- Killian Hinard (312106) <br>
- Louis Francois Robert Von Wattenwyl (302808)

## Introduction to our environment :

Oh no ! A fire has started in the middle of the city... Luckily SuperThymio is there to save the day ! As soon as the fire is located, SuperThymio takes the shortest path to reach it and arrives in no time to end the fire. The city is saved...once more... by SuperThymio.

In our project, the city is modelized by a white floor with 3d black obstacles. The white floor eases the job of the computer vision (CV) as patterns are easily recognizable with a white background. We decided to have 3D obstacles so that they could still be recognized by the local avoidance and ensures that even if a new obstacle appears near a permanent obstacle, the robot will also avoid the permanent obstacle.\
SuperThymio is equipped with a blue sign on its top, a triangle and a circle. This enables us to use pattern recognition to find the position as well as orientation of the robot. \
The fire is modelized by a red cube. \
All the different components of environment have been chosen to have high contrast between each other to ease the job of the computer vision. Alongside the live video feedback we also added a clean visualization where SuperThymio is represented by a firetruck and the target is reprensented by a fire to help the imagination.

To have a stable environment, especially regarding the lighting condition, we set the environment in the studio of one of the team members. This way, we had a room where we could run our project without worrying about the outside light or other disturbances.

In this repo, you will find a jupyter notebook ```project_main.ipynb``` containing a detailed description of how the different modules of the project are imlemented, as well as a headless version of the working project.
Running the script ```demo_app.py```, will launch a GUI showing the camera feed of as well as the simulated environment of the robot.

You can find below a demo of the robot in action <br> ![demo](image_rapport/bomr_demo.gif)
