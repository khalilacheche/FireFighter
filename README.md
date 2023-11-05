# Story
Imagine a scenario where a city is in peril due to a raging fire. In this moment of crisis, SuperThymio, the robotic hero, emerges to save the day. As soon as the fire's location is determined, SuperThymio swiftly calculates the shortest path to reach the blaze and arrives in the nick of time to extinguish it. The city is saved, once again, thanks to the heroic actions of SuperThymio.

Our project creates a virtual representation of this city, with a white floor and 3D black obstacles. The white floor offers a clean background for computer vision, making patterns easily recognizable. The 3D obstacles provide challenges for local avoidance and ensure that SuperThymio can navigate around permanent obstacles, even if new obstacles appear nearby. SuperThymio is equipped with a blue sign on its top, featuring a triangle and a circle. This allows for pattern recognition to determine the robot's position and orientation. The fire is represented by a red cube.

Each element of the environment has been carefully chosen to ensure high contrast for computer vision. In addition to live video feedback, we have provided a clean visualization where SuperThymio is symbolized by a firetruck, and the target is represented by a fire, aiding in visualization.

To maintain a stable environment, especially concerning lighting conditions, we set up the environment in the studio of one of our team members. This allowed us to run our project without concerns about external lighting or disturbances.

# What
Our project focuses on simulating a critical firefighting scenario using SuperThymio, a mobile robot. The project aims to demonstrate the robot's ability to navigate the environment, locate and approach a simulated fire, and effectively extinguish it.

# How
In this repository, you will find a Jupyter notebook (project_main.ipynb) that provides a detailed description of how the different modules of the project are implemented. It explains how SuperThymio navigates, uses computer vision, and extinguishes the fire. Additionally, there's a headless version of the working project.

We have also included a script (demo_app.py) that launches a graphical user interface (GUI). This GUI displays the camera feed and a simulated environment of the robot in action.

# Challenges
Our project involved several challenges, including creating a stable and realistic environment for simulation, implementing computer vision for navigation and fire detection, and ensuring the robot can effectively navigate around obstacles. Additionally, the lighting conditions and the need for high contrast between elements were crucial considerations in designing the environment.

# Result
Our project demonstrates the successful simulation of a firefighting scenario using the SuperThymio robot. By modeling the city and the fire, we were able to showcase the robot's ability to navigate, detect the fire, and take action to extinguish it. The detailed Jupyter notebook and the headless version of the project provide insights into the implementation and operation of SuperThymio in a critical rescue scenario.