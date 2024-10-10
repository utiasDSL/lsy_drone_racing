FROM ros:noetic-ros-base-focal

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV CSW_PYTHON=python3

WORKDIR /home
RUN apt-get update 
RUN apt-get -y install swig build-essential git python-is-python3 python3-pip libpcl-dev libusb-1.0-0-dev
RUN pip install --upgrade pip
RUN apt-get install -y ros-noetic-tf ros-noetic-tf-conversions

# Install crazyswarm
RUN python -m pip install --no-cache-dir pytest numpy PyYAML scipy
RUN git clone --depth 1 https://github.com/USC-ACTLab/crazyswarm.git
WORKDIR /home/crazyswarm
RUN source /opt/ros/noetic/setup.bash && ./build.sh

ENTRYPOINT [ "roslaunch", "/home/crazyswarm/ros_ws/src/crazyswarm/launch/hover_swarm.launch" ]