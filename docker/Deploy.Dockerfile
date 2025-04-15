FROM osrf/ros:jazzy-desktop

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_MASTER_URI=http://host.docker.internal:11311

RUN apt-get update && apt-get install -y python3 python-is-python3 python3-venv
# Avoid PEP 668 with venv
RUN python -m venv /home/python 

WORKDIR /home

RUN apt-get -y install build-essential libusb-1.0-0-dev
# Install cflib.
RUN /home/python/bin/pip install cflib

# Copy only pyproject.toml first to leverage Docker cache for dependency installation
# This allows us to avoid reinstalling dependencies if only the source code changes
WORKDIR /home/lsy_drone_racing
COPY pyproject.toml ./
# Install dependencies and cache the build step (only rebuilds when pyproject.toml changes) 
RUN /home/python/bin/pip install build
RUN /home/python/bin/pip install --no-cache-dir .[test,deploy]
# Copy the rest of the application
COPY . .
RUN /home/python/bin/pip install --no-cache-dir -e .[test]

ENTRYPOINT ["/bin/bash", "-c" , "source /opt/ros/jazzy/setup.bash && /home/python/bin/python /home/lsy_drone_racing/scripts/deploy.py"]