FROM python:3.8.10
RUN pip install --upgrade pip
RUN apt update
WORKDIR /home

# Install the drone firmare emulator
RUN apt install swig build-essential -y
RUN apt install git -y
RUN git clone https://github.com/utiasDSL/pycffirmware.git
WORKDIR /home/pycffirmware
RUN git submodule update --init --recursive
RUN pip install "numpy<2"
RUN ./wrapper/build_linux.sh

# Install the drone racing package
WORKDIR /home/lsy_drone_racing
COPY . /home/lsy_drone_racing
# Cache the dependencies to speed up build time. TODO: Currently not working, investigate why it's not caching
RUN pip install -r docker/requirements.txt
RUN pip install -e .[test,rl]
# ENTRYPOINT ["tail", "-f", "/dev/null"]
ENTRYPOINT ["python", "/home/lsy_drone_racing/scripts/sim.py", "--gui", "False"]