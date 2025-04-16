FROM python:3.11.12
RUN pip install --upgrade pip
RUN apt update
WORKDIR /home

# Copy only pyproject.toml first to leverage Docker cache for dependency installation
# This allows us to avoid reinstalling dependencies if only the source code changes
WORKDIR /home/lsy_drone_racing
COPY pyproject.toml ./
# Install dependencies and cache the build step (only rebuilds when pyproject.toml changes) 
RUN pip install build
RUN pip install --no-cache-dir .[test,gpu]
# Copy the rest of the application
COPY . .
RUN pip install -e .[test,gpu]

ENTRYPOINT ["python", "/home/lsy_drone_racing/scripts/sim.py", "--gui", "False"]