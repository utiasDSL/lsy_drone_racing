cmake_minimum_required(VERSION 3.8)
project(lsy_race_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/HostReady.msg"
  "msg/RaceStart.msg"
  "msg/ClientState.msg"
  "srv/CalibrateClock.srv"
)

ament_package()
