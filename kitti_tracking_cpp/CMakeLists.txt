cmake_minimum_required(VERSION 3.8)
project(kitti_tracking_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
  include
  src
  ${Eigen3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(cpp_tracker_node src/tracker_node.cpp)
ament_target_dependencies(cpp_tracker_node
  rclcpp
  sensor_msgs
  vision_msgs
  cv_bridge
)
target_link_libraries(cpp_tracker_node
  ${OpenCV_LIBRARIES}
  Eigen3::Eigen
)

install(TARGETS
  cpp_tracker_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
