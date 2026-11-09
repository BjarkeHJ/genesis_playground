cmake_minimum_required(VERSION 3.8)
project(policy_controller)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ONNX RUNTIME (prebuilt in ros2_ws/external/onnxruntime)
set(ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../external/onnxruntime")
get_filename_component(ORT_ROOT "${ORT_ROOT}" ABSOLUTE)
find_library(ORT_LIB onnxruntime PATHS "${ORT_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
add_library(onnxruntime::onnxruntime SHARED IMPORTED)
set_target_properties(
  onnxruntime::onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ORT_LIB}"
  INTERFACE_INCLUDE_DIRECTORIES "${ORT_ROOT}/include"
)
message(STATUS "ONNX Runtime: ${ORT_LIB}")

# ROS2 Node
set(NODE_NAME policy_controller_node)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(Eigen3 REQUIRED)

# Custom msg pkg
find_package(policy_interfaces REQUIRED)

include_directories(
  include/${PROJECT_NAME}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(
  ${NODE_NAME}
  src/controller_node.cpp
  src/policy.cpp
)
  
ament_target_dependencies(
  ${NODE_NAME}
  rclcpp
  policy_interfaces
)
    
target_link_libraries(
  ${NODE_NAME}
  onnxruntime::onnxruntime
  Eigen3::Eigen
)

set_target_properties(
  ${NODE_NAME} PROPERTIES
  BUILD_RPATH "${ORT_ROOT}/lib"
  INSTALL_RPATH "$ORIGIN/.." # bin in lib/onnx_test/
)

install(
  FILES
  ${ORT_ROOT}/lib/libonnxruntime.so.1
  ${ORT_ROOT}/lib/libonnxruntime.so.1.30.1
  DESTINATION lib
)

install(
  TARGETS ${NODE_NAME}
  DESTINATION lib/${PROJECT_NAME}
)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
