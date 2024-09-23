cmake_minimum_required(VERSION 3.10)
project(Project1)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable for imgDisplay (static image display)
add_executable(imgDisplay imgDisplay.cpp filters.cpp)
target_link_libraries(imgDisplay ${OpenCV_LIBS})

# Add the executable for vidDisplay (video processing and face detection)
add_executable(vidDisplay vidDisplay.cpp filters.cpp faceDetect.cpp)
target_link_libraries(vidDisplay ${OpenCV_LIBS})

# Add more output for debugging or verbose mode
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
