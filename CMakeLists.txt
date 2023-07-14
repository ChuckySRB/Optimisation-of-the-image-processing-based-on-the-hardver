cmake_minimum_required(VERSION 3.20)
project(Projekat)

set(CMAKE_CXX_STANDARD 23)
set(ENV{OPENCV_DIR} "D:\\Caslav\\Programiranje\\C++\\OpenCV\\opencv\\build")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Projekat main.cpp source/Image_Processor.cpp source/Image_Processor_Not_Optimised.cpp source/Image_Processor_Optimised.cpp)

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)

target_link_libraries(Projekat ${OpenCV_LIBS})
