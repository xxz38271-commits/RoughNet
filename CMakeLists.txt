cmake_minimum_required(VERSION 3.10)
project(ONNX_Image_Prediction)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED PATHS "C:/opencv/build")
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(onnxruntime REQUIRED PATHS "C:/Users/gudwn/onnxruntime_install/lib/cmake/onnxruntime")
include_directories(${onnxruntime_INCLUDE_DIRS})
include_directories("C:/Users/gudwn/onnxruntime_install/include/onnxruntime")
link_directories("C:/Users/gudwn/onnxruntime/build/Windows/Release/Release")

add_executable(ONNX_Image_Prediction module.cpp)

target_link_libraries(ONNX_Image_Prediction ${OpenCV_LIBS} onnxruntime) 