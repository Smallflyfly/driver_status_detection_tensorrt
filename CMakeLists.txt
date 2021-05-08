cmake_minimum_required(VERSION 3.19)
project(driver_status_detection_tensorrt)

set(CMAKE_CXX_STANDARD 14)

add_executable(driver_status_detection_tensorrt
        main.cpp)
find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(driver_status_detection_tensorrt ${CUDA_LIBRARIES})
target_link_libraries(driver_status_detection_tensorrt ${OpenCV_LIBS})

# win10
include_directories(C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA/v10.2\\include)
include_directories(D:\\TensorRT-7.0.0.11\\include)
include_directories(D:\\TensorRT-7.0.0.11\\samples\\common\\windows)
include_directories(D:\\TensorRT-7.0.0.11\\samples\\common)
include_directories(D:\\TensorRT-7.0.0.11\\bin)
include_directories(C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.2\\lib\\x64)

link_directories(C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.2\\lib\\x64)
link_directories(D:\\TensorRT-7.0.0.11\\lib)

add_definitions(-O2 -pthread)