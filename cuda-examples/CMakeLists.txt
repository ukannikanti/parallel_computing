cmake_minimum_required (VERSION 3.25)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.3/bin/nvcc)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)

project (cudaExamples LANGUAGES CUDA CXX)

# Adds build target of with source code files
add_executable(cudaExamples ${CMAKE_SOURCE_DIR}/src/main.cpp ${CMAKE_SOURCE_DIR}/src/timer.cpp ${CMAKE_SOURCE_DIR}/src/ConvKernel.cu) 

target_include_directories(cudaExamples PUBLIC ${CMAKE_SOURCE_DIR}/include)

set_target_properties(cudaExamples PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(cudaExamples ${OpenCV_LIBS})