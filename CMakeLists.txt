cmake_minimum_required(VERSION 3.9)
project(scan_test)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
find_package(CUDA REQUIRED)

set(INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/include)

set(CPP_SRC
    ${PROJECT_SOURCE_DIR}/src/main.cc
    ${PROJECT_SOURCE_DIR}/src/utils.cpp)

set(CUDA_SRC
    ${PROJECT_SOURCE_DIR}/src/scan.cu)

cuda_add_library(scan STATIC ${INCLUDE_DIRS} ${CUDA_SRC})

include_directories(${INCLUDE_DIRS})
add_executable(scan_test ${CPP_SRC})
target_link_libraries(scan_test scan)
