cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(DCGAN)

set(Torch_DIR "libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

include_directories(./include)

add_executable(DCGAN train.cpp)

target_link_libraries(DCGAN "${TORCH_LIBRARIES}")
set_property(TARGET DCGAN PROPERTY CXX_STANDARD 14)