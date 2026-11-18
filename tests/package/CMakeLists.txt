# A project outside poselib-gpu's tree, written in C, that finds the installed package (run.cmake).
cmake_minimum_required(VERSION 3.16)
# The code is C; CXX is there for the link, as for any static C++ library.
project(jpo_package_test LANGUAGES C CXX)
find_package(jpo 0.1 REQUIRED COMPONENTS cpu)
message(STATUS "jpo ${jpo_VERSION}, backends: ${jpo_BACKENDS}")
add_executable(package_test main.c)
target_link_libraries(package_test PRIVATE jpo::jpo)
if(NOT MSVC)
  target_link_libraries(package_test PRIVATE m)
endif()
