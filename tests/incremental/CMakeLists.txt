# The incremental reconstruction test (PLAN.md, M2): jpo's two estimators with just-ba's track building,
# triangulation and bundle adjustment. Built only when just-ba's sources are found next to this
# repository (or where JPO_JUST_BA_DIR points); skipped without error otherwise.
set(JPO_JUST_BA_DIR "${CMAKE_SOURCE_DIR}/../just-ba" CACHE PATH "just-ba sources, for the incremental reconstruction test")

if(NOT EXISTS "${JPO_JUST_BA_DIR}/CMakeLists.txt" OR NOT EXISTS "${JPO_JUST_BA_DIR}/api/include/jba/refine.h")
  message(STATUS "incremental_test: just-ba not found at ${JPO_JUST_BA_DIR}; test skipped (set JPO_JUST_BA_DIR)")
  return()
endif()
# just-ba needs CMake 3.24 and C++23; this repository asks for less.
if(CMAKE_VERSION VERSION_LESS 3.24)
  message(STATUS "incremental_test: just-ba needs CMake 3.24 or later; test skipped")
  return()
endif()
if(NOT "cxx_std_23" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
  message(STATUS "incremental_test: just-ba needs a C++23 compiler; test skipped")
  return()
endif()

# just-ba has no installed package configuration, so its sources are built here, out of its tree: the CPU
# libraries only (no tests, no command line, no GPU backend, no bindings). EXCLUDE_FROM_ALL: only what
# incremental_test links is compiled.
set(JBA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(JBA_BUILD_CLI OFF CACHE BOOL "" FORCE)
set(JBA_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(JBA_BUILD_WASM OFF CACHE BOOL "" FORCE)
set(JBA_WITH_CERES OFF CACHE BOOL "" FORCE)
set(JBA_WITH_METAL OFF CACHE BOOL "" FORCE)
set(JBA_WITH_CUDA OFF CACHE BOOL "" FORCE)
set(JBA_WITH_WEBGPU OFF CACHE BOOL "" FORCE)
add_subdirectory("${JPO_JUST_BA_DIR}" "${CMAKE_BINARY_DIR}/just-ba" EXCLUDE_FROM_ALL)

add_executable(incremental_test incremental_test.cpp)
target_include_directories(incremental_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR})
# `jba` carries the solver, the report, the tracks and the triangulation (jba_core, jba_cpu, jba_tracks).
target_link_libraries(incremental_test PRIVATE jpo jba)
set_target_properties(incremental_test PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
add_test(NAME incremental COMMAND incremental_test)
