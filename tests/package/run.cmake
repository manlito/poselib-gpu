# ctest's `package` test: install the build into a fresh prefix, configure the C project next to this
# file against it (only CMAKE_PREFIX_PATH points at the prefix), build it and run it.
#   cmake -DJPO_BUILD_DIR=<build> -DJPO_CONFIG=<config> -DJPO_SOURCE_DIR=<this dir> -DJPO_GENERATOR=<gen>
#         -P run.cmake
set(work "${JPO_BUILD_DIR}/package_test")
set(prefix "${work}/prefix")
file(REMOVE_RECURSE "${work}")

function(run)
  execute_process(COMMAND ${ARGN} RESULT_VARIABLE rc)
  if(NOT rc EQUAL 0)
    string(REPLACE ";" " " cmd "${ARGN}")
    message(FATAL_ERROR "failed (${rc}): ${cmd}")
  endif()
endfunction()

set(config_args)
if(JPO_CONFIG)
  set(config_args --config "${JPO_CONFIG}")
endif()
run(${CMAKE_COMMAND} --install "${JPO_BUILD_DIR}" --prefix "${prefix}" ${config_args})
set(build_type)
if(JPO_CONFIG)
  set(build_type "-DCMAKE_BUILD_TYPE=${JPO_CONFIG}")
endif()
run(${CMAKE_COMMAND} -S "${JPO_SOURCE_DIR}" -B "${work}/build" -G "${JPO_GENERATOR}"
    "-DCMAKE_PREFIX_PATH=${prefix}" ${build_type})
run(${CMAKE_COMMAND} --build "${work}/build" ${config_args})
find_program(exe package_test PATHS "${work}/build" "${work}/build/${JPO_CONFIG}" NO_DEFAULT_PATH)
if(NOT exe)
  message(FATAL_ERROR "package_test was not built")
endif()
run("${exe}")
