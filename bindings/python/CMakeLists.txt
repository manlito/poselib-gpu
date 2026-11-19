# The Python module poselib_gpu (nanobind, stable ABI from Python 3.12), as just-ba's. Configure with the
# interpreter that has nanobind installed, for example:
#   -DJPO_BUILD_PYTHON=ON -DPython_EXECUTABLE=bindings/python/.venv/bin/python
# The package is assembled in <build>/bindings/python/poselib_gpu; the tests run from there. pip and the
# wheel builds go through pyproject.toml (scikit-build-core), which sets SKBUILD and installs the same files.
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT RESULT_VARIABLE _nb)
if(NOT _nb EQUAL 0)
  message(FATAL_ERROR "nanobind not found for ${Python_EXECUTABLE}; install it (bindings/python/README.md) or set Python_EXECUTABLE")
endif()
find_package(nanobind CONFIG REQUIRED)

# Which package this build is (pyproject.toml and packaging/variant.py set it): "cpu" for poselib-gpu,
# "cu12" or "cu13" for the CUDA packages, "source" for a CMake build.
set(JPO_PACKAGE_VARIANT source CACHE STRING "Package variant recorded in poselib_gpu._build_info")
option(JPO_PYTHON_STUBS "Generate the type stubs (_core.pyi) by importing the built module" ON)

set(_pkg ${CMAKE_CURRENT_BINARY_DIR}/poselib_gpu)
nanobind_add_module(_core STABLE_ABI NB_STATIC src/module.cpp)
target_link_libraries(_core PRIVATE jpo)
set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${_pkg})
foreach(_cfg ${CMAKE_CONFIGURATION_TYPES})
  string(TOUPPER ${_cfg} _CFG)
  set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY_${_CFG} ${_pkg})
endforeach()

set(JPO_CUDA_VERSION "")
if(JPO_HAVE_CUDA)
  set(JPO_CUDA_VERSION ${CMAKE_CUDA_COMPILER_VERSION})
endif()
configure_file(build_info.py.in ${_pkg}/_build_info.py @ONLY)

file(GLOB _py CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/poselib_gpu/*.py ${CMAKE_CURRENT_SOURCE_DIR}/poselib_gpu/py.typed)
foreach(_f ${_py})
  get_filename_component(_n ${_f} NAME)
  add_custom_command(TARGET _core POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_f} ${_pkg}/${_n})
endforeach()

if(JPO_PYTHON_STUBS)
  # Imports poselib_gpu._core from the build directory, so the package's own files must be there first (the
  # copies above run after _core is linked).
  nanobind_add_stub(_core_stub MODULE poselib_gpu._core OUTPUT ${_pkg}/_core.pyi
                    PYTHON_PATH ${CMAKE_CURRENT_BINARY_DIR} DEPENDS _core)
endif()

if(SKBUILD)
  # The Python files come from wheel.packages in pyproject.toml; CMake adds what it builds.
  install(TARGETS _core LIBRARY DESTINATION poselib_gpu)
  install(FILES ${_pkg}/_build_info.py DESTINATION poselib_gpu)
  if(JPO_PYTHON_STUBS)
    install(FILES ${_pkg}/_core.pyi DESTINATION poselib_gpu)
  endif()
endif()

if(JPO_BUILD_TESTS)
  add_test(NAME python COMMAND ${Python_EXECUTABLE} -m pytest -q ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  set_tests_properties(python PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")
endif()
