cmake_minimum_required(VERSION 3.16...3.31)
project(poselib_gpu VERSION 0.1.0 LANGUAGES CXX)

option(JPO_BUILD_CLI "Build the jpo command line tool" ON)
option(JPO_BUILD_TESTS "Build tests" ON)
option(JPO_BUILD_WASM "Emscripten: build the WebAssembly module, JS wrapper and test page (web/)" ON)
option(JPO_BUILD_PYTHON "Build the Python module poselib_gpu (needs nanobind; bindings/python/README.md)" OFF)
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  set(jpo_top_level ON)
else()
  set(jpo_top_level OFF)
endif()
option(JPO_INSTALL "Install rules: the library, its header, the command line tool and the CMake package" ${jpo_top_level})
set(JPO_CUDA "AUTO" CACHE STRING "CUDA backend: AUTO (build it when a CUDA compiler is found), ON or OFF")
set_property(CACHE JPO_CUDA PROPERTY STRINGS AUTO ON OFF)
set(JPO_METAL "AUTO" CACHE STRING "Metal backend (Apple platforms): AUTO, ON or OFF")
set_property(CACHE JPO_METAL PROPERTY STRINGS AUTO ON OFF)
set(JPO_WEBGPU "AUTO" CACHE STRING "WebGPU backend: AUTO (build it when Dawn, or under Emscripten emdawnwebgpu, is found), ON or OFF")
set_property(CACHE JPO_WEBGPU PROPERTY STRINGS AUTO ON OFF)
set(JPO_REFERENCE "AUTO" CACHE STRING "PoseLib reference tool and parity tests: AUTO (when third_party/poselib and eigen exist), ON or OFF")
set_property(CACHE JPO_REFERENCE PROPERTY STRINGS AUTO ON OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(EMSCRIPTEN)
  # Native wasm exceptions, as the sibling libraries: one app links them all into one module.
  add_compile_options(-fwasm-exceptions)
  add_link_options(-fwasm-exceptions)
endif()

# CUDA backend: needs only the CUDA toolkit (nvcc and its runtime library).
if(NOT JPO_CUDA STREQUAL "OFF" AND NOT EMSCRIPTEN)
  if(NOT CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
    find_program(JPO_NVCC nvcc HINTS ENV CUDA_PATH PATHS /opt/cuda /usr/local/cuda PATH_SUFFIXES bin)
    if(JPO_NVCC)
      set(CMAKE_CUDA_COMPILER "${JPO_NVCC}")
    endif()
  endif()
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(JPO_HAVE_CUDA ON)
  elseif(JPO_CUDA STREQUAL "ON")
    message(FATAL_ERROR "JPO_CUDA=ON but no CUDA compiler was found (set CUDACXX or CMAKE_CUDA_COMPILER)")
  endif()
endif()

# Metal backend: Metal.framework and Foundation. Shaders are compiled at run time.
if(NOT JPO_METAL STREQUAL "OFF")
  if(APPLE)
    find_library(JPO_METAL_FRAMEWORK Metal)
    find_library(JPO_FOUNDATION_FRAMEWORK Foundation)
  endif()
  if(JPO_METAL_FRAMEWORK AND JPO_FOUNDATION_FRAMEWORK)
    enable_language(OBJCXX)
    set(JPO_HAVE_METAL ON)
  elseif(JPO_METAL STREQUAL "ON")
    message(FATAL_ERROR "JPO_METAL=ON but Metal.framework was not found (Apple platforms only)")
  endif()
endif()

# WebGPU backend: natively against Dawn's prebuilt release, under Emscripten against emdawnwebgpu from
# the same release (tools/fetch_webgpu.sh). Kernels are WGSL, compiled at run time.
if(NOT JPO_WEBGPU STREQUAL "OFF")
  if(EMSCRIPTEN)
    set(JPO_EMDAWNWEBGPU_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/webgpu/emdawnwebgpu/emdawnwebgpu_pkg"
        CACHE PATH "The unpacked emdawnwebgpu_pkg of a Dawn release")
    if(EXISTS "${JPO_EMDAWNWEBGPU_DIR}/emdawnwebgpu.port.py")
      set(JPO_HAVE_WEBGPU ON)
    endif()
  else()
    file(GLOB jpo_dawn_default "${CMAKE_CURRENT_SOURCE_DIR}/third_party/webgpu/dawn/Dawn-*")
    set(JPO_DAWN_DIR "${jpo_dawn_default}" CACHE PATH "The unpacked Dawn release (with lib/cmake/Dawn)")
    find_package(Dawn CONFIG QUIET PATHS "${JPO_DAWN_DIR}/lib/cmake/Dawn" "${JPO_DAWN_DIR}/lib64/cmake/Dawn"
                 NO_DEFAULT_PATH)
    if(Dawn_FOUND)
      set(JPO_HAVE_WEBGPU ON)
    endif()
  endif()
  if(NOT JPO_HAVE_WEBGPU AND JPO_WEBGPU STREQUAL "ON")
    message(FATAL_ERROR "JPO_WEBGPU=ON but neither Dawn (JPO_DAWN_DIR) nor, under Emscripten, emdawnwebgpu "
                        "(JPO_EMDAWNWEBGPU_DIR) was found")
  endif()
endif()

include(GNUInstallDirs)
add_library(jpo
  src/core/api.cpp
  src/core/auto_rule.cpp
  src/core/estimator.cpp
  src/core/geometry.cpp
  src/backend/cpu/cpu_backend.cpp)
add_library(jpo::jpo ALIAS jpo)
target_include_directories(jpo PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
                               PRIVATE src)
# The header has no export macros: a Windows DLL exports every symbol (the API is all jpo_ functions).
set_target_properties(jpo PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR}
                                     WINDOWS_EXPORT_ALL_SYMBOLS ON)
if(NOT EMSCRIPTEN)
  find_package(Threads REQUIRED)
  target_link_libraries(jpo PRIVATE Threads::Threads)
endif()
if(EMSCRIPTEN)
  # Built with native wasm exceptions (above): the module that links jpo must use them too.
  target_link_options(jpo INTERFACE -fwasm-exceptions)
endif()
if(NOT MSVC)
  # Fused multiply-add is left to the compiler (measured 15-25 % faster on Apple silicon). Backends
  # therefore agree on the best hypothesis except at near ties, not bit for bit; tests compare that way.
  target_compile_options(jpo PRIVATE $<$<COMPILE_LANGUAGE:CXX,OBJCXX>:-Wall;-Wextra>)
endif()

# One kernel source for CPU, CUDA and Metal: src/kernels/kernels.inc.
set(jpo_kernels "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/kernels.inc")
set_property(SOURCE src/backend/cpu/cpu_backend.cpp APPEND PROPERTY OBJECT_DEPENDS "${jpo_kernels}")

if(JPO_HAVE_CUDA)
  target_sources(jpo PRIVATE src/backend/cuda/cuda_backend.cu)
  set_property(SOURCE src/backend/cuda/cuda_backend.cu APPEND PROPERTY OBJECT_DEPENDS "${jpo_kernels}")
  target_compile_definitions(jpo PRIVATE JPO_HAVE_CUDA)
  # The runtime is linked by name rather than left to CMake's CUDA language support, so that a project
  # that finds the installed package links it without enabling CUDA itself.
  find_package(CUDAToolkit REQUIRED)
  set_target_properties(jpo PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_RUNTIME_LIBRARY None)
  target_link_libraries(jpo PRIVATE CUDA::cudart_static)
endif()

if(JPO_HAVE_METAL)
  # The Metal source is the prelude followed by the shared kernels and the entry points.
  set(metal_cpp "${CMAKE_CURRENT_BINARY_DIR}/metal_kernels.cpp")
  set(metal_parts "${CMAKE_CURRENT_SOURCE_DIR}/src/backend/metal/prelude.metal" "${jpo_kernels}"
                  "${CMAKE_CURRENT_SOURCE_DIR}/src/backend/metal/entry.metal")
  string(REPLACE ";" "|" metal_parts_arg "${metal_parts}")
  add_custom_command(
    OUTPUT "${metal_cpp}"
    COMMAND ${CMAKE_COMMAND} "-DIN=${metal_parts_arg}" "-DOUT=${metal_cpp}" -DNAME=kJpoMetalSource
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_text.cmake"
    DEPENDS ${metal_parts} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_text.cmake"
    VERBATIM)
  target_sources(jpo PRIVATE src/backend/metal/metal_backend.mm "${metal_cpp}")
  target_compile_definitions(jpo PRIVATE JPO_HAVE_METAL)
  target_compile_options(jpo PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  # By name, not by the SDK path find_library returned: the installed package must not carry that path.
  target_link_libraries(jpo PRIVATE "-framework Metal" "-framework Foundation")
endif()

if(JPO_HAVE_WEBGPU)
  set(wgsl_src "${CMAKE_CURRENT_SOURCE_DIR}/src/backend/webgpu/kernels.wgsl")
  set(wgsl_cpp "${CMAKE_CURRENT_BINARY_DIR}/webgpu_kernels.cpp")
  add_custom_command(
    OUTPUT "${wgsl_cpp}"
    COMMAND ${CMAKE_COMMAND} "-DIN=${wgsl_src}" "-DOUT=${wgsl_cpp}" -DNAME=kJpoWgslSource
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_text.cmake"
    DEPENDS "${wgsl_src}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_text.cmake"
    VERBATIM)
  target_sources(jpo PRIVATE src/backend/webgpu/webgpu_backend.cpp "${wgsl_cpp}")
  target_compile_definitions(jpo PRIVATE JPO_HAVE_WEBGPU)
  if(EMSCRIPTEN)
    # PUBLIC: whatever links jpo into a module needs the port's JavaScript library too.
    # The installed package adds the same options in jpoConfig.cmake, from a directory the consumer can
    # change (JPO_EMDAWNWEBGPU_DIR), instead of this machine's path.
    target_compile_options(jpo PUBLIC "$<BUILD_INTERFACE:--use-port=${JPO_EMDAWNWEBGPU_DIR}/emdawnwebgpu.port.py>")
    target_link_options(jpo PUBLIC "$<BUILD_INTERFACE:--use-port=${JPO_EMDAWNWEBGPU_DIR}/emdawnwebgpu.port.py>")
  else()
    target_link_libraries(jpo PUBLIC dawn::webgpu_dawn)
  endif()
endif()

set(jpo_backends cpu)
foreach(b CUDA METAL WEBGPU)
  if(JPO_HAVE_${b})
    string(TOLOWER ${b} name)
    list(APPEND jpo_backends ${name})
  endif()
endforeach()
string(REPLACE ";" ", " jpo_backends_text "${jpo_backends}")
message(STATUS "jpo backends: ${jpo_backends_text}")

# The CMake package: find_package(jpo [COMPONENTS cuda metal webgpu]) and target jpo::jpo, from an
# install prefix or from this build tree. A component is a backend this build has.
if(JPO_INSTALL)
  include(CMakePackageConfigHelpers)
  set(jpo_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/jpo")
  install(TARGETS jpo EXPORT jpoTargets
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(FILES include/jpo/jpo.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/jpo)
  install(FILES LICENSE.PoseLib DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/jpo)
  install(EXPORT jpoTargets NAMESPACE jpo:: DESTINATION "${jpo_cmake_dir}")
  export(EXPORT jpoTargets NAMESPACE jpo:: FILE "${CMAKE_CURRENT_BINARY_DIR}/jpoTargets.cmake")

  # What jpoConfig.cmake must find again for the consumer.
  get_target_property(jpo_type jpo TYPE)
  set(JPO_CONFIG_STATIC OFF)
  if(jpo_type STREQUAL "STATIC_LIBRARY")
    set(JPO_CONFIG_STATIC ON)
  endif()
  set(JPO_CONFIG_THREADS OFF)
  if(jpo_type STREQUAL "STATIC_LIBRARY" AND NOT EMSCRIPTEN)
    set(JPO_CONFIG_THREADS ON)
  endif()
  set(JPO_CONFIG_CUDA OFF)
  if(JPO_HAVE_CUDA AND jpo_type STREQUAL "STATIC_LIBRARY")
    set(JPO_CONFIG_CUDA ON)
  endif()
  set(JPO_CONFIG_DAWN OFF)
  if(JPO_HAVE_WEBGPU AND NOT EMSCRIPTEN)
    set(JPO_CONFIG_DAWN ON)
    set(JPO_CONFIG_THREADS ON)  # Dawn's own package names Threads::Threads without finding it
    set(JPO_CONFIG_DAWN_DIR "${Dawn_DIR}")
  endif()
  set(JPO_CONFIG_EMDAWNWEBGPU OFF)
  if(JPO_HAVE_WEBGPU AND EMSCRIPTEN)
    set(JPO_CONFIG_EMDAWNWEBGPU ON)
  endif()
  set(JPO_CONFIG_BACKENDS "${jpo_backends}")
  foreach(location install build)
    if(location STREQUAL "install")
      set(out "${CMAKE_CURRENT_BINARY_DIR}/package/jpoConfig.cmake")
      set(dest "${jpo_cmake_dir}")
    else()
      set(out "${CMAKE_CURRENT_BINARY_DIR}/jpoConfig.cmake")
      set(dest "${CMAKE_CURRENT_BINARY_DIR}")
    endif()
    configure_package_config_file(cmake/jpoConfig.cmake.in "${out}" INSTALL_DESTINATION "${dest}")
  endforeach()
  write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/jpoConfigVersion.cmake"
                                   COMPATIBILITY SameMajorVersion)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/package/jpoConfig.cmake"
                "${CMAKE_CURRENT_BINARY_DIR}/jpoConfigVersion.cmake"
          DESTINATION "${jpo_cmake_dir}")
endif()

# The WebAssembly module (jpo_wasm.js + .wasm) with web/jpo.js and the test page, side by side in
# <build>/web, ready to serve. Single-threaded; waits for the GPU through JSPI.
if(EMSCRIPTEN AND JPO_BUILD_WASM)
  add_executable(jpo_wasm web/jpo_wasm.cpp)
  target_include_directories(jpo_wasm PRIVATE src ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(jpo_wasm PRIVATE jpo)
  set_target_properties(jpo_wasm PROPERTIES SUFFIX ".js" RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/web")
  target_link_options(jpo_wasm PRIVATE
    -sJSPI
    "-sJSPI_EXPORTS=jpow_create,jpow_relative,jpow_absolute"
    "-sEXPORTED_FUNCTIONS=_malloc,_free,_jpow_backend_available,_jpow_create,_jpow_destroy,_jpow_relative,_jpow_absolute,_jpow_synth,_jpow_error,_jpow_set_route,_jpow_last_stats"
    "-sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPU32,HEAPF64,UTF8ToString"
    -sMODULARIZE
    -sEXPORT_ES6
    -sEXPORT_NAME=createJpo
    -sENVIRONMENT=web
    -sALLOW_MEMORY_GROWTH
    -sMAXIMUM_MEMORY=4GB
    -sSTACK_SIZE=1MB)
  foreach(f jpo.js test.html bench.html demo.html demo.js benchmark.html)
    add_custom_command(TARGET jpo_wasm POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       "${CMAKE_CURRENT_SOURCE_DIR}/web/${f}" "${CMAKE_CURRENT_BINARY_DIR}/web/${f}")
  endforeach()
  add_custom_command(TARGET jpo_wasm POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory
                     "${CMAKE_CURRENT_SOURCE_DIR}/web/samples" "${CMAKE_CURRENT_BINARY_DIR}/web/samples")
endif()

if(JPO_BUILD_CLI AND NOT EMSCRIPTEN)
  add_executable(jpo_cli cli/main.cpp cli/bench.cpp)
  set_target_properties(jpo_cli PROPERTIES OUTPUT_NAME jpo)
  target_include_directories(jpo_cli PRIVATE src ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(jpo_cli PRIVATE jpo)
  if(JPO_INSTALL)
    install(TARGETS jpo_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
endif()

# PoseLib: the reference tool and the parity tests. Test-only; the library does not use it.
set(jpo_poselib "${CMAKE_CURRENT_SOURCE_DIR}/third_party/poselib")
set(jpo_eigen "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen")
if(NOT JPO_REFERENCE STREQUAL "OFF" AND NOT EMSCRIPTEN)
  if(EXISTS "${jpo_poselib}/PoseLib/robust.cc" AND EXISTS "${jpo_eigen}/Eigen/Dense")
    set(JPO_HAVE_REFERENCE ON)
  elseif(JPO_REFERENCE STREQUAL "ON")
    message(FATAL_ERROR "JPO_REFERENCE=ON but third_party/poselib or third_party/eigen is missing (tools/fetch_reference.sh)")
  endif()
endif()
if(JPO_HAVE_REFERENCE)
  file(GLOB_RECURSE poselib_sources "${jpo_poselib}/PoseLib/*.cc")
  add_library(poselib_ref STATIC ${poselib_sources})
  target_include_directories(poselib_ref SYSTEM PUBLIC "${jpo_poselib}" "${jpo_eigen}")
  target_compile_options(poselib_ref PRIVATE -w)
  add_executable(poselib_ref_tool tools/poselib_ref.cc)
  set_target_properties(poselib_ref_tool PROPERTIES OUTPUT_NAME poselib_ref)
  target_include_directories(poselib_ref_tool PRIVATE include ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(poselib_ref_tool PRIVATE poselib_ref Threads::Threads)
endif()

if(JPO_BUILD_TESTS AND NOT EMSCRIPTEN)
  enable_testing()
  # Solvers, cameras and refinement on generated data; the whole estimators against known poses.
  add_executable(core_test tests/core_test.cpp)
  target_include_directories(core_test PRIVATE src ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(core_test PRIVATE jpo)
  add_test(NAME core COMMAND core_test)
  # The float build of the kernels against the double build (PLAN.md, "Precision").
  add_executable(precision_test tests/precision_test.cpp)
  target_include_directories(precision_test PRIVATE src ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(precision_test PRIVATE jpo)
  add_test(NAME precision COMMAND precision_test)
  if(JPO_HAVE_REFERENCE)
    # Each solver against PoseLib's on the same samples.
    add_executable(parity_test tests/parity_test.cpp)
    target_include_directories(parity_test PRIVATE src ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(parity_test PRIVATE jpo poselib_ref)
    add_test(NAME poselib_parity COMMAND parity_test)
  endif()
  # A project outside this tree finds the installed package and solves a problem through it.
  if(JPO_INSTALL)
    add_test(NAME package
             COMMAND ${CMAKE_COMMAND} "-DJPO_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}" "-DJPO_CONFIG=$<CONFIG>"
                     "-DJPO_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tests/package"
                     "-DJPO_GENERATOR=${CMAKE_GENERATOR}"
                     -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/package/run.cmake")
  endif()
  # The incremental reconstruction test (needs just-ba; see tests/incremental/CMakeLists.txt).
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/incremental/CMakeLists.txt")
    add_subdirectory(tests/incremental)
  endif()
  set(gpu_backends ${jpo_backends})
  list(REMOVE_ITEM gpu_backends cpu)
  if(gpu_backends)
    # Each GPU backend against the float CPU build: the same samples, the same best hypothesis.
    add_executable(backend_test tests/backend_test.cpp)
    target_include_directories(backend_test PRIVATE src ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(backend_test PRIVATE jpo)
    foreach(b CUDA METAL WEBGPU)
      if(JPO_HAVE_${b})
        target_compile_definitions(backend_test PRIVATE JPO_HAVE_${b})  # the internal factories (backend.h)
      endif()
    endforeach()
    foreach(backend IN LISTS gpu_backends)
      add_test(NAME ${backend}_vs_cpu COMMAND backend_test ${backend})
    endforeach()
    if(JPO_HAVE_WEBGPU)
      add_executable(webgpu_shared_test tests/webgpu_shared_test.cpp)
      target_link_libraries(webgpu_shared_test PRIVATE jpo)
      add_test(NAME webgpu_shared_device COMMAND webgpu_shared_test)
    endif()
  endif()
endif()

# The Python module (bindings/python), after the tests so that its own test is registered with them.
if(JPO_BUILD_PYTHON AND NOT EMSCRIPTEN)
  add_subdirectory(bindings/python)
endif()
