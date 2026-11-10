cmake_minimum_required(VERSION 3.16)

# Default compiler: GCC 15 (the c++ on PATH if it is GCC 15, else g++-15),
# unless -DCMAKE_CXX_COMPILER or CXX selects another one.
if(NOT CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX})
  set(_default_is_gcc15 FALSE)
  find_program(_default_cxx NAMES c++)
  if(_default_cxx)
    execute_process(COMMAND "${_default_cxx}" --version
                    OUTPUT_VARIABLE _cxx_banner ERROR_QUIET)
    execute_process(COMMAND "${_default_cxx}" -dumpversion
                    OUTPUT_VARIABLE _cxx_version ERROR_QUIET)
    if(NOT _cxx_banner MATCHES "clang" AND _cxx_version MATCHES "^15(\\.|[ \t\r\n]*$)")
      set(_default_is_gcc15 TRUE)
    endif()
  endif()
  if(NOT _default_is_gcc15)
    find_program(_gxx15 NAMES g++-15)
    if(_gxx15)
      set(CMAKE_CXX_COMPILER "${_gxx15}" CACHE FILEPATH "C++ compiler")
    else()
      message(WARNING "GCC 15 not found; using the system default C++ compiler.")
    endif()
  endif()
endif()

project(UltraKMeans VERSION 0.1.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ULTRA_KMEANS_NATIVE "Tune the binary for the host CPU (-march=native)" ON)

find_package(Threads REQUIRED)

# Dependencies are git submodules: git submodule update --init
set(ULTRA_KMEANS_THIRD_PARTY "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
foreach(header
    parlaylib/include/parlay/parallel.h
    ParlayANN/algorithms/vamana/index.h
    CPAM/include/pam/pam.h
    eigen/Eigen/Core)
  if(NOT EXISTS "${ULTRA_KMEANS_THIRD_PARTY}/${header}")
    message(FATAL_ERROR
      "third_party/${header} is missing; run: git submodule update --init")
  endif()
endforeach()

add_executable(ultra_kmeans lloyds/kmeans.cpp)

target_include_directories(ultra_kmeans PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(ultra_kmeans SYSTEM PRIVATE
  ${ULTRA_KMEANS_THIRD_PARTY}/parlaylib/include
  ${ULTRA_KMEANS_THIRD_PARTY}/ParlayANN
  ${ULTRA_KMEANS_THIRD_PARTY}/CPAM/include
  ${ULTRA_KMEANS_THIRD_PARTY}/eigen)

# Same flags as .bazelrc.
target_compile_definitions(ultra_kmeans PRIVATE PARLAY_USE_STD_ALLOC)
target_compile_options(ultra_kmeans PRIVATE -O3 -mcx16 -fno-plt -UNDEBUG)
if(ULTRA_KMEANS_NATIVE)
  target_compile_options(ultra_kmeans PRIVATE -march=native)
endif()

target_link_libraries(ultra_kmeans PRIVATE Threads::Threads)
