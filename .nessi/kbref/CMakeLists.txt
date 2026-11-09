# kbref: generators of plain-text NESSi reference data for KadanoffBaym.
#
# Build (from the repository root):
#   cmake -S .nessi/kbref -B .nessi/kbref/build
#   cmake --build .nessi/kbref/build --target <target>
#
# Prerequisites (see .omo/plans/01-phase1-foundation.md, section 3.2):
#   .nessi/install         NESSi libcntr (hdf5/omp/mpi/ness OFF)
#   .nessi/eigen3-install  Eigen3 install prefix
cmake_minimum_required(VERSION 3.10)
project(kbref CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

get_filename_component(KBREF_NESSI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(KBREF_CNTR_PREFIX "${KBREF_NESSI_DIR}/install")
set(KBREF_EIGEN_PREFIX "${KBREF_NESSI_DIR}/eigen3-install")
set(KBREF_CNTR_LIB "${KBREF_CNTR_PREFIX}/lib/libcntr.so")

if(NOT EXISTS "${KBREF_CNTR_LIB}")
    message(FATAL_ERROR
        "kbref: ${KBREF_CNTR_LIB} not found. Build and install NESSi first:\n"
        "  cmake -S .nessi/nessi-2.0.0/libcntr -B .nessi/build ... && "
        "cmake --build .nessi/build && cmake --install .nessi/build")
endif()
if(NOT EXISTS "${KBREF_CNTR_PREFIX}/include/cntr/cntr.hpp")
    message(FATAL_ERROR
        "kbref: ${KBREF_CNTR_PREFIX}/include/cntr/cntr.hpp not found. "
        "Run: cmake --install .nessi/build")
endif()
if(NOT EXISTS "${KBREF_EIGEN_PREFIX}/include/eigen3/Eigen/Dense")
    message(FATAL_ERROR
        "kbref: Eigen3 headers not found in ${KBREF_EIGEN_PREFIX}/include. "
        "Run: cmake --install .nessi/eigen3-build")
endif()

# NESSi uses #include <eigen3/Eigen/Dense>, so the plain include prefix of
# Eigen is required. The NESSi headers include each other without the
# cntr/ prefix, so include/cntr is added as well.
set(KBREF_INCLUDES
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${KBREF_CNTR_PREFIX}/include"
    "${KBREF_CNTR_PREFIX}/include/cntr"
    "${KBREF_EIGEN_PREFIX}/include")

set(CMAKE_BUILD_RPATH "${KBREF_CNTR_PREFIX}/lib")

add_executable(weights weights.cpp)
target_include_directories(weights PRIVATE ${KBREF_INCLUDES})
target_link_libraries(weights PRIVATE "${KBREF_CNTR_LIB}")

add_executable(convolution convolution.cpp)
target_include_directories(convolution PRIVATE ${KBREF_INCLUDES})
target_compile_definitions(convolution PRIVATE CNTR_NO_EXTERN_TEMPLATES)
target_link_libraries(convolution PRIVATE "${KBREF_CNTR_LIB}")
