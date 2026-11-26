# Reference data generators: plain-text NESSi reference data for
# KadanoffBaym.
#
# Build (from the repository root):
#   cmake -S .nessi/kbref -B .nessi/kbref/build
#   cmake --build .nessi/kbref/build --target <target>
#
# Prerequisites:
#   .nessi/install         NESSi libcntr (hdf5/omp/mpi OFF, ness ON)
#   .nessi/eigen3-install  Eigen3 install prefix
cmake_minimum_required(VERSION 3.10)
project(reference CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

get_filename_component(REFERENCE_NESSI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(REFERENCE_CNTR_PREFIX "${REFERENCE_NESSI_DIR}/install")
set(REFERENCE_EIGEN_PREFIX "${REFERENCE_NESSI_DIR}/eigen3-install")
set(REFERENCE_CNTR_LIB "${REFERENCE_CNTR_PREFIX}/lib/libcntr.so")

if(NOT EXISTS "${REFERENCE_CNTR_LIB}")
    message(FATAL_ERROR
        "reference: ${REFERENCE_CNTR_LIB} not found. Build and install NESSi first:\n"
        "  cmake -S .nessi/nessi-2.0.0/libcntr -B .nessi/build ... && "
        "cmake --build .nessi/build && cmake --install .nessi/build")
endif()
if(NOT EXISTS "${REFERENCE_CNTR_PREFIX}/include/cntr/cntr.hpp")
    message(FATAL_ERROR
        "reference: ${REFERENCE_CNTR_PREFIX}/include/cntr/cntr.hpp not found. "
        "Run: cmake --install .nessi/build")
endif()
if(NOT EXISTS "${REFERENCE_EIGEN_PREFIX}/include/eigen3/Eigen/Dense")
    message(FATAL_ERROR
        "reference: Eigen3 headers not found in ${REFERENCE_EIGEN_PREFIX}/include. "
        "Run: cmake --install .nessi/eigen3-build")
endif()

# NESSi uses #include <eigen3/Eigen/Dense>, so the plain include prefix of
# Eigen is required. The NESSi headers include each other without the
# cntr/ prefix, so include/cntr is added as well.
set(REFERENCE_INCLUDES
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${REFERENCE_CNTR_PREFIX}/include"
    "${REFERENCE_CNTR_PREFIX}/include/cntr"
    "${REFERENCE_EIGEN_PREFIX}/include")

set(CMAKE_BUILD_RPATH "${REFERENCE_CNTR_PREFIX}/lib")

add_executable(weights weights.cpp)
target_include_directories(weights PRIVATE ${REFERENCE_INCLUDES})
target_link_libraries(weights PRIVATE "${REFERENCE_CNTR_LIB}")

add_executable(convolution convolution.cpp)
target_include_directories(convolution PRIVATE ${REFERENCE_INCLUDES})
target_compile_definitions(convolution PRIVATE CNTR_NO_EXTERN_TEMPLATES)
target_link_libraries(convolution PRIVATE "${REFERENCE_CNTR_LIB}")

# Tool generators
foreach(target utilities differentiation convolution_utilities response)
    add_executable(${target} ${target}.cpp)
    target_include_directories(${target} PRIVATE ${REFERENCE_INCLUDES})
    target_compile_definitions(${target} PRIVATE CNTR_NO_EXTERN_TEMPLATES)
    target_link_libraries(${target} PRIVATE "${REFERENCE_CNTR_LIB}")
endforeach()

# Dyson generators
foreach(target dyson_ref equilibrium_ref)
    add_executable(${target} ${target}.cpp)
    target_include_directories(${target} PRIVATE ${REFERENCE_INCLUDES})
    target_compile_definitions(${target} PRIVATE CNTR_NO_EXTERN_TEMPLATES)
    target_link_libraries(${target} PRIVATE "${REFERENCE_CNTR_LIB}")
endforeach()

# Real-time generators
foreach(target vie2 bubble observables)
    add_executable(${target} ${target}.cpp)
    target_include_directories(${target} PRIVATE ${REFERENCE_INCLUDES})
    target_compile_definitions(${target} PRIVATE CNTR_NO_EXTERN_TEMPLATES)
    target_link_libraries(${target} PRIVATE "${REFERENCE_CNTR_LIB}")
endforeach()

# Pseudo-particle generator
add_executable(pseudo_reference pseudo_reference.cpp)
target_include_directories(pseudo_reference PRIVATE ${REFERENCE_INCLUDES})
target_compile_definitions(pseudo_reference PRIVATE CNTR_NO_EXTERN_TEMPLATES)
target_link_libraries(pseudo_reference PRIVATE "${REFERENCE_CNTR_LIB}")

# Steady-state and example generators. They
# require libcntr built with ness=ON and the system FFTW.
find_library(REFERENCE_FFTW_LIB NAMES fftw3 REQUIRED)
foreach(target fourier_ref steady_ref examples_siam_ref
        examples_bethe_ref examples_bethe_prop_ref)
    add_executable(${target} ${target}.cpp)
    target_include_directories(${target} PRIVATE ${REFERENCE_INCLUDES})
    target_compile_definitions(${target} PRIVATE CNTR_NO_EXTERN_TEMPLATES CNTR_USE_NESS)
    target_link_libraries(${target} PRIVATE "${REFERENCE_CNTR_LIB}" "${REFERENCE_FFTW_LIB}")
endforeach()

# Moving-window and example generators. The
# moving Dyson solver of the installed libcntr is cntr_dyson_moving_impl_org.hpp.
foreach(target trunc_bethe_ref moving_solver_ref moving_ref gw_ref)
    add_executable(${target} ${target}.cpp)
    target_include_directories(${target} PRIVATE ${REFERENCE_INCLUDES})
    target_compile_definitions(${target} PRIVATE CNTR_NO_EXTERN_TEMPLATES)
    target_link_libraries(${target} PRIVATE "${REFERENCE_CNTR_LIB}")
endforeach()

# Example generator with NESS2 (needs ness=ON and FFTW, see the steady-state
# block)
add_executable(steady_bethe_trunc_ref steady_bethe_trunc_ref.cpp)
target_include_directories(steady_bethe_trunc_ref PRIVATE ${REFERENCE_INCLUDES})
target_compile_definitions(steady_bethe_trunc_ref PRIVATE CNTR_NO_EXTERN_TEMPLATES CNTR_USE_NESS)
target_link_libraries(steady_bethe_trunc_ref PRIVATE "${REFERENCE_CNTR_LIB}" "${REFERENCE_FFTW_LIB}")

# Example generators. They
# link only libcntr and, where needed, the physics implementation of the
# NESSi examples (no HDF5, MPI, or OpenMP). A target whose source file is
# not written yet is reported as pending.
set(REFERENCE_EXAMPLES "${CMAKE_CURRENT_SOURCE_DIR}/../nessi-2.0.0/examples/programs")
set(EXAMPLE_TARGETS hubbard_chain_text fkm_bethe_quench_text holstein_bethe_text
    holstein_nambu_text holstein_impurity_text)
set(EXAMPLE_IMPL_hubbard_chain_text "${REFERENCE_EXAMPLES}/hubbard_chain_selfen_impl.cpp")
set(EXAMPLE_IMPL_fkm_bethe_quench_text "")
set(EXAMPLE_IMPL_holstein_bethe_text "${REFERENCE_EXAMPLES}/Holstein_impurity_impl.cpp;${REFERENCE_EXAMPLES}/Holstein_utils_impl.cpp")
set(EXAMPLE_IMPL_holstein_nambu_text "${EXAMPLE_IMPL_holstein_bethe_text}")
set(EXAMPLE_IMPL_holstein_impurity_text "${EXAMPLE_IMPL_holstein_bethe_text}")
foreach(target ${EXAMPLE_TARGETS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${target}.cpp")
        add_executable(${target} ${target}.cpp ${EXAMPLE_IMPL_${target}})
        # The extern templates of libcntr are used, as in the NESSi example
        # build (with CNTR_NO_EXTERN_TEMPLATES cntr_response_convolution_impl.hpp
        # undefines the GREEN macro needed by the example headers)
        target_include_directories(${target} PRIVATE ${REFERENCE_INCLUDES} "${REFERENCE_EXAMPLES}"
                                   "${REFERENCE_EIGEN_PREFIX}/include/eigen3")
        target_link_libraries(${target} PRIVATE "${REFERENCE_CNTR_LIB}")
        # Drive files of the tutor directories
        target_compile_definitions(${target} PRIVATE
                                   REFERENCE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/../..")
    else()
        message(STATUS "reference: example target ${target} pending (no source file)")
    endif()
endforeach()
